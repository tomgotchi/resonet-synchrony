"""Pair selection and the per-window trading simulation.

Causality contract: every decision at bar t uses prices with index <= t only, and is executed `exec_delay_bars`
later. Pair parameters (alpha, beta, mean, sd, AR(1)) are frozen at the end of the formation window.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibrate import calibrate_pair
from .coint import ar1_fit, benjamini_hochberg, half_life_corrected, ols_alpha_beta, symmetric_eg
from .config import StrategyConfig


@dataclass(frozen=True)
class Pair:
    a: str           # traded as  +1 unit of A  vs  -beta units (in dollars) of B
    b: str
    alpha: float
    beta: float
    mu: float        # formation mean of the spread
    sd: float        # formation sd of the spread
    ar_c: float
    ar_phi: float
    ar_sigma: float
    half_life: float
    p: float
    tstat: float
    z_scale: float = 1.0       # selection-aware calibration: z = (s - mu) / (sd * z_scale)
    cusum_h: float = float("nan")
    group: str = ""
    p_threshold: float = 1.0   # the BH cut-off the pair passed (needed to re-run its calibration)


def spread(la, lb, pair: Pair):
    return la - pair.alpha - pair.beta * lb


def candidate_pairs(tickers, sectors: dict | None):
    tickers = sorted(tickers)                       # canonical order: A/B swap cannot change anything
    for a, b in itertools.combinations(tickers, 2):
        if sectors is None or (sectors.get(a) is not None and sectors.get(a) == sectors.get(b)):
            yield a, b


def select_pairs(formation: pd.DataFrame, cfg: StrategyConfig, sectors: dict | None = None,
                 return_table: bool = False):
    """Choose pairs on the formation window only. Columns with any missing price are not eligible."""
    px = formation.dropna(axis=1, how="any")
    px = px.loc[:, (px > 0).all()]
    logp = np.log(px)
    rows = []
    for a, b in candidate_pairs(px.columns, sectors):
        la, lb = logp[a].to_numpy(), logp[b].to_numpy()
        r = symmetric_eg(la, lb, cfg.adf_lags)
        rows.append({"a": a, "b": b, "group": sectors.get(a, "") if sectors else "", **r})
    table = pd.DataFrame(rows)
    if table.empty:
        return ([], table) if return_table else []
    table["fdr_pass"] = False
    table["p_threshold"] = np.nan
    families = table.groupby("group").groups.items() if cfg.fdr_per_group else [("", table.index)]
    for _, ix in families:                                   # BH within each group (board round 3, D5)
        mask = benjamini_hochberg(table.loc[ix, "p"].to_numpy(), cfg.fdr_q)
        table.loc[ix, "fdr_pass"] = mask
        if mask.any():
            table.loc[ix, "p_threshold"] = table.loc[ix, "p"].to_numpy()[mask].max()
    chosen, used = [], set()
    for _, row in table[table["fdr_pass"]].sort_values(["p", "a", "b"]).iterrows():
        if not cfg.beta_min <= row.beta <= cfg.beta_max:
            continue
        if cfg.one_pair_per_ticker and (row.a in used or row.b in used):
            continue
        p = make_pair(logp[row.a].to_numpy(), logp[row.b].to_numpy(), row.a, row.b, cfg,
                      p_threshold=float(row.p_threshold), p=float(row.p), tstat=float(row.tstat),
                      group=str(row.group))
        if not cfg.half_life_min <= p.half_life <= cfg.half_life_max:
            continue
        chosen.append(p)
        used.update([row.a, row.b])
        if len(chosen) >= cfg.max_pairs:
            break
    return (chosen, table) if return_table else chosen


def make_pair(la, lb, a, b, cfg, p_threshold=1.0, p=1.0, tstat=0.0, group="") -> Pair:
    """Fit a pair on formation log prices and calibrate it (half-life, z scale, CUSUM h)."""
    alpha, beta = ols_alpha_beta(la, lb)
    s = la - alpha - beta * lb
    c, phi, sig = ar1_fit(s)
    hl = half_life_corrected(phi, len(s))
    z_scale, h = 1.0, cfg.cusum_h
    if cfg.calibrate and np.isfinite(hl) and hl > 0:
        seed = int.from_bytes(f"{a}|{b}|{len(s)}|{la[-1]:.10f}".encode()[:8].ljust(8, b"\0"), "little")
        cal = calibrate_pair(la, lb, p_threshold, cfg, seed=seed % (2 ** 32), n_sims=cfg.calibration_sims)
        hl, z_scale, h = cal["half_life"], cal["z_scale"], cal["cusum_h"]
    return Pair(a, b, alpha, beta, float(s.mean()), float(s.std(ddof=1)), c, phi, sig, float(hl), p, tstat,
                float(z_scale), float(h), group, float(p_threshold))


@dataclass
class Trade:
    pair: str
    side: int              # +1 = long spread (long A, short B), -1 = short spread
    entry_signal: pd.Timestamp
    entry_fill: pd.Timestamp
    exit_fill: pd.Timestamp | None = None
    exit_reason: str = ""
    pnl: float = 0.0       # net of costs and borrow, dollars
    costs: float = 0.0
    entry_z: float = 0.0
    features: dict | None = None


def trade_window(prices: pd.DataFrame, pairs: list[Pair], start: int, end: int, cfg: StrategyConfig,
                 entry_filter=None, final: bool = True, open_state: dict | None = None,
                 dividends: pd.DataFrame | None = None) -> tuple[pd.Series, list[Trade]]:
    """Simulate trading bars [start, end) of `prices` (integer positions). All positions are flat at `end`.

    Returns daily net P&L in dollars (index = prices.index[start:end]) and the list of trades.
    `entry_filter(pair, t, features) -> bool` may veto entries (used by the optional ML filter).
    final=False (paper trading mid-window): do not flatten at the last bar; instead write each pair's open
    position and the decision taken at the last close into `open_state`.
    `dividends`: per-share cash dividends on ex-dates (same index/columns as prices, 0 elsewhere). Prices are
    then raw (split-adjusted only) closes; long legs receive and short legs pay the dividend.
    """
    idx = prices.index
    n_bars = len(idx)
    pnl = np.zeros(end - start)
    trades: list[Trade] = []
    if not pairs:
        return pd.Series(pnl, index=idx[start:end]), trades
    per_pair_gross = cfg.capital / cfg.max_pairs
    cost_rate = cfg.cost_bps / 1e4
    borrow_daily = cfg.borrow_bps_per_year / 1e4 / 252.0
    d = cfg.exec_delay_bars
    last_bar = min(end, n_bars) - 1

    for pair in pairs:
        A = prices[pair.a].to_numpy(float)
        B = prices[pair.b].to_numpy(float)
        if dividends is not None:
            DA = dividends[pair.a].reindex(idx).fillna(0.0).to_numpy(float) if pair.a in dividends else 0 * A
            DB = dividends[pair.b].reindex(idx).fillna(0.0).to_numpy(float) if pair.b in dividends else 0 * B
        else:
            DA = DB = np.zeros(len(A))
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.log(A) - pair.alpha - pair.beta * np.log(B)
        z = (s - pair.mu) / (pair.sd * pair.z_scale)
        cusum_h = pair.cusum_h if np.isfinite(pair.cusum_h) else cfg.cusum_h
        nu = np.full_like(s, np.nan)
        nu[1:] = (s[1:] - pair.ar_c - pair.ar_phi * s[:-1]) / pair.ar_sigma
        nu = np.clip(nu, -cfg.cusum_clip, cfg.cusum_clip)     # Huber clip: one fat-tailed day cannot alarm
        max_hold = int(math.ceil(cfg.time_stop_half_lives * pair.half_life))

        qa = qb = 0.0          # shares held
        side = 0
        pending = None         # (fill_bar, action, side, reason)
        trade = None
        held_since = None
        cus_p = cus_m = 0.0
        broken = False
        name = f"{pair.a}/{pair.b}"
        last_valid = {"A": np.nan, "B": np.nan}

        for t in range(start, last_bar + 1):
            a_t, b_t = A[t], B[t]
            valid = np.isfinite(a_t) and np.isfinite(b_t) and a_t > 0 and b_t > 0
            # 1) mark to market from t-1 to t (positions held over the bar)
            if side != 0 and t > start:
                if valid and np.isfinite(last_valid["A"]):
                    day = qa * (a_t - last_valid["A"]) + qb * (b_t - last_valid["B"])
                    short_notional = (-qa * a_t if qa < 0 else 0.0) + (-qb * b_t if qb < 0 else 0.0)
                    day -= borrow_daily * short_notional
                    day += qa * DA[t] + qb * DB[t]                     # dividends: long receives, short pays
                    pnl[t - start] += day
                    trade.pnl += day
            if valid:
                last_valid = {"A": a_t, "B": b_t}

            # 2) execute an order scheduled for this bar
            def fill(new_side, reason):
                nonlocal qa, qb, side, trade, held_since
                pa, pb = last_valid["A"], last_valid["B"]
                if side != 0:                                        # close existing
                    c = cost_rate * (abs(qa) * pa + abs(qb) * pb)
                    pnl[t - start] -= c
                    trade.pnl -= c
                    trade.costs += c
                    trade.exit_fill, trade.exit_reason = idx[t], reason
                    trades.append(trade)
                    qa = qb = 0.0
                    side, trade, held_since = 0, None, None
                if new_side != 0:
                    n_a = per_pair_gross / (1.0 + pair.beta)            # dollars in A; beta * n_a in B
                    qa = new_side * n_a / pa
                    qb = -new_side * pair.beta * n_a / pb
                    c = cost_rate * (abs(qa) * pa + abs(qb) * pb)
                    pnl[t - start] -= c
                    side, held_since = new_side, t
                    trade = Trade(name, new_side, pending[4], idx[t], pnl=-c, costs=c, entry_z=pending[5],
                                  features=pending[6])

            if pending is not None and pending[0] == t:
                if pending[1] == "enter":
                    if valid:                                         # never open on a missing/stale price
                        fill(pending[2], pending[3])
                    pending = None
                elif valid:
                    fill(0, pending[3])
                    pending = None
                else:                                                 # halted: wait for the first real price
                    pending = (t + 1,) + pending[1:]

            if t == last_bar and final:                               # window end: flatten now
                if side != 0:
                    if not valid:                                     # still no price: assume a bad exit
                        h = cfg.delist_haircut
                        new = {"A": last_valid["A"] * (1 - h * np.sign(qa)),
                               "B": last_valid["B"] * (1 - h * np.sign(qb))}
                        bad = qa * (new["A"] - last_valid["A"]) + qb * (new["B"] - last_valid["B"])
                        last_valid = new
                        pnl[t - start] += bad
                        trade.pnl += bad
                    fill(0, "window_end" if valid else "no_price_at_end")
                break

            # 3) decide at the close of bar t (only data <= t)
            if pending is not None:
                continue
            if not valid:
                if side != 0:                                         # missing data: get out, log it
                    pending = (t + d, "exit", 0, "missing_data", idx[t], np.nan, None)
                continue
            if np.isfinite(nu[t]) and not broken:
                cus_p = max(0.0, cus_p + nu[t] - cfg.cusum_k)
                cus_m = max(0.0, cus_m - nu[t] - cfg.cusum_k)
                if max(cus_p, cus_m) > cusum_h:
                    broken = True
            zt = z[t]
            if side == 0:
                if final and t + d >= last_bar:                       # an entry must be held at least a bar
                    continue
                if broken or abs(zt) <= cfg.entry_z or abs(zt) >= cfg.stop_z:
                    continue
                new_side = -1 if zt > 0 else 1
                feats = {"abs_z": abs(zt), "half_life": pair.half_life, "tstat": pair.tstat, "beta": pair.beta,
                         "sd": pair.sd, "cusum": max(cus_p, cus_m), "p": pair.p}
                if entry_filter is not None and not entry_filter(pair, t, feats):
                    continue
                pending = (t + d, "enter", new_side, "", idx[t], zt, feats)
            else:
                reason = ""
                if broken:
                    reason = "break"
                elif side * zt >= -cfg.exit_z:
                    reason = "revert"
                elif abs(zt) >= cfg.stop_z:
                    reason = "stop"
                elif t - held_since >= max_hold:
                    reason = "time"
                if reason and not (final and t + d > last_bar):
                    pending = (t + d, "exit", 0, reason, idx[t], zt, None)
        if not final and open_state is not None:
            open_state[name] = {
                "side": side, "qa": qa, "qb": qb,
                "entry_fill": str(trade.entry_fill.date()) if trade else None,
                "z": float(z[last_bar]) if np.isfinite(z[last_bar]) else None,
                "broken": broken,
                "next_action": None if pending is None else {"action": pending[1], "side": pending[2],
                                                             "reason": pending[3]},
            }
    return pd.Series(pnl, index=idx[start:end]), trades
