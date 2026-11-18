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

from .coint import ar1_fit, benjamini_hochberg, half_life_corrected, symmetric_eg
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
        rows.append({"a": a, "b": b, **r})
    table = pd.DataFrame(rows)
    if table.empty:
        return ([], table) if return_table else []
    table["fdr_pass"] = benjamini_hochberg(table["p"].to_numpy(), cfg.fdr_q)
    chosen, used = [], set()
    for _, row in table[table["fdr_pass"]].sort_values(["p", "a", "b"]).iterrows():
        if not cfg.beta_min <= row.beta <= cfg.beta_max:
            continue
        if cfg.one_pair_per_ticker and (row.a in used or row.b in used):
            continue
        s = logp[row.a].to_numpy() - row.alpha - row.beta * logp[row.b].to_numpy()
        c, phi, sig = ar1_fit(s)
        hl = half_life_corrected(phi, len(s))
        if not cfg.half_life_min <= hl <= cfg.half_life_max:
            continue
        chosen.append(Pair(row.a, row.b, row.alpha, row.beta, float(s.mean()), float(s.std(ddof=1)),
                           c, phi, sig, hl, float(row.p), float(row.tstat)))
        used.update([row.a, row.b])
        if len(chosen) >= cfg.max_pairs:
            break
    return (chosen, table) if return_table else chosen


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
                 entry_filter=None, final: bool = True, open_state: dict | None = None
                 ) -> tuple[pd.Series, list[Trade]]:
    """Simulate trading bars [start, end) of `prices` (integer positions). All positions are flat at `end`.

    Returns daily net P&L in dollars (index = prices.index[start:end]) and the list of trades.
    `entry_filter(pair, t, features) -> bool` may veto entries (used by the optional ML filter).
    final=False (paper trading mid-window): do not flatten at the last bar; instead write each pair's open
    position and the decision taken at the last close into `open_state`.
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
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.log(A) - pair.alpha - pair.beta * np.log(B)
        z = (s - pair.mu) / pair.sd
        nu = np.full_like(s, np.nan)
        nu[1:] = (s[1:] - pair.ar_c - pair.ar_phi * s[:-1]) / pair.ar_sigma
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
                elif np.isfinite(last_valid["A"]):                    # exits use the last valid price
                    fill(0, pending[3])
                pending = None

            if t == last_bar and final:                               # window end: flatten now
                if side != 0:
                    fill(0, "window_end")
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
                if max(cus_p, cus_m) > cfg.cusum_h:
                    broken = True
            zt = z[t]
            if final and t + d > last_bar:
                continue
            if side == 0:
                if broken or abs(zt) <= cfg.entry_z or abs(zt) >= cfg.stop_z:
                    continue
                new_side = -1 if zt > 0 else 1
                feats = {"abs_z": abs(zt), "half_life": pair.half_life, "tstat": pair.tstat, "beta": pair.beta,
                         "sd": pair.sd, "cusum": max(cus_p, cus_m)}
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
                if reason:
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
