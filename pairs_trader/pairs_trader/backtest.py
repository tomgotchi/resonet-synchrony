"""Walk-forward backtest, statistics, baselines and stress tests.

Windows: select on [s - formation, s), trade [s, s + trading), step s by `trading` (non-overlapping trading
windows; every position is closed at the end of its window, so trade labels never cross a window boundary).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .config import StrategyConfig
from .strategy import Pair, Trade, select_pairs, trade_window

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    daily_pnl: pd.Series
    trades: list[Trade]
    windows: list[dict] = field(default_factory=list)
    cfg: StrategyConfig | None = None

    @property
    def daily_ret(self) -> pd.Series:
        return self.daily_pnl / self.cfg.capital

    def summary(self) -> dict:
        r = self.daily_ret
        tr = pd.DataFrame([{"pnl": t.pnl, "reason": t.exit_reason} for t in self.trades])
        out = {
            "days": int(len(r)),
            "years": round(len(r) / TRADING_DAYS, 2),
            "ann_return": float(r.mean() * TRADING_DAYS),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan"),
            "sharpe": sharpe(r),
            "max_drawdown": max_drawdown(r),
            "n_trades": len(self.trades),
            "n_pairs_traded": int(sum(w["n_pairs"] for w in self.windows)),
            "win_rate_net": float((tr.pnl > 0).mean()) if len(tr) else float("nan"),
            "exit_reasons": tr.reason.value_counts().to_dict() if len(tr) else {},
        }
        lo, hi = block_bootstrap_sharpe_ci(r)
        out["sharpe_ci95"] = (lo, hi)
        return out


def walk_forward(prices: pd.DataFrame, cfg: StrategyConfig, sectors: dict | None = None,
                 pair_picker=None, entry_filter_factory=None) -> BacktestResult:
    """`pair_picker(formation_df, cfg, sectors, window_no) -> list[Pair]` replaces the selection rule (placebo).
    `entry_filter_factory(past_trades) -> filter | None` builds the ML filter from trades closed before the window.
    """
    n = len(prices)
    pnl_parts, trades, windows = [], [], []
    s = cfg.formation_days
    w = 0
    while s + 1 < n:
        e = min(s + cfg.trading_days, n)
        formation = prices.iloc[s - cfg.formation_days:s]
        pairs = (pair_picker(formation, cfg, sectors, w) if pair_picker
                 else select_pairs(formation, cfg, sectors))
        flt = entry_filter_factory(list(trades)) if entry_filter_factory else None
        pnl, tr = trade_window(prices, pairs, s, e, cfg, entry_filter=flt)
        pnl_parts.append(pnl)
        trades.extend(tr)
        windows.append({"start": prices.index[s], "end": prices.index[e - 1], "n_pairs": len(pairs),
                        "pairs": [f"{p.a}/{p.b}" for p in pairs], "pnl": float(pnl.sum())})
        s = e
        w += 1
    daily = pd.concat(pnl_parts) if pnl_parts else pd.Series(dtype=float)
    return BacktestResult(daily, trades, windows, cfg)


# ----------------------------------------------------------------------------------------------- statistics
def sharpe(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> float:
    eq = pd.Series(r).fillna(0).cumsum()           # simple (non-compounded) returns on fixed capital
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def block_bootstrap_sharpe_ci(r, block: int = 63, reps: int = 2000, seed: int = 0, level: float = 0.95):
    """Circular block bootstrap of daily returns (board: trades overlap and pairs share regimes, so days are
    the resampling unit, in blocks of ~a quarter)."""
    x = np.asarray(pd.Series(r).dropna(), float)
    n = len(x)
    if n < 2 * block or x.std() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    xx = np.concatenate([x, x[:block]])
    for i in range(reps):
        starts = rng.integers(0, n, nb)
        sample = np.concatenate([xx[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    a = (1 - level) / 2
    return (float(np.quantile(out, a)), float(np.quantile(out, 1 - a)))


def probabilistic_sharpe(r, sr0_annual: float = 0.0) -> float:
    """P(true Sharpe > sr0) given skew/kurtosis of daily returns (Bailey & Lopez de Prado 2012)."""
    x = np.asarray(pd.Series(r).dropna(), float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    sr0 = sr0_annual / np.sqrt(TRADING_DAYS)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r, trial_sharpes_annual: list[float]) -> dict:
    """Deflated Sharpe ratio: PSR against the expected max Sharpe of N trials under the null.
    N and the spread of Sharpes come from the trials ledger (every configuration ever run)."""
    N = len(trial_sharpes_annual)
    if N < 2:
        return {"n_trials": N, "sr0_annual": 0.0, "dsr": probabilistic_sharpe(r, 0.0),
                "note": "fewer than 2 logged trials: this is the plain PSR vs 0"}
    v = float(np.var(np.asarray(trial_sharpes_annual) / np.sqrt(TRADING_DAYS), ddof=1))
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    sr0_annual = float(sr0 * np.sqrt(TRADING_DAYS))
    return {"n_trials": N, "sr0_annual": sr0_annual, "dsr": probabilistic_sharpe(r, sr0_annual)}


# ------------------------------------------------------------------------------------------------ baselines
def random_pair_picker(seed: int):
    """Placebo: same number of pairs as the real rule would pick, drawn at random from the same sectors,
    with parameters fitted on the formation window exactly like real pairs (no significance filter)."""
    from .coint import ar1_fit, half_life_corrected, ols_alpha_beta
    from .strategy import candidate_pairs

    def pick(formation, cfg, sectors, w):
        real = select_pairs(formation, cfg, sectors)
        px = formation.dropna(axis=1, how="any")
        cands = list(candidate_pairs(px.columns, sectors))
        rng = np.random.default_rng([seed, w])
        rng.shuffle(cands)
        out, used = [], set()
        for a, b in cands:
            if len(out) >= len(real):
                break
            if a in used or b in used:
                continue
            la, lb = np.log(px[a].to_numpy()), np.log(px[b].to_numpy())
            al, be = ols_alpha_beta(la, lb)
            if not cfg.beta_min <= be <= cfg.beta_max:
                continue
            sp = la - al - be * lb
            c, phi, sig = ar1_fit(sp)
            hl = min(max(half_life_corrected(phi, len(sp)), cfg.half_life_min), cfg.half_life_max)
            out.append(Pair(a, b, al, be, float(sp.mean()), float(sp.std(ddof=1)), c, phi, sig, hl, 1.0, 0.0))
            used.update([a, b])
        return out
    return pick


def placebo_distribution(prices, cfg, sectors, draws: int = 20) -> list[float]:
    return [sharpe(walk_forward(prices, cfg, sectors, pair_picker=random_pair_picker(k)).daily_ret)
            for k in range(draws)]


def stress_table(prices, cfg, sectors=None) -> pd.DataFrame:
    """Mandatory sensitivity: costs x1/x2/x3 and one extra bar of execution delay."""
    rows = []
    for mult in (1, 2, 3):
        for delay in (cfg.exec_delay_bars, cfg.exec_delay_bars + 1):
            c = cfg.with_(cost_bps=cfg.cost_bps * mult, borrow_bps_per_year=cfg.borrow_bps_per_year * mult,
                          exec_delay_bars=delay)
            res = walk_forward(prices, c, sectors)
            s = res.summary()
            rows.append({"cost_mult": mult, "delay_bars": delay, "sharpe": s["sharpe"],
                         "ann_return": s["ann_return"], "n_trades": s["n_trades"]})
    return pd.DataFrame(rows)
