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
        lo, hi = block_bootstrap_sharpe_ci(r, block=126)          # whole trading windows (board round 2)
        out["sharpe_ci95"] = (lo, hi)
        out["sharpe_ci95_block63"] = block_bootstrap_sharpe_ci(r, block=63)
        out["sharpe_ci95_block252"] = block_bootstrap_sharpe_ci(r, block=252)
        return out


def walk_forward(prices: pd.DataFrame, cfg: StrategyConfig, sectors: dict | None = None,
                 pair_picker=None, entry_filter_factory=None, dividends: pd.DataFrame | None = None
                 ) -> BacktestResult:
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
        pnl, tr = trade_window(prices, pairs, s, e, cfg, entry_filter=flt, dividends=dividends)
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
    N = distinct configurations in the ledger for this universe/date range. The variance of trial Sharpes is
    floored at the null sampling variance (1/years, annual units) so re-running near-identical configs cannot
    shrink the hurdle to zero (board round 2)."""
    N = len(trial_sharpes_annual)
    if N < 2:
        return {"n_trials": N, "sr0_annual": 0.0, "dsr": probabilistic_sharpe(r, 0.0),
                "note": "fewer than 2 distinct configurations logged: plain PSR vs 0, not a deflated Sharpe"}
    years = max(len(pd.Series(r).dropna()) / TRADING_DAYS, 1e-9)
    v_annual = max(float(np.var(np.asarray(trial_sharpes_annual), ddof=1)), 1.0 / years)
    v = v_annual / TRADING_DAYS
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    sr0_annual = float(sr0 * np.sqrt(TRADING_DAYS))
    return {"n_trials": N, "sr0_annual": sr0_annual, "dsr": probabilistic_sharpe(r, sr0_annual)}


# ------------------------------------------------------------------------------------------------ baselines
def random_pair_picker(seed: int):
    """Matched placebo (board round 3, Zhao D2): as many pairs as the real rule picks, drawn at random from
    same-group candidates that pass the same beta and (unclipped) half-life screens but were NOT significant
    (0.05 < p < 0.5). They get the same calibration and the same trading rules. If the real strategy cannot beat
    these, the cointegration screen is not what earns the money."""
    from .strategy import make_pair

    def pick(formation, cfg, sectors, w):
        real, table = select_pairs(formation, cfg, sectors, return_table=True)
        if not real or table.empty:
            return []
        px = formation.dropna(axis=1, how="any")
        logp = np.log(px)
        cands = table[(table.p > 0.05) & (table.p < 0.5) & table.beta.between(cfg.beta_min, cfg.beta_max)]
        order = np.random.default_rng([seed, w]).permutation(len(cands))
        out, used = [], set()
        for i in order:
            row = cands.iloc[i]
            if len(out) >= len(real):
                break
            if row.a in used or row.b in used:
                continue
            p = make_pair(logp[row.a].to_numpy(), logp[row.b].to_numpy(), row.a, row.b, cfg, p_threshold=1.0,
                          p=float(row.p), tstat=float(row.tstat), group=str(row.group))
            if not cfg.half_life_min <= p.half_life <= cfg.half_life_max:
                continue
            out.append(p)
            used.update([row.a, row.b])
        return out
    return pick


def placebo_distribution(prices, cfg, sectors, draws: int = 39, dividends=None) -> list[float]:
    """39 draws and 'no draw may tie or beat the real Sharpe' gives an exact one-sided p = 1/40 = 0.025."""
    return [sharpe(walk_forward(prices, cfg, sectors, pair_picker=random_pair_picker(k),
                                dividends=dividends).daily_ret) for k in range(draws)]


def dose_response(result: "BacktestResult") -> pd.DataFrame:
    """Net P&L per trade by the pair's selection p-value bucket: it should fall as p rises."""
    rows = [{"p": t.features.get("p", np.nan) if t.features else np.nan, "pnl": t.pnl} for t in result.trades]
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return df
    df["bucket"] = pd.cut(df.p, [0, 1e-4, 1e-3, 1e-2, 0.05, 1.0])
    return df.groupby("bucket", observed=True).pnl.agg(["count", "mean"]).reset_index()


def stress_table(prices, cfg, sectors=None, dividends=None) -> pd.DataFrame:
    """Mandatory sensitivity: costs x1/x2/x3 and one extra bar of execution delay."""
    rows = []
    for mult in (1, 2, 3):
        for delay in (cfg.exec_delay_bars, cfg.exec_delay_bars + 1):
            c = cfg.with_(cost_bps=cfg.cost_bps * mult, borrow_bps_per_year=cfg.borrow_bps_per_year * mult,
                          exec_delay_bars=delay)
            res = walk_forward(prices, c, sectors, dividends=dividends)
            s = res.summary()
            rows.append({"cost_mult": mult, "delay_bars": delay, "sharpe": s["sharpe"],
                         "ann_return": s["ann_return"], "n_trades": s["n_trades"]})
    return pd.DataFrame(rows)
