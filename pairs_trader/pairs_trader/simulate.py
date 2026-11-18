"""Synthetic price panels with KNOWN pair structure, for testing the pipeline.

Every stock loads on one market factor (so unrelated stocks are still correlated, as in real data).
Pair kinds:
  "coint":  log B = market + own random walk;  log A = c + beta * log B + s,  s ~ AR(1)/OU with a known half-life.
  "null":   A and B are independent random walks that share only the market factor (no pair-specific tie).
  "near_null": "coint" with a slow half-life (40-80 days): should usually NOT be selected or should not pay.
  "break":  like "coint" until `break_day`, then the tie breaks (`break_kind`): "unit_root" (spread becomes a
            random walk), "mu_step" (permanent level shift, e.g. M&A news), or "beta_step" (hedge ratio changes).
Realism knobs: Student-t innovations, earnings-style jumps on each leg (permanent, do not revert), and iid
bid-ask "bounce" noise on observed closes (creates spurious lag-1 reversion). Prices start near 50.
Not modelled: volatility clustering (GARCH), delistings, borrow constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelConfig:
    n_days: int = 252 * 6
    kinds: tuple = (("coint", 10), ("null", 30), ("break", 5))
    half_life_range: tuple = (5.0, 30.0)     # days
    spread_sd: float = 0.03                  # stationary sd of the log spread
    beta_range: tuple = (0.6, 1.5)
    market_vol: float = 0.010                # daily log-vol of the market factor
    idio_vol: float = 0.015                  # daily log-vol of each stock's own random walk
    t_dof: float | None = 5.0                # Student-t innovations; None = Gaussian
    break_day: int | None = None             # default: middle of the sample
    break_kind: str = "unit_root"            # "unit_root" | "mu_step" | "beta_step"
    jumps_per_year: float = 0.0              # per leg; e.g. 4 = quarterly earnings
    jump_sd: float = 0.05
    bounce_sd: float = 0.0                   # iid log noise on observed closes
    pairs_per_sector: int = 5                # sectors mix pair kinds at random; screening runs within sectors
    seed: int = 0


@dataclass
class Panel:
    prices: pd.DataFrame                     # index: business days, columns: tickers
    truth: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per planted pair
    sectors: dict = field(default_factory=dict)                 # ticker -> sector label


def _innov(rng, size, dof):
    if dof is None:
        return rng.standard_normal(size)
    return rng.standard_t(dof, size) / np.sqrt(dof / (dof - 2.0))   # unit variance


def simulate_panel(cfg: PanelConfig = PanelConfig()) -> Panel:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days
    dates = pd.bdate_range("2015-01-02", periods=n)
    market = np.cumsum(cfg.market_vol * _innov(rng, n, cfg.t_dof))
    brk = cfg.break_day if cfg.break_day is not None else n // 2
    cols, truth = {}, []
    k = 0
    for kind, count in cfg.kinds:
        for _ in range(count):
            a, b = f"S{k:03d}A", f"S{k:03d}B"
            k += 1
            def walk():
                w = cfg.idio_vol * _innov(rng, n, cfg.t_dof)
                jumps = rng.random(n) < cfg.jumps_per_year / 252.0
                return np.cumsum(w + jumps * rng.normal(0.0, cfg.jump_sd, n))
            log_b = np.log(50) + market + walk()
            if kind == "null":
                log_a = np.log(50) + market + walk()
                hl, beta = np.nan, np.nan
            else:
                hl = rng.uniform(40.0, 80.0) if kind == "near_null" else rng.uniform(*cfg.half_life_range)
                beta = rng.uniform(*cfg.beta_range)
                phi = 0.5 ** (1.0 / hl)
                eps_sd = cfg.spread_sd * np.sqrt(1 - phi ** 2)
                e = eps_sd * _innov(rng, n, cfg.t_dof)
                s = np.zeros(n)
                s[0] = cfg.spread_sd * rng.standard_normal()
                unit_root = kind == "break" and cfg.break_kind == "unit_root"
                for t in range(1, n):
                    if unit_root and t >= brk:
                        s[t] = s[t - 1] + e[t]                 # unit root: the tie is gone
                    else:
                        s[t] = phi * s[t - 1] + e[t]
                beta_t = np.full(n, beta)
                if kind == "break" and cfg.break_kind == "mu_step":
                    s[brk:] += 4.0 * cfg.spread_sd
                if kind == "break" and cfg.break_kind == "beta_step":
                    beta_t[brk:] = beta * 1.3
                # jumps on A that do not revert (idiosyncratic news); B's jumps already live in log_b
                ja = (rng.random(n) < cfg.jumps_per_year / 252.0) * rng.normal(0.0, cfg.jump_sd, n)
                log_a = np.log(50) + beta_t * (log_b - np.log(50)) + s + np.cumsum(ja)
            # bounce noise is always drawn so changing bounce_sd never shifts the RNG stream for later pairs
            na, nb = rng.standard_normal(n), rng.standard_normal(n)
            cols[a] = np.exp(log_a + cfg.bounce_sd * na)
            cols[b] = np.exp(log_b + cfg.bounce_sd * nb)
            truth.append({"a": a, "b": b, "kind": kind, "half_life": hl, "beta": beta,
                          "break_day": dates[brk] if kind == "break" else pd.NaT,
                          "break_kind": cfg.break_kind if kind == "break" else ""})
    truth = pd.DataFrame(truth)
    perm = np.random.default_rng([cfg.seed, 1]).permutation(len(truth))
    truth["sector"] = [f"SEC{i // cfg.pairs_per_sector:02d}" for i in np.argsort(perm)]
    sectors = {}
    for _, r in truth.iterrows():
        sectors[r.a] = sectors[r.b] = r.sector
    return Panel(prices=pd.DataFrame(cols, index=dates), truth=truth, sectors=sectors)
