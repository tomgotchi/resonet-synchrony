"""Study 2 synthetic panels (board 2026-11-19, Volkov round 2 spec). A separate module so simulate.py, and every
Study 1 test that uses it, stays bit-identical.

Per sector (10 planted pairs, 20 tickers; every other within-sector combination is a "cross" null):
  coint x3       sigma_e log-U(25, 200) bp/day, half-life U(3, 30), sd = sigma_e / sqrt(1 - phi^2)
  clone x2       beta U(0.97, 1.03), half-life U(1, 5), sigma_e log-U(1, 4) bp/day   (untradable after costs)
  clone_null x1  clone legs, but the tracking error is a random walk, sigma_e log-U(1, 4) bp/day
  null x2        independent walks sharing only the market factor
  beta_drift x1  log A = c + sum beta_t dlog B + s, beta_t a random walk with sd 0.10 per sqrt(year)
  level_step x1  coint (half of them hl U(3,15), half U(25,45)) plus a 4 sd step on a uniform trading-window day
Market vol 1%/day, idiosyncratic vol 1.5%/day, Student-t(5) innovations, 2 bp bounce on every close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import Panel, _innov

SECTOR_MIX = (("coint", 3), ("clone", 2), ("clone_null", 1), ("null", 2), ("beta_drift", 1), ("level_step", 1))


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def simulate_study2_panel(seed: int, n_sectors: int = 5, formation: int = 504, trading: int = 126,
                          market_vol: float = 0.010, idio_vol: float = 0.015, dof: float = 5.0,
                          bounce_bp: float = 2.0, straddle_per_sector: int = 0) -> Panel:
    n = formation + trading
    dates = pd.bdate_range("2015-01-02", periods=n)
    market = np.cumsum(market_vol * _innov(np.random.default_rng([seed, 3, 0]), n, dof))
    cols, truth, sectors = {}, [], {}
    k = 0
    for sec in range(n_sectors):
        mix = SECTOR_MIX + ((("straddle_null", straddle_per_sector),) if straddle_per_sector else ())
        for kind, count in mix:
            for _ in range(count):
                k += 1
                rng = np.random.default_rng([seed, 3, k])
                a, b = f"S{sec:02d}P{k:04d}A", f"S{sec:02d}P{k:04d}B"
                log_b = np.log(50) + market + np.cumsum(idio_vol * _innov(rng, n, dof))
                hl = beta = sig = np.nan
                step_day = -1
                bb = bounce_bp
                if kind == "null":
                    log_a = np.log(50) + market + np.cumsum(idio_vol * _innov(rng, n, dof))
                else:
                    bb = bounce_bp
                    if kind == "straddle_null":           # Study 3 (board r3): straddles the filter threshold
                        beta, sig = rng.uniform(0.97, 1.03), _logu(rng, 5e-4, 60e-4)
                        bb = float(rng.choice([0.0, 1.0, 2.0]))
                    elif kind in ("clone", "clone_null"):
                        beta, hl, sig = rng.uniform(0.97, 1.03), rng.uniform(1, 5), _logu(rng, 1e-4, 4e-4)
                    elif kind == "level_step":
                        beta = rng.uniform(0.6, 1.5)
                        hl = rng.uniform(3, 15) if rng.random() < 0.5 else rng.uniform(25, 45)
                        sig = _logu(rng, 25e-4, 200e-4)
                    else:                                         # coint, beta_drift
                        beta, hl, sig = rng.uniform(0.6, 1.5), rng.uniform(3, 30), _logu(rng, 25e-4, 200e-4)
                    phi = 1.0 if kind in ("clone_null", "straddle_null") else 0.5 ** (1.0 / hl)
                    e = sig * _innov(rng, n, dof)
                    s = np.zeros(n)
                    s[0] = 0.0 if phi >= 1 else sig / np.sqrt(1 - phi ** 2) * rng.standard_normal()
                    for t in range(1, n):
                        s[t] = phi * s[t - 1] + e[t]
                    if kind == "level_step":
                        step_day = int(rng.integers(formation, n))
                        s[step_day:] += 4.0 * sig / np.sqrt(1 - phi ** 2)
                    db = np.diff(log_b, prepend=log_b[0])
                    if kind == "beta_drift":
                        bt = beta + np.cumsum(0.10 / np.sqrt(252) * rng.standard_normal(n))
                        log_a = np.log(50) + np.cumsum(bt * db) + s
                    else:
                        log_a = np.log(50) + beta * (log_b - np.log(50)) + s
                na, nb = rng.standard_normal(n), rng.standard_normal(n)
                cols[a] = np.exp(log_a + bb / 1e4 * na)
                cols[b] = np.exp(log_b + bb / 1e4 * nb)
                g = f"SEC{sec:02d}"
                sectors[a] = sectors[b] = g
                truth.append({"a": a, "b": b, "kind": kind, "half_life": hl, "beta": beta, "sigma_e": sig,
                              "sector": g, "step_day": step_day, "bounce_bp": bb})
    return Panel(prices=pd.DataFrame(cols, index=dates), truth=pd.DataFrame(truth), sectors=sectors)


NULL_KINDS = {"null", "clone_null", "cross", "straddle_null"}
TRUE_KINDS = {"coint", "clone", "level_step"}      # cointegrated over the formation window
# beta_drift is neither: excluded from FDR, reported as a diagnostic (pre-registered)
