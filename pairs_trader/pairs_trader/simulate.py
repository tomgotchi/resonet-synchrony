"""Synthetic binary-event markets with a KNOWN true probability.

The true probability is an exact martingale: a latent Gaussian walk Z_t with Z_T > 0  <=>  YES, so
p_t = P(Z_T > 0 | Z_t) = Phi(Z_t / (sigma * sqrt(T - t))). The outcome is drawn consistently with that path.

The quoted market is a distorted copy of p_t:
    logit(q_t) = slope * logit(p_t) + noise          (slope < 1: favourite-longshot bias, longshots overpriced)
then bid/ask = q_t -/+ spread/2, rounded to whole cents and clipped to [0.01, 0.99].
slope = 1 and noise = 0 is an *efficient* market: no strategy should make money after fees and spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr


@dataclass(frozen=True)
class SimConfig:
    n_events: int = 2000
    markets_per_event: int = 1
    hours: int = 168               # market life
    sigma: float = 1.0             # latent volatility per sqrt(hour) (scale only matters relative to drift)
    drift_sd: float = 8.0          # spread of starting latent values -> spread of starting probabilities
    slope: float = 1.0             # market bias (1 = efficient)
    noise: float = 0.0             # extra logit noise on the quote
    spread: float = 0.02           # dollars
    seed: int = 0


def simulate(cfg: SimConfig) -> pd.DataFrame:
    """Return hourly snapshots: one row per (market, hour). Columns are what a real data feed would give us,
    plus `p_true` (never available in reality; kept only for tests) and `outcome`."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    T = cfg.hours
    for e in range(cfg.n_events):
        open_at = t0 + pd.Timedelta(hours=int(rng.integers(0, 24 * 365)))
        z0 = rng.normal(0.0, cfg.drift_sd)
        steps = rng.normal(0.0, cfg.sigma, size=T)
        z = z0 + np.concatenate([[0.0], np.cumsum(steps)])          # z[0..T]
        outcome = int(z[T] > 0)
        remaining = cfg.sigma * np.sqrt(np.arange(T, 0, -1))         # for hours 0..T-1
        p_true = ndtr(z[:T] / remaining)
        p_c = np.clip(p_true, 1e-6, 1 - 1e-6)
        for m in range(cfg.markets_per_event):
            q = expit(cfg.slope * logit(p_c) + rng.normal(0.0, cfg.noise, size=T) if cfg.noise else
                      cfg.slope * logit(p_c))
            bid = np.clip(np.round((q - cfg.spread / 2) * 100) / 100, 0.01, 0.98)
            ask = np.clip(np.round((q + cfg.spread / 2) * 100) / 100, 0.02, 0.99)
            ask = np.maximum(ask, bid + 0.01)
            vol = rng.poisson(50, size=T)
            ts = open_at + pd.to_timedelta(np.arange(T), unit="h")
            rows.append(pd.DataFrame({
                "ts": ts,
                "market_id": f"E{e:05d}-M{m}",
                "event_id": f"E{e:05d}",
                "open_ts": open_at,
                "close_ts": open_at + pd.Timedelta(hours=T),
                "yes_bid": bid,
                "yes_ask": ask,
                "volume": vol,
                "p_true": p_true,
                "outcome": outcome,
            }))
    return pd.concat(rows, ignore_index=True)
