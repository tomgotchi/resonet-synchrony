"""Frozen strategy configuration. Changing any value is a new *trial* and must be logged in the ledger."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class StrategyConfig:
    # --- pair selection (formation window) ---
    formation_days: int = 504          # 2 years: 252 days cannot tell a 20-day half-life from noise (board 2026-11-18)
    trading_days: int = 126
    fdr_q: float = 0.05                # Benjamini-Hochberg over all candidate pairs, using max(p_AB, p_BA)
    adf_lags: int = 1
    half_life_min: float = 2.0         # bias-corrected (Kendall) half-life, trading days
    half_life_max: float = 30.0
    beta_min: float = 0.2              # hedge ratio bounds (log-price regression A on B)
    beta_max: float = 5.0
    max_pairs: int = 10
    one_pair_per_ticker: bool = True   # pairs never share a leg -> less hidden concentration
    # --- signal (z uses the formation mean/sd of the spread, frozen for the window) ---
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    time_stop_half_lives: float = 3.0
    cusum_k: float = 0.5               # break detector on standardised AR(1) innovations
    cusum_h: float = 8.0
    # --- execution & costs ---
    exec_delay_bars: int = 1           # decide at close t, fill at close t+delay (no opens in the data)
    cost_bps: float = 5.0              # per leg, per fill (so a round trip on a pair = 4 fills)
    borrow_bps_per_year: float = 100.0 # on the short leg's notional
    capital: float = 100_000.0
    # --- optional ML entry filter (OFF by default; board: no evidence it adds information) ---
    use_ml_filter: bool = False
    ml_min_train_trades: int = 200
    ml_threshold: float = 0.5

    def with_(self, **kw) -> "StrategyConfig":
        return replace(self, **kw)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]
