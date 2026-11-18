"""Exchange fees. All prices are dollars in [0, 1]; counts are whole contracts.

Default formula is the published Kalshi-style taker fee: ceil_to_cent(rate * C * P * (1 - P)).
Check the current fee schedule of your venue before trusting this; it is a config value, not a constant.
"""
import math


def taker_fee(price: float, contracts: int, rate: float = 0.07) -> float:
    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be in (0, 1) dollars, got {price}")
    if contracts < 0:
        raise ValueError("contracts must be >= 0")
    raw = rate * contracts * price * (1.0 - price)
    return math.ceil(round(raw * 100, 9)) / 100.0


def fee_per_contract(price: float, rate: float = 0.07) -> float:
    """Marginal fee per contract, ignoring cent rounding (used for edge thresholds)."""
    return rate * price * (1.0 - price)
