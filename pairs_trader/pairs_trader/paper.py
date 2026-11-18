"""Daily paper-trading run. NO real orders are ever sent by this module.

Each run:
 1. loads the full price history (CSV, one column per ticker, date index, adjusted closes);
 2. checks the data (latest bar present, no NaN in held tickers, no unexplained >40% one-day jumps);
 3. replays the strategy causally over the current trading window (same code as the backtest), which gives the
    position the strategy holds now and the decision taken at the latest close;
 4. fills last run's paper orders at today's close (the backtest's one-bar delay), then writes new paper
    orders with deterministic ids (re-running the same day never duplicates an order);
 5. honours a kill switch (file KILL in the state dir, or env PAIRS_KILL=1): only flattening orders are written.

State lives in a JSON file; the order log is append-only CSV.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .strategy import Pair, select_pairs, trade_window


LOG_COLUMNS = ["order_id", "date", "pair", "ticker", "qty", "action", "status", "fill_date", "fill_price",
               "cost"]


class DataError(RuntimeError):
    pass


def check_data(prices: pd.DataFrame, tickers: list[str], max_jump: float = 0.40) -> None:
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise DataError("price index must be strictly increasing dates")
    last = prices.iloc[-1]
    missing = [t for t in tickers if t in prices and not np.isfinite(last[t])]
    if missing:
        raise DataError(f"latest bar {prices.index[-1].date()} missing prices for {missing}")
    if len(prices) >= 2 and tickers:
        r = (prices[tickers].iloc[-1] / prices[tickers].iloc[-2] - 1).abs()
        bad = r[r > max_jump]
        if len(bad):
            raise DataError(f"one-day move > {max_jump:.0%} (unadjusted split? bad tick?): {bad.to_dict()}")


def order_id(date, pair, leg, action) -> str:
    return hashlib.sha256(f"{date}|{pair}|{leg}|{action}".encode()).hexdigest()[:16]


def current_window(n_bars: int, cfg: StrategyConfig) -> tuple[int, int]:
    if n_bars <= cfg.formation_days + 1:
        raise DataError(f"need more than {cfg.formation_days + 1} bars of history, have {n_bars}")
    k = (n_bars - 1 - cfg.formation_days) // cfg.trading_days
    start = cfg.formation_days + k * cfg.trading_days
    return start, start + cfg.trading_days


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"book": {}, "open_orders": [], "realised": 0.0,
                                                               "equity": []}


def run_paper(prices: pd.DataFrame, cfg: StrategyConfig, state_dir, sectors: dict | None = None) -> dict:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path, log_path = state_dir / "state.json", state_dir / "orders.csv"
    state = _load_state(state_path)
    if state.get("config_hash") not in (None, cfg.hash()):
        raise RuntimeError("config changed since the paper book was opened: start a new state dir "
                           "(changing settings mid-run makes the paper record meaningless)")
    state["config_hash"] = cfg.hash()
    killed = (state_dir / "KILL").exists() or os.environ.get("PAIRS_KILL") == "1"

    n = len(prices)
    today = prices.index[-1]
    if state.get("last_run_date") == str(today.date()):
        # Re-run on the same bar: do nothing (no duplicate fills, no duplicate orders).
        return {**state["last_summary"], "rerun": True}
    if state.get("last_run_date") and str(today.date()) < state["last_run_date"]:
        raise DataError(f"data ends {today.date()}, before the last run {state['last_run_date']}")
    start, end = current_window(n, cfg)
    sel_key = str(prices.index[start].date())
    if state.get("selection", {}).get("window_start") != sel_key:
        pairs = select_pairs(prices.iloc[start - cfg.formation_days:start], cfg, sectors)
        state["selection"] = {"window_start": sel_key, "pairs": [asdict(p) for p in pairs]}
    pairs = [Pair(**p) for p in state["selection"]["pairs"]]
    tickers = sorted({t for p in pairs for t in (p.a, p.b)} | {t for k in state["book"] for t in k.split("/")})
    check_data(prices, tickers)

    # 1) fill yesterday's paper orders at today's close
    fills = []
    for o in state["open_orders"]:
        px = float(prices[o["ticker"]].iloc[-1])
        cost = cfg.cost_bps / 1e4 * abs(o["qty"]) * px
        pos = state["book"].setdefault(o["pair"], {})
        pos[o["ticker"]] = pos.get(o["ticker"], 0.0) + o["qty"]
        state["realised"] -= o["qty"] * px + cost        # cash flow
        fills.append({**o, "fill_date": str(today.date()), "fill_price": px, "cost": cost, "status": "filled"})
    state["open_orders"] = []
    state["book"] = {k: v for k, v in state["book"].items() if any(abs(q) > 1e-9 for q in v.values())}

    # 2) replay the strategy to find what it wants to hold after the next fill
    is_last_bar_of_window = n == end
    open_state: dict = {}
    trade_window(prices, pairs, start, n, cfg, final=False, open_state=open_state)
    targets = {}
    for p in pairs:
        name = f"{p.a}/{p.b}"
        st = open_state.get(name, {"side": 0, "next_action": None})
        side = st["side"]
        if st["next_action"]:
            side = st["next_action"]["side"] if st["next_action"]["action"] == "enter" else 0
        if killed or is_last_bar_of_window:
            side = 0
        if side == 0:
            targets[name] = {p.a: 0.0, p.b: 0.0}
        else:
            held = state["book"].get(name)
            if held and np.sign(held.get(p.a, 0.0)) == side:
                targets[name] = held                      # keep existing shares; no daily re-hedging
            else:
                n_a = cfg.capital / cfg.max_pairs / (1.0 + p.beta)
                pa, pb = float(prices[p.a].iloc[-1]), float(prices[p.b].iloc[-1])
                targets[name] = {p.a: float(np.round(side * n_a / pa)),
                                 p.b: float(np.round(-side * p.beta * n_a / pb))}
    for name in state["book"]:                           # anything held that is no longer a selected pair
        targets.setdefault(name, {t: 0.0 for t in state["book"][name]})

    # 3) new paper orders (idempotent ids)
    new_orders = []
    date = str(today.date())
    for name, tgt in targets.items():
        held = state["book"].get(name, {})
        for ticker, q in tgt.items():
            delta = q - held.get(ticker, 0.0)
            if abs(delta) < 1e-9:
                continue
            action = "buy" if delta > 0 else "sell"
            new_orders.append({"order_id": order_id(date, name, ticker, action), "date": date, "pair": name,
                               "ticker": ticker, "qty": delta, "action": action, "status": "submitted"})
    seen = set()
    if log_path.exists():
        seen = set(pd.read_csv(log_path, usecols=["order_id"])["order_id"].astype(str))
    new_orders = [o for o in new_orders if o["order_id"] not in seen]
    state["open_orders"] = new_orders

    mtm = sum(q * float(prices[t].iloc[-1]) for pos in state["book"].values() for t, q in pos.items())
    equity = state["realised"] + mtm
    state["equity"] = [e for e in state["equity"] if e["date"] != date] + [{"date": date, "pnl": equity}]
    rows = fills + new_orders
    if rows:
        pd.DataFrame(rows).reindex(columns=LOG_COLUMNS).to_csv(log_path, mode="a", header=not log_path.exists(),
                                                               index=False)
    summary = {"date": date, "killed": killed, "pairs": [f"{p.a}/{p.b}" for p in pairs], "fills": len(fills),
               "new_orders": new_orders, "paper_pnl": equity, "book": state["book"]}
    state["last_run_date"], state["last_summary"] = date, summary
    state_path.write_text(json.dumps(state, indent=1, default=str))
    return {**summary, "rerun": False}
