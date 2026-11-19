"""Daily paper-trading run. NO real orders are ever sent by this module.

Prices are split-adjusted closes (NOT dividend-adjusted, so history does not move); dividends come separately as
cash per share on ex-dates (board round 3, Asante D1). Each run on bar t (the latest bar in the file):
  0. data checks (abort with DataError): stale bar vs `expected_date`, NaN in the previous bar, revised history
     (a constant ratio is treated as a split and the book's shares are rescaled), one-day moves > 25% on
     traded tickers unless explicitly accepted;
  1. borrow and dividends accrue on the book held over (last run, t];
  2. last run's orders fill at today's close (one-bar delay, as in the backtest). Entry orders carry a dollar
     size and become shares at the fill price, so paper == backtest up to float error;
  3. the strategy is replayed causally over the current window (pair *identities* are cached per window; their
     parameters are refitted from this file every run, so a re-download cannot desynchronise the state);
  4. new orders with deterministic ids. A held ticker with no price today freezes that pair: the healthy leg is
     flattened and the pair is flagged for a human. The kill switch (file KILL, or PAIRS_KILL=1) only flattens.
State is written atomically before the order log is appended (crash-safe; a re-run of the same bar is a no-op).
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
from .strategy import Pair, make_pair, select_pairs, trade_window, whole_share_legs

LOG_COLUMNS = ["order_id", "date", "pair", "ticker", "kind", "side", "notional", "qty", "status", "fill_date",
               "fill_price", "cost"]


class DataError(RuntimeError):
    pass


def order_id(date, pair, leg, action, salt: str = "") -> str:
    """Deterministic, so a retry finds the same order. `salt` (Study 2: config hash + book) keeps two books on one
    broker account from colliding (Novak M5); empty salt reproduces the Study 1 ids."""
    key = f"{date}|{pair}|{leg}|{action}" + (f"|{salt}" if salt else "")
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def window_for(prices: pd.DataFrame, anchor: pd.Timestamp, cfg: StrategyConfig) -> tuple[int, int, int]:
    """Windows are counted in bars from a fixed calendar anchor (the first date of the book), not from row 0
    of whatever file was downloaded today (board round 2, Asante)."""
    off = int(prices.index.searchsorted(anchor))
    n = len(prices)
    if n - off <= cfg.formation_days:
        raise DataError(f"need more than {cfg.formation_days} bars after {anchor.date()}, have {n - off}")
    k = (n - 1 - off - cfg.formation_days) // cfg.trading_days
    start = off + cfg.formation_days + k * cfg.trading_days
    return off, start, start + cfg.trading_days


def check_data(prices, tickers, state, expected_date=None, accept_jumps=(), max_jump=0.25) -> dict:
    """Returns {ticker: split_ratio} for detected re-basings; raises DataError on anything suspicious."""
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise DataError("price index must be strictly increasing dates")
    today = prices.index[-1]
    if expected_date is not None and today.normalize() < pd.Timestamp(expected_date).normalize():
        raise DataError(f"stale data: last bar {today.date()}, expected {pd.Timestamp(expected_date).date()}")
    tickers = [t for t in tickers if t in prices.columns]
    if len(prices) >= 2:
        prev = prices[tickers].iloc[-2]
        if prev.isna().any():
            raise DataError(f"previous bar has missing prices for {list(prev[prev.isna()].index)}")
        r = (prices[tickers].iloc[-1] / prev - 1).abs()
        bad = {t: round(float(v), 3) for t, v in r.items() if np.isfinite(v) and v > max_jump
               and t not in accept_jumps}
        if bad:
            raise DataError(f"one-day move > {max_jump:.0%} on {bad}: check for an unadjusted split or bad tick; "
                            f"re-run with --accept-jump TICKER if it is real news")
    splits = {}
    for t, hist in state.get("snapshot", {}).items():
        if t not in prices.columns:
            continue
        old = pd.Series(hist)
        old.index = pd.to_datetime(old.index)
        new = prices[t].reindex(old.index)
        if new.isna().any():
            raise DataError(f"history for {t} lost dates that were present at the last run")
        ratio = (new / old).to_numpy()
        if np.allclose(ratio, 1.0, rtol=1e-6):
            continue
        if np.allclose(ratio, ratio[0], rtol=1e-6):
            splits[t] = float(ratio[0])                     # whole history rescaled by one factor = a split
        else:
            raise DataError(f"history for {t} was revised (not a constant split factor); refusing to trade")
    return splits


def _write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _flush_log(state, log_path: Path):
    rows = state.get("log_queue", [])
    if rows:
        pd.DataFrame(rows).reindex(columns=LOG_COLUMNS).to_csv(log_path, mode="a", header=not log_path.exists(),
                                                               index=False)
    state["log_queue"] = []


def run_paper(prices: pd.DataFrame, cfg: StrategyConfig, state_dir, sectors: dict | None = None,
              dividends: pd.DataFrame | None = None, expected_date=None, accept_jumps=()) -> dict:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path, log_path = state_dir / "state.json", state_dir / "orders.csv"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "book": {}, "open_orders": [], "cash": 0.0, "equity": [], "order_ids": [], "frozen": {}}
    if state.get("log_queue"):                                   # finish a log append interrupted by a crash
        _flush_log(state, log_path)
        _write_atomic(state_path, state)
    if state.get("config_hash") not in (None, cfg.hash()):
        raise RuntimeError("config changed since the paper book was opened: start a new state dir "
                           "(changing settings mid-run makes the paper record meaningless)")
    state["config_hash"] = cfg.hash()
    state.setdefault("anchor", str(prices.index[0].date()))
    killed = (state_dir / "KILL").exists() or os.environ.get("PAIRS_KILL") == "1"

    today = prices.index[-1]
    date = str(today.date())
    if state.get("last_run_date") == date:
        return {**state["last_summary"], "rerun": True}
    if state.get("last_run_date") and date < state["last_run_date"]:
        raise DataError(f"data ends {date}, before the last run {state['last_run_date']}")

    off, start, end = window_for(prices, pd.Timestamp(state["anchor"]), cfg)
    n = len(prices)
    formation = prices.iloc[start - cfg.formation_days:start]
    sel_key = str(prices.index[start].date())
    if state.get("selection", {}).get("window_start") != sel_key:
        chosen = select_pairs(formation, cfg, sectors)
        state["selection"] = {"window_start": sel_key,
                              "pairs": [{"a": p.a, "b": p.b, "p": p.p, "tstat": p.tstat, "group": p.group,
                                         "p_threshold": p.p_threshold} for p in chosen]}
        state["frozen"] = {}
    logf = np.log(formation)
    cache = state["selection"].setdefault("fit_cache", {})
    pairs = []
    for q in state["selection"]["pairs"]:
        la, lb = logf[q["a"]].to_numpy(), logf[q["b"]].to_numpy()
        key = hashlib.sha256(la.tobytes() + lb.tobytes()).hexdigest()[:16]   # refit only if the data changed
        name = f"{q['a']}/{q['b']}"
        if cache.get(name, {}).get("key") != key:
            fit = make_pair(la, lb, q["a"], q["b"], cfg, q["p_threshold"], q["p"], q["tstat"], q["group"])
            cache[name] = {"key": key, "pair": asdict(fit)}
        pairs.append(Pair(**cache[name]["pair"]))
    tickers = sorted({t for p in pairs for t in (p.a, p.b)} | {t for v in state["book"].values() for t in v})
    splits = check_data(prices, tickers, state, expected_date, accept_jumps)
    for pos in state["book"].values():                           # a split rescales prices: rescale shares too
        for t in pos:
            if t in splits:
                pos[t] /= splits[t]
    px_today = prices.iloc[-1]
    events = []

    # 1) borrow and dividends on the book held over (last run, today]
    last = pd.Timestamp(state["last_run_date"]) if state.get("last_run_date") else None
    bars = int(((prices.index > last) & (prices.index <= today)).sum()) if last is not None else 0
    for name, pos in state["book"].items():
        for t, q in pos.items():
            pt = float(px_today[t]) if np.isfinite(px_today[t]) else np.nan
            if q < 0 and np.isfinite(pt):
                state["cash"] -= cfg.borrow_bps_per_year / 1e4 / 252.0 * (-q) * pt * bars
            if dividends is not None and t in dividends.columns and last is not None:
                d = dividends[t][(dividends.index > last) & (dividends.index <= today)].fillna(0).sum()
                state["cash"] += q * float(d)

    # 2) fill last run's orders at today's close
    fills, still_open = [], []
    for o in state["open_orders"]:
        px = float(px_today[o["ticker"]])
        if cfg.study >= 2 and o["kind"] == "open":
            age = int(((prices.index > pd.Timestamp(o["date"])) & (prices.index <= today)).sum())
            if age > cfg.exec_delay_bars or not np.isfinite(px):   # entries expire at t+delay (Asante C2)
                fills.append({**o, "qty": 0.0, "fill_date": date, "fill_price": px, "cost": 0.0,
                              "status": "expired"})
                events.append(f"EXPIRED entry {o['pair']} {o['ticker']} (decided {o['date']}, age {age} bars)")
                continue
        if not np.isfinite(px):
            still_open.append(o)                                 # halted: try again next run
            continue
        pos = state["book"].setdefault(o["pair"], {})
        if o["kind"] == "open":
            qty = (o["qty_target"] if o.get("qty_target") is not None   # Study 2: whole shares, sized at decision
                   else o["notional"] / px)                      # Study 1: signed dollars -> shares at fill
        else:
            qty = -pos.get(o["ticker"], 0.0)
        cost = cfg.cost_bps / 1e4 * abs(qty) * px
        pos[o["ticker"]] = pos.get(o["ticker"], 0.0) + qty
        state["cash"] -= qty * px + cost
        fills.append({**o, "qty": qty, "fill_date": date, "fill_price": px, "cost": cost, "status": "filled"})
    state["open_orders"] = still_open
    state["book"] = {k: {t: q for t, q in v.items() if abs(q) > 1e-12} for k, v in state["book"].items()}
    state["book"] = {k: v for k, v in state["book"].items() if v}

    # 3) replay the strategy over the current window
    flatten_all = killed or (n - 1 >= end - 1 - cfg.exec_delay_bars)   # same bar as the backtest's window end
    open_state: dict = {}
    trade_window(prices, pairs, start, n, cfg, final=False, open_state=open_state, dividends=dividends)
    pending_pairs = {o["pair"] for o in state["open_orders"]}
    new_orders = []

    def emit(name, ticker, kind, side=0, notional=0.0, qty_target=None):
        oid = order_id(date, name, ticker, kind, salt=f"{cfg.hash()}|{Path(state_dir).name}" if cfg.study >= 2 else "")
        if oid in state["order_ids"]:
            return
        new_orders.append({"order_id": oid, "date": date, "pair": name, "ticker": ticker, "kind": kind,
                           "side": side, "notional": notional, "qty_target": qty_target, "status": "submitted"})

    names = {f"{p.a}/{p.b}": p for p in pairs}
    for name in sorted(set(names) | set(state["book"])):
        if name in pending_pairs:
            continue
        held = state["book"].get(name, {})
        missing = [t for t in name.split("/") if not np.isfinite(px_today.get(t, np.nan))]
        if missing:
            if held:
                state["frozen"][name] = f"no price for {missing} on {date}: flatten by hand"
                events.append(f"FROZEN {name}: {state['frozen'][name]}")
                for t in held:
                    if t not in missing:
                        emit(name, t, "close")
            continue
        p = names.get(name)
        st = open_state.get(name, {"side": 0, "next_action": None})
        want = st["side"]
        if st["next_action"]:
            want = st["next_action"]["side"] if st["next_action"]["action"] == "enter" else 0
        if p is None or flatten_all or name in state["frozen"]:
            want = 0
        if not held:
            have = 0
        elif (p is not None and set(held) == {p.a, p.b}
              and np.sign(held[p.a]) == -np.sign(held[p.b]) != 0):
            have = int(np.sign(held[p.a]))
        else:                     # one-legged or wrong-signed book (Novak M4): never "equal" to anything
            have = None
            events.append(f"BROKEN LEGS {name}: {held} -> closing")
        if want == have:
            continue
        for t in held:
            emit(name, t, "close")
        if want != 0:
            n_a = cfg.capital / cfg.max_pairs / (1.0 + p.beta)
            if cfg.integer_shares:
                ka, kb = whole_share_legs(cfg.capital / cfg.max_pairs, p.beta, float(px_today[p.a]),
                                          float(px_today[p.b]))
                emit(name, p.a, "open", want, want * ka * float(px_today[p.a]), qty_target=float(want * ka))
                emit(name, p.b, "open", want, -want * kb * float(px_today[p.b]), qty_target=float(-want * kb))
            else:
                emit(name, p.a, "open", want, want * n_a)
                emit(name, p.b, "open", want, -want * p.beta * n_a)
    state["open_orders"] += new_orders
    state["order_ids"] += [o["order_id"] for o in new_orders]

    mtm = sum(q * float(px_today[t]) for pos in state["book"].values() for t, q in pos.items()
              if np.isfinite(px_today[t]))
    equity = state["cash"] + mtm
    state["equity"] = [e for e in state["equity"] if e["date"] != date] + [{"date": date, "pnl": equity}]
    state["snapshot"] = {t: {str(d.date()): float(v) for d, v in prices[t].iloc[-60:].items()}
                         for t in tickers if t in prices.columns}
    summary = {"date": date, "killed": killed, "window": [sel_key, str(prices.index[min(end, n) - 1].date())],
               "pairs": [{**asdict(p), "z": open_state.get(f"{p.a}/{p.b}", {}).get("z")} for p in pairs],
               "fills": fills, "new_orders": new_orders, "paper_pnl": equity, "book": state["book"],
               "frozen": state["frozen"], "events": events + [f"split {t} x{r:.4f}" for t, r in splits.items()]}
    state["last_run_date"], state["last_summary"] = date, summary
    state["log_queue"] = fills + new_orders
    _write_atomic(state_path, state)                             # 1) state first (atomic)
    _flush_log(state, log_path)                                  # 2) then the human-readable log
    _write_atomic(state_path, state)
    return {**summary, "rerun": False}
