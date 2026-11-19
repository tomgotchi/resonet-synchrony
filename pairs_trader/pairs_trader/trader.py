"""Daily broker loop (board 2026-11-19, Asante r2 state machine; Novak broker fault tests).

Run once per session after 19:00 ET on day t. `paper.run_paper` stays the single source of strategy decisions
(the "shadow book"). The broker only executes them:

  HALT?  -> refuse while runs/<book>/HALT exists (a human removes it)
  SYNC   -> statuses of yesterday's orders. An entry the broker did not fill (rejected or cancelled) is withdrawn
            from the shadow book before it "fills" there.
  DECIDE -> run_paper: fills the shadow book at today's close and emits today's orders (whole shares)
  RECONCILE -> shadow positions must equal broker positions, share for share; otherwise HALT and alert
  PRECHECK  -> an entry whose short leg is not shortable is withdrawn as a pair (both legs)
  SUBMIT    -> market-on-close orders with deterministic client ids; a retry finds the order instead of resending
  CONFIRM   -> one leg rejected: cancel its sibling and withdraw both
  HEARTBEAT -> runs/<book>/heartbeat.json, plus a GET to $HEARTBEAT_URL if set (a dead-man's switch)

Orders never carry over silently: entries expire at t+1 (paper.py), and a missed session sends nothing.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import TERMINAL_UNFILLED, BrokerError
from .config import StrategyConfig
from .paper import _write_atomic, run_paper


class Halt(RuntimeError):
    pass


def _state(state_dir):
    p = Path(state_dir) / "state.json"
    return json.loads(p.read_text()) if p.exists() else None


def _withdraw(state_dir, order_ids: set, reason: str, events: list):
    """Remove not-yet-filled orders from the shadow book (atomic), so the shadow never fills what the broker
    will not."""
    if not order_ids:
        return
    p = Path(state_dir) / "state.json"
    st = json.loads(p.read_text())
    kept = [o for o in st["open_orders"] if o["order_id"] not in order_ids]
    gone = [o for o in st["open_orders"] if o["order_id"] in order_ids]
    st["open_orders"] = kept
    _write_atomic(p, st)
    for o in gone:
        events.append(f"WITHDRAWN {o['pair']} {o['ticker']} {o['kind']}: {reason}")


def shadow_positions(st) -> dict:
    out: dict = {}
    for pos in st.get("book", {}).values():
        for t, q in pos.items():
            out[t] = out.get(t, 0) + q
    return {t: int(round(q)) for t, q in out.items() if abs(q) > 1e-9}


def _alert(state_dir, msg):
    Path(state_dir, "HALT").write_text(msg + "\n")
    with open(Path(state_dir) / "alerts.log", "a") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}\n")


def _heartbeat(state_dir, info):
    Path(state_dir, "heartbeat.json").write_text(json.dumps(info, default=str))
    url = os.environ.get("HEARTBEAT_URL")
    if url:
        try:
            urllib.request.urlopen(url, timeout=10).read()
        except Exception:
            pass


def run_trader(prices: pd.DataFrame, cfg: StrategyConfig, state_dir, broker, sectors=None, dividends=None,
               next_session: str = "", **paper_kw) -> dict:
    if not cfg.integer_shares:
        raise Halt("broker mode needs integer_shares=True (whole-share orders)")
    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)
    if (sd / "HALT").exists():
        raise Halt(f"HALT is set: {(sd / 'HALT').read_text().strip()}")
    events: list = []
    today = str(prices.index[-1].date())

    # SYNC: yesterday's orders as the broker saw them
    st = _state(sd)
    if st is not None:
        bad_entries, by_pair = set(), {}
        for o in st["open_orders"]:
            by_pair.setdefault(o["pair"], []).append(o)
        for o in st["open_orders"]:
            bo = broker.get_order(o["order_id"])
            if o["kind"] == "open" and (bo is None or bo.status in TERMINAL_UNFILLED):
                bad_entries.add(o["order_id"])
        _withdraw(sd, bad_entries, "broker did not fill the entry", events)

    # DECIDE (shadow book fills at today's close, then emits today's orders)
    out = run_paper(prices, cfg, sd, sectors, dividends=dividends, **paper_kw)
    st = _state(sd)

    # slippage: broker fill vs the close the shadow used (plumbing only on a paper account)
    with open(sd / "slippage.csv", "a", newline="") as f:
        w = csv.writer(f)
        for fl in out["fills"]:
            bo = broker.get_order(fl["order_id"])
            if bo is not None and bo.filled_avg_price and fl.get("fill_price"):
                w.writerow([today, fl["ticker"], fl["order_id"], bo.filled_avg_price, fl["fill_price"],
                            round(1e4 * (bo.filled_avg_price / fl["fill_price"] - 1), 2)])

    # RECONCILE
    shadow, real = shadow_positions(st), {k: int(v) for k, v in broker.positions().items() if v}
    if shadow != real:
        diff = {t: (shadow.get(t, 0), real.get(t, 0)) for t in set(shadow) | set(real)
                if shadow.get(t, 0) != real.get(t, 0)}
        _alert(sd, f"{today} RECONCILE MISMATCH shadow vs broker {diff}; nothing submitted")
        raise Halt(f"positions differ (shadow, broker): {diff}")

    # PRECHECK entries: short leg shortable & tradable, else withdraw the pair
    new = [o for o in st["open_orders"] if o["date"] == today]
    pairs_new: dict = {}
    for o in new:
        pairs_new.setdefault((o["pair"], o["kind"]), []).append(o)
    drop = set()
    for (pair, kind), legs in pairs_new.items():
        if kind != "open":
            continue
        for o in legs:
            a = broker.asset(o["ticker"])
            if not a.get("tradable", True) or (o["qty_target"] < 0 and not a.get("shortable", False)):
                drop |= {x["order_id"] for x in legs}
    _withdraw(sd, drop, "precheck: not tradable/shortable", events)
    st = _state(sd)
    new = [o for o in st["open_orders"] if o["date"] == today]

    # SUBMIT
    held = shadow_positions(st)
    submitted = []
    for o in new:
        qty = int(o["qty_target"]) if o["kind"] == "open" else -held.get(o["ticker"], 0)
        if qty == 0:
            continue
        try:
            bo = broker.submit(o["order_id"], o["ticker"], qty, session=next_session)
        except BrokerError as e:
            bo = None
            events.append(f"SUBMIT ERROR {o['ticker']}: {e}")
        submitted.append((o, bo))

    # CONFIRM: a rejected leg withdraws its pair's entry (cancel the sibling at the broker)
    rejected = [(o, bo) for o, bo in submitted if bo is None or bo.status in ("rejected",)]
    withdraw = set()
    for o, _ in rejected:
        if o["kind"] == "open":
            for x, xb in submitted:
                if x["pair"] == o["pair"] and x["kind"] == "open":
                    broker.cancel(x["order_id"])
                    withdraw.add(x["order_id"])
        else:
            _alert(sd, f"{today} CLOSE ORDER REJECTED {o['ticker']} for {o['pair']}: flatten by hand")
    _withdraw(sd, withdraw, "a leg was rejected at submit", events)

    info = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "session": today,
            "next_session": next_session, "submitted": len(submitted), "rejected": len(rejected),
            "positions": real, "events": events, "paper_pnl": out["paper_pnl"]}
    _heartbeat(sd, info)
    return {**out, "broker": info, "events": out["events"] + events}
