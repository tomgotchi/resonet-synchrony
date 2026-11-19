"""Broker adapters (board 2026-11-19: Asante C1-C3, M1-M3; Novak broker interface and fault tests).

Both brokers take whole-share, market-on-close orders (time_in_force "cls") identified by a deterministic
client_order_id, so a retry after a crash finds the order instead of sending it twice.

- SimBroker: a local simulated broker. Orders fill at the next close fed in by `settle`. It can inject faults
  (rejections, partial fills) for tests.
- AlpacaPaperBroker: Alpaca's *paper* REST API only. The base URL is a constant, and no live URL exists anywhere
  in this code. Keys come from the environment (APCA_API_KEY_ID, APCA_API_SECRET_KEY) and are never logged or
  printed. Alpaca facts checked against docs.alpaca.markets on 2026-11-19:
  - no fractional short sales, so whole shares are used;
  - cls orders sent after 19:00 ET are queued for the next session's closing auction, and between 15:50 and
    19:00 ET they are rejected;
  - paper fills are simulated from live quotes, so they test plumbing, not costs.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

TERMINAL_UNFILLED = {"canceled", "expired", "rejected", "done_for_day"}


@dataclass
class BrokerOrder:
    client_order_id: str
    symbol: str
    qty: int                     # signed: + buy, - sell
    status: str = "new"          # new | accepted | filled | partially_filled | rejected | canceled | expired
    filled_qty: int = 0          # signed
    filled_avg_price: float | None = None
    submitted_for: str = ""      # the session date whose close should fill it
    reason: str = ""


class BrokerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- simulated
@dataclass
class SimBroker:
    path: Path | None = None
    orders: dict = field(default_factory=dict)
    pos: dict = field(default_factory=dict)
    reject: set = field(default_factory=set)            # symbols whose next order is rejected (tests)
    partial: dict = field(default_factory=dict)         # symbol -> fraction filled (tests)
    not_shortable: set = field(default_factory=set)

    @classmethod
    def load(cls, path) -> "SimBroker":
        p = Path(path)
        b = cls(path=p)
        if p.exists():
            d = json.loads(p.read_text())
            b.orders = {k: BrokerOrder(**v) for k, v in d["orders"].items()}
            b.pos = {k: int(v) for k, v in d["pos"].items()}
        return b

    def save(self):
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"orders": {k: asdict(v) for k, v in self.orders.items()}, "pos": self.pos}))
        os.replace(tmp, self.path)

    def asset(self, symbol) -> dict:
        return {"tradable": True, "shortable": symbol not in self.not_shortable, "borrow_status": "easy"}

    def submit(self, client_order_id, symbol, qty: int, session: str = "") -> BrokerOrder:
        if client_order_id in self.orders:                  # idempotent: a retry returns the original order
            return self.orders[client_order_id]
        if int(qty) != qty or qty == 0:
            raise BrokerError(f"whole, non-zero shares only: {qty}")
        o = BrokerOrder(client_order_id, symbol, int(qty), "accepted", submitted_for=session)
        if symbol in self.reject:
            self.reject.discard(symbol)
            o.status, o.reason = "rejected", "injected rejection"
        elif qty < 0 and self.pos.get(symbol, 0) + qty < 0 and symbol in self.not_shortable:
            o.status, o.reason = "rejected", "not shortable"
        self.orders[client_order_id] = o
        self.save()
        return o

    def get_order(self, client_order_id) -> BrokerOrder | None:
        return self.orders.get(client_order_id)

    def cancel(self, client_order_id) -> None:
        o = self.orders.get(client_order_id)
        if o and o.status in ("new", "accepted"):
            o.status = "canceled"
            self.save()

    def positions(self) -> dict:
        return {k: v for k, v in self.pos.items() if v != 0}

    def settle(self, session: str, closes: dict) -> None:
        """The closing auction of `session`: fill every accepted order due on or before it."""
        for o in self.orders.values():
            if o.status != "accepted" or (o.submitted_for and o.submitted_for > session):
                continue
            px = closes.get(o.symbol)
            if px is None or px != px:                      # no print (halt): MOC orders are cancelled
                o.status, o.reason = "canceled", "no closing print"
                continue
            frac = self.partial.pop(o.symbol, 1.0)
            q = int(o.qty * frac)
            o.filled_qty, o.filled_avg_price = q, float(px)
            o.status = "filled" if q == o.qty else "partially_filled"
            self.pos[o.symbol] = self.pos.get(o.symbol, 0) + q
        self.save()


# ---------------------------------------------------------------------------------------------- Alpaca paper
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"


class AlpacaPaperBroker:
    """Alpaca paper trading over REST. Needs APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment."""

    def __init__(self, timeout: float = 20.0):
        self._kid = os.environ.get("APCA_API_KEY_ID")
        self._sec = os.environ.get("APCA_API_SECRET_KEY")
        if not self._kid or not self._sec:
            raise BrokerError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys) in the environment")
        self.timeout = timeout

    def __repr__(self):                                     # never show keys
        return "AlpacaPaperBroker(paper-api, keys=<redacted>)"

    def _req(self, method, path, body=None, ok404=False):
        req = urllib.request.Request(ALPACA_PAPER_URL + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"APCA-API-KEY-ID": self._kid, "APCA-API-SECRET-KEY": self._sec,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if ok404 and e.code == 404:
                return None
            msg = e.read()[:300].decode(errors="replace")
            raise BrokerError(f"Alpaca {method} {path} -> HTTP {e.code}: {msg}") from None

    @staticmethod
    def _order(d) -> BrokerOrder:
        side = 1 if d["side"] == "buy" else -1
        fq = float(d.get("filled_qty") or 0)
        return BrokerOrder(d["client_order_id"], d["symbol"], side * int(float(d["qty"])), d["status"],
                           side * int(fq), float(d["filled_avg_price"]) if d.get("filled_avg_price") else None)

    def asset(self, symbol) -> dict:
        return self._req("GET", f"/v2/assets/{symbol}")

    def clock(self) -> dict:
        return self._req("GET", "/v2/clock")

    def calendar(self, start: str, end: str) -> list:
        return self._req("GET", f"/v2/calendar?start={start}&end={end}")

    def submit(self, client_order_id, symbol, qty: int, session: str = "") -> BrokerOrder:
        prior = self.get_order(client_order_id)             # retry after a crash: never send twice
        if prior is not None:
            return prior
        if int(qty) != qty or qty == 0:
            raise BrokerError(f"whole, non-zero shares only: {qty}")
        d = self._req("POST", "/v2/orders", {"symbol": symbol, "qty": str(abs(int(qty))),
                                             "side": "buy" if qty > 0 else "sell", "type": "market",
                                             "time_in_force": "cls", "client_order_id": client_order_id})
        return self._order(d)

    def get_order(self, client_order_id) -> BrokerOrder | None:
        d = self._req("GET", f"/v2/orders:by_client_order_id?client_order_id={client_order_id}", ok404=True)
        return self._order(d) if d else None

    def cancel(self, client_order_id) -> None:
        o = self._req("GET", f"/v2/orders:by_client_order_id?client_order_id={client_order_id}", ok404=True)
        if o and o["status"] in ("new", "accepted", "pending_new"):
            self._req("DELETE", f"/v2/orders/{o['id']}")

    def positions(self) -> dict:
        return {p["symbol"]: int(float(p["qty"])) for p in self._req("GET", "/v2/positions")}
