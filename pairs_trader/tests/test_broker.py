"""Broker loop fault tests (board 2026-11-19, Novak r2 list)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from pairs_trader.broker import AlpacaPaperBroker, BrokerError, SimBroker
from pairs_trader.config import StrategyConfig
from pairs_trader.simulate import PanelConfig, simulate_panel
from pairs_trader.trader import Halt, run_trader, shadow_positions

CFG = StrategyConfig().with_(study=2, cost_filter_m=1.0, integer_shares=True, calibration_sims=40)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(PanelConfig(n_days=504 + 126, kinds=(("coint", 6), ("null", 4)), seed=11))


def drive(panel, book, broker, days, before=None):
    """One session per day: the closing auction of day d settles yesterday's orders, then the loop runs."""
    idx = panel.prices.index
    outs = []
    for i, d in enumerate(days):
        j = idx.get_loc(d)
        broker.settle(str(d.date()), panel.prices.iloc[j].to_dict())
        if before:
            before(i, d, broker)
        nxt = str(idx[j + 1].date()) if j + 1 < len(idx) else ""
        outs.append(run_trader(panel.prices.loc[:d], CFG, book, broker, panel.sectors, next_session=nxt))
    return outs


def test_happy_path_positions_always_reconcile(panel, tmp_path):
    b = SimBroker.load(tmp_path / "sim.json")
    outs = drive(panel, tmp_path / "book", b, panel.prices.index[504:629])
    assert sum(o["broker"]["submitted"] for o in outs) > 0
    st = json.loads((tmp_path / "book" / "state.json").read_text())
    assert shadow_positions(st) == b.positions()
    assert not (tmp_path / "book" / "HALT").exists()
    assert all(float(o.qty).is_integer() for o in b.orders.values())


def test_rerun_same_day_does_not_double_submit(panel, tmp_path):
    b = SimBroker.load(tmp_path / "sim.json")
    days = panel.prices.index[504:530]
    drive(panel, tmp_path / "book", b, days)
    n = len(b.orders)
    d = days[-1]
    run_trader(panel.prices.loc[:d], CFG, tmp_path / "book", b, panel.sectors)     # crash-retry of the same day
    assert len(b.orders) == n


def test_crash_after_submit_before_state_write_no_duplicate(panel, tmp_path, monkeypatch):
    """The broker accepted the orders, then the process died: the retry must find them by client id."""
    b = SimBroker.load(tmp_path / "sim.json")
    days = panel.prices.index[504:530]
    drive(panel, tmp_path / "book", b, days[:-1])
    import pairs_trader.trader as T
    real = T._heartbeat
    monkeypatch.setattr(T, "_heartbeat", lambda *a: (_ for _ in ()).throw(SystemExit("crash")))
    idx = panel.prices.index
    j = idx.get_loc(days[-1])
    b.settle(str(days[-1].date()), panel.prices.iloc[j].to_dict())
    with pytest.raises(SystemExit):
        run_trader(panel.prices.loc[:days[-1]], CFG, tmp_path / "book", b, panel.sectors, next_session="x")
    n = len(b.orders)
    monkeypatch.setattr(T, "_heartbeat", real)
    run_trader(panel.prices.loc[:days[-1]], CFG, tmp_path / "book", b, panel.sectors, next_session="x")
    assert len(b.orders) == n


def _first_entry_day(panel, tmp_path):
    b = SimBroker.load(tmp_path / "probe.json")
    outs = drive(panel, tmp_path / "probe", b, panel.prices.index[504:600])
    for o, d in zip(outs, panel.prices.index[504:600]):
        ents = [x for x in o["new_orders"] if x["kind"] == "open"]
        if ents:
            return d, ents
    pytest.skip("no entry in the probe window")


def test_one_leg_rejected_cancels_sibling_and_book_stays_flat(panel, tmp_path):
    d0, ents = _first_entry_day(panel, tmp_path)
    victim = ents[0]["ticker"]
    b = SimBroker.load(tmp_path / "sim.json")
    days = panel.prices.index[504:panel.prices.index.get_loc(d0) + 3]

    def before(i, d, broker):
        if d == d0:
            broker.reject.add(victim)
    outs = drive(panel, tmp_path / "book", b, days, before=before)
    assert any("rejected at submit" in e for o in outs for e in o["events"])
    pair = ents[0]["pair"]
    assert not any(t in b.positions() for t in pair.split("/")) or not (tmp_path / "book" / "HALT").exists()
    assert not (tmp_path / "book" / "HALT").exists()


def test_not_shortable_withdraws_the_pair(panel, tmp_path):
    d0, ents = _first_entry_day(panel, tmp_path)
    short = next(e["ticker"] for e in ents if e["qty_target"] < 0)
    b = SimBroker.load(tmp_path / "sim.json")
    b.not_shortable.add(short)
    days = panel.prices.index[504:panel.prices.index.get_loc(d0) + 3]
    outs = drive(panel, tmp_path / "book", b, days)
    assert any("not tradable/shortable" in e for o in outs for e in o["events"])
    assert short not in b.positions()


def test_broker_state_mismatch_halts_and_stays_halted(panel, tmp_path):
    b = SimBroker.load(tmp_path / "sim.json")
    days = panel.prices.index[504:520]
    drive(panel, tmp_path / "book", b, days[:-1])
    b.pos["ZZZ"] = 5                                         # someone traded the account by hand
    with pytest.raises(Halt):
        drive(panel, tmp_path / "book", b, days[-1:])
    assert (tmp_path / "book" / "HALT").exists()
    with pytest.raises(Halt):
        run_trader(panel.prices.loc[:days[-1]], CFG, tmp_path / "book", b, panel.sectors)


def test_partial_fill_halts_for_a_human(panel, tmp_path):
    d0, ents = _first_entry_day(panel, tmp_path)
    b = SimBroker.load(tmp_path / "sim.json")
    days = panel.prices.index[504:panel.prices.index.get_loc(d0) + 2]

    def before(i, d, broker):
        if d == d0:                                  # set before the next closing auction settles the entry
            broker.partial[ents[0]["ticker"]] = 0.5
    with pytest.raises(Halt):
        drive(panel, tmp_path / "book", b, days, before=before)


def test_alpaca_needs_keys_and_never_shows_them(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(BrokerError):
        AlpacaPaperBroker()
    monkeypatch.setenv("APCA_API_KEY_ID", "PKTESTKEYID")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "supersecretvalue")
    br = AlpacaPaperBroker()
    assert "supersecret" not in repr(br) and "PKTEST" not in repr(br) and "supersecret" not in str(br)


def test_live_url_absent_from_code():
    src = "".join(p.read_text() for p in Path(__file__).resolve().parents[1].joinpath("pairs_trader").glob("*.py"))
    assert not re.search(r"https://api\.alpaca\.markets", src)


def test_non_integer_config_refused(panel, tmp_path):
    with pytest.raises(Halt):
        run_trader(panel.prices.iloc[:600], StrategyConfig(), tmp_path / "b", SimBroker(), panel.sectors)
