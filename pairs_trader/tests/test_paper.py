import json

import numpy as np
import pandas as pd
import pytest

from pairs_trader import cli
from pairs_trader.config import StrategyConfig
from pairs_trader.paper import DataError, run_paper
from pairs_trader.simulate import PanelConfig, simulate_panel
from pairs_trader.strategy import select_pairs, trade_window

CFG = StrategyConfig()
F, M = CFG.formation_days, CFG.trading_days


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(PanelConfig(n_days=F + M + 30, kinds=(("coint", 6), ("null", 4)),
                                      half_life_range=(3.0, 8.0), seed=11))


def _run_days(panel, state, first, last, **kw):
    out = None
    for end in range(first, last + 1):
        out = run_paper(panel.prices.iloc[:end], CFG, state, panel.sectors, **kw)
    return out


def test_paper_equals_backtest_over_a_full_window(panel, tmp_path):
    """Board round 2 (Novak, Asante): the paper code path must reproduce the backtest's P&L."""
    out = _run_days(panel, tmp_path, F + 1, F + M)          # bars F .. F+M-1 = one whole trading window
    pairs = select_pairs(panel.prices.iloc[:F], CFG, panel.sectors)
    pnl, trades = trade_window(panel.prices, pairs, F, F + M, CFG)
    assert len(trades) > 0
    assert out["book"] == {}                                   # flat at the window end, like the backtest
    assert out["paper_pnl"] == pytest.approx(pnl.sum(), rel=1e-6, abs=1e-6)


def test_rerun_same_day_is_idempotent(panel, tmp_path):
    px = panel.prices.iloc[:F + 40]
    _run_days(panel, tmp_path, F + 1, F + 39)
    r1 = run_paper(px, CFG, tmp_path, panel.sectors)
    r2 = run_paper(px, CFG, tmp_path, panel.sectors)
    assert r2["rerun"] and r2["new_orders"] == r1["new_orders"]
    log = pd.read_csv(tmp_path / "orders.csv")
    sub = log[log.status == "submitted"]
    assert not sub["order_id"].duplicated().any()


def test_crash_between_state_and_log_loses_nothing(panel, tmp_path, monkeypatch):
    """Board round 2 (Novak MAJOR 1): orders must survive a crash after the state write."""
    import pairs_trader.paper as P
    _run_days(panel, tmp_path, F + 1, F + 20)
    calls = {"n": 0}
    real = P._flush_log

    def boom(state, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt("simulated crash")
        return real(state, path)
    monkeypatch.setattr(P, "_flush_log", boom)
    with pytest.raises(KeyboardInterrupt):
        run_paper(panel.prices.iloc[:F + 21], CFG, tmp_path, panel.sectors)
    st = json.loads((tmp_path / "state.json").read_text())
    queued = len(st["log_queue"])
    monkeypatch.setattr(P, "_flush_log", real)
    run_paper(panel.prices.iloc[:F + 22], CFG, tmp_path, panel.sectors)
    log = pd.read_csv(tmp_path / "orders.csv")
    assert (log.date == str(panel.prices.index[F + 20].date())).sum() + \
        (log.fill_date == str(panel.prices.index[F + 20].date())).sum() >= queued


def test_missing_held_leg_flattens_the_other_leg_and_flags(panel, tmp_path):
    out = None
    for end in range(F + 1, F + M):
        out = run_paper(panel.prices.iloc[:end], CFG, tmp_path, panel.sectors)
        if out["book"]:
            break
    assert out["book"], "no position opened in this window"
    name = next(iter(out["book"]))
    dead = name.split("/")[0]
    px = panel.prices.iloc[:end + 1].copy()
    px.iloc[-1, px.columns.get_loc(dead)] = np.nan
    r = run_paper(px, CFG, tmp_path, panel.sectors)
    assert name in r["frozen"]
    assert any(o["pair"] == name and o["kind"] == "close" and o["ticker"] != dead for o in r["new_orders"])


def test_kill_switch_only_flattens(panel, tmp_path):
    _run_days(panel, tmp_path, F + 1, F + 60)
    (tmp_path / "KILL").write_text("stop")
    out = _run_days(panel, tmp_path, F + 61, F + 63)
    assert out["killed"] and out["book"] == {}
    assert all(o["kind"] == "close" for o in out["new_orders"])


def test_config_change_refused(panel, tmp_path):
    run_paper(panel.prices.iloc[:F + 10], CFG, tmp_path, panel.sectors)
    with pytest.raises(RuntimeError):
        run_paper(panel.prices.iloc[:F + 11], CFG.with_(entry_z=2.5), tmp_path, panel.sectors)


def test_jump_and_revision_and_stale_checks(panel, tmp_path):
    _run_days(panel, tmp_path, F + 1, F + 29)
    px = panel.prices.iloc[:F + 30].copy()
    t = json.loads((tmp_path / "state.json").read_text())["selection"]["pairs"][0]["a"]
    bad = px.copy()
    bad.iloc[-1, bad.columns.get_loc(t)] *= 1.5                  # 50% one-day move
    with pytest.raises(DataError):
        run_paper(bad, CFG, tmp_path, panel.sectors)
    rev = px.copy()
    rev.iloc[-5, rev.columns.get_loc(t)] *= 1.01                  # one past bar silently changed
    with pytest.raises(DataError):
        run_paper(rev, CFG, tmp_path, panel.sectors)
    with pytest.raises(DataError):
        run_paper(px, CFG, tmp_path, panel.sectors, expected_date=px.index[-1] + pd.Timedelta(days=3))
    assert not run_paper(px, CFG, tmp_path, panel.sectors)["rerun"]   # the clean file still runs


def test_split_rescales_book(panel, tmp_path):
    out = None
    for end in range(F + 1, F + M):
        out = run_paper(panel.prices.iloc[:end], CFG, tmp_path, panel.sectors)
        if out["book"]:
            break
    name = next(iter(out["book"]))
    t = name.split("/")[0]
    q_before = out["book"][name][t]
    px = panel.prices.iloc[:end + 1].copy()
    px[t] = px[t] / 2.0                                         # 2:1 split applied to the whole history
    r = run_paper(px, CFG, tmp_path, panel.sectors)
    assert any("split" in e for e in r["events"])
    if name in r["book"]:
        assert r["book"][name][t] == pytest.approx(q_before * 2.0)


def test_live_refuses():
    assert cli.main(["live"]) == 2
