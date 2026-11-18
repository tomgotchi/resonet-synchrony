import numpy as np
import pandas as pd
import pytest

from pairs_trader import cli
from pairs_trader.config import StrategyConfig
from pairs_trader.paper import DataError, run_paper
from pairs_trader.simulate import PanelConfig, simulate_panel

CFG = StrategyConfig()


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(PanelConfig(n_days=CFG.formation_days + 100, kinds=(("coint", 6), ("null", 4)),
                                      half_life_range=(3.0, 8.0), seed=11))


def test_rerun_same_day_is_idempotent(panel, tmp_path):
    px = panel.prices.iloc[:CFG.formation_days + 40]
    r1 = run_paper(px, CFG, tmp_path, panel.sectors)
    r2 = run_paper(px, CFG, tmp_path, panel.sectors)
    assert r2["rerun"] and r2["new_orders"] == r1["new_orders"]
    log = pd.read_csv(tmp_path / "orders.csv") if (tmp_path / "orders.csv").exists() else pd.DataFrame()
    assert log.empty or not log["order_id"].duplicated().any()


def test_daily_sequence_fills_orders_and_tracks_pnl(panel, tmp_path):
    n_orders = 0
    for end in range(CFG.formation_days + 2, CFG.formation_days + 90):
        out = run_paper(panel.prices.iloc[:end], CFG, tmp_path, panel.sectors)
        n_orders += len(out["new_orders"])
    assert n_orders > 0
    log = pd.read_csv(tmp_path / "orders.csv")
    assert (log.status == "filled").sum() > 0
    assert not log[log.status == "submitted"]["order_id"].duplicated().any()


def test_kill_switch_only_flattens(panel, tmp_path):
    for end in range(CFG.formation_days + 2, CFG.formation_days + 60):
        run_paper(panel.prices.iloc[:end], CFG, tmp_path, panel.sectors)
    (tmp_path / "KILL").write_text("stop")
    out = run_paper(panel.prices.iloc[:CFG.formation_days + 60], CFG, tmp_path, panel.sectors)
    out2 = run_paper(panel.prices.iloc[:CFG.formation_days + 61], CFG, tmp_path, panel.sectors)
    assert out2["book"] == {}
    assert out["killed"] and out2["killed"]


def test_config_change_refused(panel, tmp_path):
    run_paper(panel.prices.iloc[:CFG.formation_days + 10], CFG, tmp_path, panel.sectors)
    with pytest.raises(RuntimeError):
        run_paper(panel.prices.iloc[:CFG.formation_days + 11], CFG.with_(entry_z=2.5), tmp_path, panel.sectors)


def test_bad_data_refused(panel, tmp_path):
    px = panel.prices.iloc[:CFG.formation_days + 30].copy()
    out = run_paper(px.iloc[:-1], CFG, tmp_path, panel.sectors)
    held = [t for p in out["pairs"] for t in p.split("/")]
    px.iloc[-1, px.columns.get_loc(held[0])] *= 3.0          # 200% one-day jump
    with pytest.raises(DataError):
        run_paper(px, CFG, tmp_path, panel.sectors)


def test_live_refuses():
    assert cli.main(["live"]) == 2
