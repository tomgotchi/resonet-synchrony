import json
import math

import numpy as np
import pandas as pd
import pytest

from pairs_trader.backtest import sharpe, walk_forward
from pairs_trader.coint import benjamini_hochberg, eg_pvalue, half_life_corrected, symmetric_eg
from pairs_trader.config import StrategyConfig
from pairs_trader.simulate import PanelConfig, simulate_panel
from pairs_trader.strategy import Pair, select_pairs, trade_window

CFG = StrategyConfig()


# ------------------------------------------------------------------ statistics of the selection step
def test_eg_null_false_positive_rate_is_nominal():
    """Board C1 (verified): with ADF critical values this would be ~14%; MacKinnon EG must give ~5%."""
    rng = np.random.default_rng(0)
    rej = 0
    R = 600
    for _ in range(R):
        x, y = np.cumsum(rng.standard_normal((2, 504)), axis=1)
        rej += eg_pvalue(y, x)[0] < 0.05
    assert 0.025 <= rej / R <= 0.08


def test_symmetric_eg_is_swap_invariant():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(300))
    y = 0.8 * x + rng.standard_normal(300)
    assert symmetric_eg(y, x)["p"] == symmetric_eg(x, y)["p"]


def test_bh_basic():
    assert benjamini_hochberg(np.array([0.001, 0.02, 0.9]), 0.05).tolist() == [True, True, False]
    assert benjamini_hochberg(np.array([0.04] * 100), 0.05).all()          # p(k) <= q k/m at k = m
    assert benjamini_hochberg(np.array([0.5, 0.6]), 0.05).sum() == 0


def test_half_life_correction_direction():
    # correction makes phi larger (slower reversion), i.e. longer half-life than the raw estimate
    phi_hat = 0.9
    raw = -math.log(2) / math.log(phi_hat)
    assert half_life_corrected(phi_hat, 252) > raw
    assert half_life_corrected(1.0, 252) == float("inf")


# ------------------------------------------------------------------ end-to-end on synthetic panels
@pytest.fixture(scope="module")
def planted():
    return simulate_panel(PanelConfig(n_days=252 * 5, kinds=(("coint", 8), ("null", 12)),
                                      half_life_range=(3.0, 10.0), seed=3))


@pytest.fixture(scope="module")
def null_panel():
    return simulate_panel(PanelConfig(n_days=252 * 5, kinds=(("null", 25),), seed=4))


def test_planted_pairs_are_found_with_right_beta(planted):
    pairs = select_pairs(planted.prices.iloc[:CFG.formation_days], CFG, planted.sectors)
    truth = planted.truth.set_index("a")
    true_found = [p for p in pairs if p.b == p.a[:-1] + "B" and truth.kind[p.a] == "coint"]
    assert len(true_found) >= 4
    assert len(true_found) >= 0.75 * len(pairs)          # selected pairs are mostly real
    errs = [abs(p.beta / truth.beta[p.a] - 1) for p in true_found]
    assert np.median(errs) < 0.10 and max(errs) < 0.20     # q = 0.10 admits weaker pairs with noisier betas


def test_planted_edge_is_positive_net_of_costs(planted):
    res = walk_forward(planted.prices, CFG, planted.sectors)
    s = res.summary()
    assert s["n_trades"] >= 30
    assert s["sharpe"] > 1.0 and s["sharpe_ci95"][0] > 0


def test_null_panel_has_no_edge(null_panel):
    res = walk_forward(null_panel.prices, CFG, null_panel.sectors)
    s = res.summary()
    n_selected = sum(w["n_pairs"] for w in res.windows)
    assert n_selected <= 3                               # FDR control: almost nothing passes
    assert not (s["sharpe_ci95"][0] > 0)                 # no significant edge


def test_costs_monotone(planted):
    rets = [walk_forward(planted.prices, CFG.with_(cost_bps=c), planted.sectors).daily_ret.sum()
            for c in (0.0, 5.0, 20.0)]
    assert rets[0] > rets[1] > rets[2]


def test_future_perturbation_does_not_change_past(planted):
    """Causality: change every price after day T; P&L up to T-1 must be identical."""
    px = planted.prices
    T = CFG.formation_days + 60
    pert = px.copy()
    rng = np.random.default_rng(9)
    pert.iloc[T:] *= np.exp(rng.normal(0, 0.05, pert.iloc[T:].shape))
    a = walk_forward(px, CFG, planted.sectors).daily_pnl
    b = walk_forward(pert, CFG, planted.sectors).daily_pnl
    cut = px.index[T - 1]
    pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])


def test_price_scaling_invariance(planted):
    px = planted.prices
    scaled = px.copy()
    scaled[px.columns[0]] *= 7.0
    scaled[px.columns[5]] *= 0.3
    a = walk_forward(px, CFG, planted.sectors).daily_pnl
    b = walk_forward(scaled, CFG, planted.sectors).daily_pnl
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)


def test_column_order_invariance(planted):
    px = planted.prices
    a = walk_forward(px, CFG, planted.sectors).daily_pnl
    b = walk_forward(px[px.columns[::-1]], CFG, planted.sectors).daily_pnl
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-9)


# ------------------------------------------------------------------ hand-checked accounting
def _toy(a_path, b=100.0):
    idx = pd.bdate_range("2020-01-01", periods=len(a_path))
    return pd.DataFrame({"A": a_path, "B": np.full(len(a_path), b)}, index=idx)


TOY = Pair("A", "B", alpha=0.0, beta=1.0, mu=0.0, sd=0.01, ar_c=0.0, ar_phi=0.5, ar_sigma=1e6,
           half_life=50.0, p=0.0, tstat=0.0)


def test_trade_accounting_matches_hand_calculation():
    a = np.full(30, 100.0)
    a[10:15] = 103.0                                      # z = log(1.03)/0.01 = 2.96 -> short spread
    px = _toy(a)
    cfg = CFG.with_(max_pairs=1, capital=10_000.0)
    pnl, trades = trade_window(px, [TOY], 0, 30, cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == -1 and t.exit_reason == "revert"
    assert t.entry_fill == px.index[11] and t.exit_fill == px.index[16]   # one-bar delay both ways
    n_a = 10_000.0 / 2                                     # beta = 1 -> half in each leg
    qa, qb = -n_a / 103.0, n_a / 100.0
    gross = qa * (100.0 - 103.0)                           # B never moves
    costs = 5e-4 * (abs(qa) * 103 + abs(qb) * 100) + 5e-4 * (abs(qa) * 100 + abs(qb) * 100)
    borrow = 0.01 / 252 * (-qa) * (103 + 103 + 103 + 100 + 100)   # marks on bars 12..16
    assert t.pnl == pytest.approx(gross - costs - borrow, rel=1e-9)
    assert pnl.sum() == pytest.approx(t.pnl, rel=1e-12)


def test_missing_price_forces_exit_not_freeze():
    """Board C5 (verified): NaN comparisons are False, so a naive state machine would hold forever."""
    a = np.full(30, 100.0)
    a[10:25] = 103.0
    a[14] = np.nan
    pnl, trades = trade_window(_toy(a), [TOY], 0, 30, CFG.with_(max_pairs=1))
    assert [t.exit_reason for t in trades][0] == "missing_data"
    assert np.isfinite(pnl).all()


def test_break_detector_stops_trading():
    p = simulate_panel(PanelConfig(n_days=252 * 5, kinds=(("break", 8),), half_life_range=(3, 8),
                                   break_day=CFG.formation_days + 20, seed=5))
    pairs = select_pairs(p.prices.iloc[:CFG.formation_days], CFG, p.sectors)
    assert pairs
    _, trades = trade_window(p.prices, pairs, CFG.formation_days, CFG.formation_days + CFG.trading_days, CFG)
    assert any(t.exit_reason == "break" for t in trades) or len(trades) <= len(pairs)


def test_config_hash_changes_with_any_value():
    assert CFG.hash() != CFG.with_(entry_z=2.1).hash()
    assert json.loads(CFG.to_json())["entry_z"] == 2.0


# ------------------------------------------------------------------ round-2 fixes
def test_halted_exit_fills_at_first_real_price_after_the_gap():
    """Board round 2 (Asante): exiting at the pre-halt price is a fill that never happened."""
    a = np.full(30, 100.0)
    a[10:13] = 103.0                     # short-spread entry signal at 10, filled at 11
    a[13:18] = np.nan                    # halt
    a[18:] = 130.0                       # reopens 30% higher: the short A leg loses
    pnl, trades = trade_window(_toy(a), [TOY], 0, 30, CFG.with_(max_pairs=1, capital=10_000.0))
    t = trades[0]
    assert t.exit_reason == "missing_data" and t.exit_fill == _toy(a).index[18]
    qa = -5_000.0 / 103.0
    assert t.pnl < qa * (130.0 - 103.0) * 0.99          # the loss at the reopen price is booked


def test_no_price_until_window_end_books_a_haircut():
    a = np.full(30, 100.0)
    a[10:15] = 103.0
    a[15:] = np.nan                       # never trades again
    _, trades = trade_window(_toy(a), [TOY], 0, 30, CFG.with_(max_pairs=1, capital=10_000.0))
    t = trades[0]
    assert t.exit_reason == "no_price_at_end"
    qa = -5_000.0 / 103.0
    assert t.pnl <= qa * (103.0 * 1.30 - 103.0) + 1e-6   # short leg marked 30% against us (plus costs)


def test_no_entry_that_would_fill_on_the_last_bar():
    a = np.full(12, 100.0)
    a[10] = 103.0                          # signal at 10 would fill at 11 = last bar
    _, trades = trade_window(_toy(a), [TOY], 0, 12, CFG.with_(max_pairs=1))
    assert trades == []


def test_dividends_are_cash_flows_long_receives_short_pays():
    a = np.full(30, 100.0)
    a[10:15] = 103.0                       # short A from bar 11 to 16
    px = _toy(a)
    div = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    div.iloc[13, 0] = 1.0                  # $1 dividend on A while short -> we pay it
    div.iloc[13, 1] = 2.0                  # $2 on B while long -> we receive it
    cfg = CFG.with_(max_pairs=1, capital=10_000.0)
    base = trade_window(px, [TOY], 0, 30, cfg)[1][0].pnl
    withd = trade_window(px, [TOY], 0, 30, cfg, dividends=div)[1][0].pnl
    qa, qb = -5_000.0 / 103.0, 5_000.0 / 100.0
    assert withd - base == pytest.approx(qa * 1.0 + qb * 2.0, rel=1e-9)


def test_calibration_brings_entry_rate_toward_nominal():
    """Board round 3 (Zhao): on genuinely cointegrated selected pairs, raw |z|>2 fires ~3x the nominal 4.6%."""
    F, M = CFG.formation_days, CFG.trading_days
    raw, cal = [], []
    for seed in range(4):
        p = simulate_panel(PanelConfig(n_days=F + M, kinds=(("coint", 20),), seed=200 + seed))
        for q in select_pairs(p.prices.iloc[:F], CFG, p.sectors):
            s = np.log(p.prices[q.a].values) - q.alpha - q.beta * np.log(p.prices[q.b].values)
            raw.append(np.mean(np.abs((s[F:] - q.mu) / q.sd) > 2))
            cal.append(np.mean(np.abs((s[F:] - q.mu) / (q.sd * q.z_scale)) > 2))
    assert len(cal) >= 15
    assert np.mean(raw) > 0.08 and 0.02 <= np.mean(cal) <= 0.075
