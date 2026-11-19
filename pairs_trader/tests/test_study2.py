"""Study 2 (board 2026-11-19): pre-BH cost filter, whole-share sizing, entry expiry, Study 1 left untouched."""
import hashlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pairs_trader.backtest import random_pair_picker, walk_forward
from pairs_trader.coint import cost_filter_ok, cost_filter_stat, symmetric_eg
from pairs_trader.config import StrategyConfig
from pairs_trader.paper import order_id, run_paper
from pairs_trader.simulate import PanelConfig, simulate_panel
from pairs_trader.strategy import select_pairs, whole_share_legs

S1 = StrategyConfig()
S2 = StrategyConfig().with_(study=2, cost_filter_m=1.0, integer_shares=True)


# ----------------------------------------------------------------- Study 1 must stay bit-for-bit reproducible
def test_study1_hash_pinned():
    assert S1.hash() == "f201d54b4912"


def test_study1_golden_daily_pnl():
    """Golden numbers computed with the pre-Study-2 commit (c7a7c19) on the same synthetic panel."""
    p = simulate_panel(PanelConfig(n_days=252 * 4, kinds=(("coint", 6), ("null", 10)), seed=3))
    r = walk_forward(p.prices, S1.with_(calibration_sims=40), p.sectors)
    h = hashlib.sha256(r.daily_pnl.round(6).to_numpy().tobytes()).hexdigest()[:16]
    assert (len(r.trades), round(float(r.daily_pnl.sum()), 6), h) == (20, 4500.703327, "3baa40153939f122")


def test_study2_fields_change_hash():
    assert S2.hash() != S1.hash()
    assert S1.with_(cost_filter_m=1.0).hash() != S1.hash()


# ----------------------------------------------------------------- the filter
def test_filter_arithmetic():
    """A spread whose daily move per gross dollar is 15 bp passes m=1 (2 x 5 bp = 10 bp) and fails m=2."""
    rng = np.random.default_rng(0)
    lb = np.cumsum(0.01 * rng.standard_normal(505))
    la = lb + 0.0030 * rng.standard_normal(505)          # dla - dlb has sd 0.003*sqrt(2) ~ 42 bp; /(1+1) ~ 21 bp
    st = cost_filter_stat(la, lb)
    assert 0.0018 < st < 0.0024
    assert cost_filter_ok(la, lb, S2) and not cost_filter_ok(la, lb, S2.with_(cost_filter_m=2.5))
    assert cost_filter_ok(la, lb, S1)                     # m = 0: always passes


def test_filter_removes_clones_keeps_wide_pairs():
    rng = np.random.default_rng(1)
    n = 504
    common = np.cumsum(0.01 * rng.standard_normal(n))
    clone = common + 0.0005 * rng.standard_normal(n)      # ~5 bp tracking error: untradable after costs
    wide = common + np.cumsum(0.01 * rng.standard_normal(n)) * 0 + 0.02 * rng.standard_normal(n)
    assert not cost_filter_ok(clone, common, S2)
    assert cost_filter_ok(wide, common, S2)


def test_filter_statistic_independent_of_eg_pvalue_under_null():
    """Board round 2 (Novak, Zhao): the levels-space beta is not ancillary; the returns-space statistic must be.
    Null pairs with heterogeneous vols: Spearman(p, stat) ~ 0 and equal rejection on both halves."""
    rng = np.random.default_rng(7)
    p, st = [], []
    for _ in range(600):
        va, vb = np.exp(rng.uniform(np.log(0.005), np.log(0.03), 2))
        la = np.cumsum(va * rng.standard_normal(504))
        lb = np.cumsum(vb * rng.standard_normal(504))
        p.append(symmetric_eg(la, lb, 1)["p"])
        st.append(cost_filter_stat(la, lb))
    p, st = np.array(p), np.array(st)
    assert abs(spearmanr(p, st)[0]) < 0.1
    k = st >= np.median(st)
    assert abs(np.mean(p[k] < 0.1) - np.mean(p[~k] < 0.1)) < 0.05


def test_filter_runs_before_bh_and_placebo_respects_it():
    p = simulate_panel(PanelConfig(n_days=600, kinds=(("coint", 6), ("null", 10)), seed=4))
    f = p.prices.iloc[:504]
    cfg = S2.with_(cost_filter_m=1e6)                     # nothing can pass
    pairs, table = select_pairs(f, cfg, p.sectors, return_table=True)
    assert pairs == [] and not table.fdr_pass.any() and not table.cost_ok.any()
    assert random_pair_picker(0)(f, cfg, p.sectors, 0) == []
    pairs1, table1 = select_pairs(f, S2, p.sectors, return_table=True)   # 300 bp spreads: all pass m=1
    assert table1.cost_ok.all() and pairs1


# ----------------------------------------------------------------- sizing and execution
@pytest.mark.parametrize("pa,pb,beta", [(50, 60, 1.0), (3000, 40, 0.8), (412.5, 97.3, 1.3), (12, 900, 1.1)])
def test_whole_share_legs_near_target(pa, pb, beta):
    qa, qb = whole_share_legs(10_000, beta, pa, pb)
    assert qa >= 1 and qb >= 1 and float(qa).is_integer() and float(qb).is_integer()
    naive_qa = max(1, round(10_000 / (1 + beta) / pa))
    naive_err = abs(max(1, round(beta * naive_qa * pa / pb)) * pb / (naive_qa * pa) - beta) / beta
    assert abs(qb * pb / (qa * pa) - beta) / beta <= naive_err + 1e-12


def test_integer_backtest_holds_whole_shares_and_paper_matches(tmp_path):
    p = simulate_panel(PanelConfig(n_days=504 + 126, kinds=(("coint", 6), ("null", 4)), seed=11))
    cfg = S2.with_(calibration_sims=40)
    res = walk_forward(p.prices, cfg, p.sectors)
    assert res.trades
    last = None
    for d in p.prices.index[504:]:
        last = run_paper(p.prices.loc[:d], cfg, tmp_path / "book", p.sectors)
    log = pd.read_csv(tmp_path / "book" / "orders.csv")
    filled = log[(log.status == "filled") & (log.kind == "open")]
    assert len(filled) and np.allclose(filled.qty, filled.qty.round())
    assert last["paper_pnl"] == pytest.approx(res.daily_pnl.sum(), abs=0.05)


def test_entry_expires_after_a_missed_day(tmp_path):
    p = simulate_panel(PanelConfig(n_days=504 + 126, kinds=(("coint", 6), ("null", 4)), seed=11))
    cfg = S2.with_(calibration_sims=40)
    days = p.prices.index[504:]
    skipped = expired = False
    for i, d in enumerate(days):
        if skipped is False and i > 0:
            import json
            st = json.loads((tmp_path / "b" / "state.json").read_text())
            if any(o["kind"] == "open" for o in st["open_orders"]):
                skipped = True                              # skip the next session entirely
                continue
        out = run_paper(p.prices.loc[:d], cfg, tmp_path / "b", p.sectors)
        if any(e.startswith("EXPIRED") for e in out["events"]):
            expired = True
            break
    assert skipped and expired


def test_order_ids_salted_for_study2_only():
    assert order_id("2026-01-02", "A/B", "A", "open") == order_id("2026-01-02", "A/B", "A", "open", salt="")
    assert order_id("2026-01-02", "A/B", "A", "open", salt="x|book1") != order_id("2026-01-02", "A/B", "A", "open",
                                                                                  salt="x|book2")


def test_b_leg_only_book_is_flattened(tmp_path):
    """Novak M4: a book holding only the B leg must be closed, not left naked forever."""
    import json
    p = simulate_panel(PanelConfig(n_days=504 + 126, kinds=(("coint", 6), ("null", 4)), seed=11))
    cfg = S2.with_(calibration_sims=40)
    days = p.prices.index[504:]
    book = tmp_path / "m4"
    for d in days[:5]:
        run_paper(p.prices.loc[:d], cfg, book, p.sectors)
    st = json.loads((book / "state.json").read_text())
    a, b = sorted(st["last_summary"]["pairs"], key=lambda x: x["a"])[0]["a"], None
    pair = st["last_summary"]["pairs"][0]
    st["book"][f"{pair['a']}/{pair['b']}"] = {pair["b"]: -10.0}
    (book / "state.json").write_text(json.dumps(st))
    out = run_paper(p.prices.loc[:days[5]], cfg, book, p.sectors)
    assert any(e.startswith("BROKEN LEGS") for e in out["events"])
    assert any(o["ticker"] == pair["b"] and o["kind"] == "close" for o in out["new_orders"])
