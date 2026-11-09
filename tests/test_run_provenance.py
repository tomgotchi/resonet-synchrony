"""Board 2026-11-09 (Novak, Halvorsen): resume keeps provenance, refuses silent config/data changes, stores learning
curves; Holm counts only tested rows; no inferential output below the required n."""
import json

import numpy as np
import pytest
import torch

from resonet_sync import data_hgd
from resonet_sync import run as R
from resonet_sync.stats import contrast_table, holm
from resonet_sync.synth_smoke import tiny_dataset


def test_holm_counts_only_tested_rows():
    rng = np.random.default_rng(0)
    per = [{"a": 0.6 + 0.01 * rng.standard_normal(), "b": 0.5 + 0.01 * rng.standard_normal(),
            "c": 0.55 + 0.01 * rng.standard_normal()} for _ in range(12)]
    rows = contrast_table(per, [("a-b", "a", "b"), ("c-b", "c", "b"), ("x-b", "x", "b"), ("y-b", "y", "b")])
    assert [r["n"] for r in rows] == [12, 12, 0, 0]
    assert all(r["holm_m"] == 2 for r in rows)
    assert np.allclose([rows[0]["p_holm"], rows[1]["p_holm"]], holm([rows[0]["p"], rows[1]["p"]]))
    assert np.isnan(rows[2]["p_holm"]) and np.isnan(rows[3]["p_holm"])


def test_min_n_makes_rows_descriptive():
    per = [{"a": 0.6 + 0.01 * i, "b": 0.5} for i in range(6)]
    r = contrast_table(per, [("a-b", "a", "b")], min_n=14)[0]
    assert r["incomplete"] and np.isnan(r["p"]) and np.isnan(r["p_holm"]) and np.isnan(r["ci95"]).all()
    assert r["mean_diff"] > 0 and r["wins"] == 6 and r["holm_m"] == 0
    r = contrast_table(per, [("a-b", "a", "b")], min_n=6)[0]
    assert not r["incomplete"] and r["p"] is not None and r["holm_m"] == 1


def test_contrast_table_unchanged_when_all_rows_tested():
    """Screens and Phase 2 (every row tested) must give exactly the old numbers."""
    rng = np.random.default_rng(1)
    per = [{k: rng.uniform() for k in "abc"} for _ in range(10)]
    rows = contrast_table(per, [("a-b", "a", "b"), ("c-b", "c", "b")])
    assert np.allclose([r["p_holm"] for r in rows], holm([r["p"] for r in rows]))


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    Xtr, ytr, Xte, yte = tiny_dataset(seed=0)
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "subj_1.npz").write_bytes(b"x")
    monkeypatch.setattr(data_hgd, "band_check", lambda c: {"full_band": True})
    monkeypatch.setattr(data_hgd, "load", lambda c, s, d, cap, seed: (Xtr, ytr, Xte, yte))
    monkeypatch.setattr(data_hgd, "sha256", lambda p: "sha-A")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "P1", dict(R.P1, n_osc=6, batch=32))
    return cache


def _args(cache, arms, epochs=2):
    return R.argparse.Namespace(cache=str(cache), arms=arms, subjects="1", epochs=epochs, rk4_substeps=1, smoke=False)


def test_resume_keeps_first_provenance_and_stores_curves(fake_cache, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "git_hash", lambda: "OLD")
    R.phase1(_args(fake_cache, "lin:power"))
    monkeypatch.setattr(R, "git_hash", lambda: "NEW")
    R.phase1(_args(fake_cache, "lin:power,lin:sync"))
    rec = json.loads((tmp_path / "results" / "phase1" / "subj_01.json").read_text())
    assert rec["git"] == "OLD" and rec["first_run"]["git"] == "OLD"
    assert rec["arms"]["lin:power"]["provenance"]["git"] == "OLD"
    assert rec["arms"]["lin:sync"]["provenance"]["git"] == "NEW"
    assert [r["arms"] for r in rec["resumes"]] == [["lin:sync"]] and rec["resumes"][0]["git"] == "NEW"
    c = rec["arms"]["lin:sync"]["curves"]
    assert len(c["val_loss"]) == rec["arms"]["lin:sync"]["epochs_run"] == len(c["train_loss"])
    assert abs(min(c["val_loss"]) - rec["arms"]["lin:sync"]["val_loss_best"]) < 1e-5
    w = tmp_path / "results" / "phase1" / "weights" / "subj_01"
    assert sorted(p.name for p in w.glob("*.pt")) == ["lin__power.pt", "lin__sync.pt"]
    sd = torch.load(w / "lin__sync.pt")
    assert any(k.startswith("bank.") for k in sd) and any(k.startswith("head.") for k in sd)
    assert not list((tmp_path / "results" / "phase1").glob("*.tmp"))
    s = json.loads((tmp_path / "results" / "phase1" / "summary.json").read_text())
    assert s["n_per_arm"] == {"lin:power": 1, "lin:sync": 1} and not s["complete"]


def test_resume_refuses_config_or_data_change(fake_cache, monkeypatch):
    R.phase1(_args(fake_cache, "lin:power"))
    with pytest.raises(SystemExit, match="config differs"):
        R.phase1(_args(fake_cache, "lin:power,lin:sync", epochs=3))
    monkeypatch.setattr(data_hgd, "sha256", lambda p: "sha-B")
    with pytest.raises(SystemExit, match="data file changed"):
        R.phase1(_args(fake_cache, "lin:power,lin:sync"))


def test_per_arm_result_identical_with_or_without_resume(fake_cache, tmp_path):
    """Logging changes must not change results: an arm run after a resume equals the same arm run fresh."""
    R.phase1(_args(fake_cache, "lin:power"))
    R.phase1(_args(fake_cache, "lin:power,lin:sync"))
    a = json.loads((tmp_path / "results" / "phase1" / "subj_01.json").read_text())["arms"]["lin:sync"]
    (tmp_path / "results" / "phase1" / "subj_01.json").unlink()
    R.phase1(_args(fake_cache, "lin:sync"))
    b = json.loads((tmp_path / "results" / "phase1" / "subj_01.json").read_text())["arms"]["lin:sync"]
    for k in ("test_acc", "test_bal_acc", "val_loss_best", "best_epoch", "curves"):
        assert a[k] == b[k], k


def test_confirmatory_families_are_declared_over_two_banks():
    per = [{f"{b}:{r}": 0.5 + 0.01 * i for b in R.BANKS for r in ("sync", "surrogate", "power", "sync+power",
                                                                   "power+surrogate")} for i in range(14)]
    fam = R.contrasts_phase1(per, min_n=14)
    for k in ("H2_sync_vs_surrogate", "incremental_sync_over_power", "incremental_capacity_matched"):
        assert [r["contrast"].split(":")[0] for r in fam[k]] == ["lin", "nl_coupled"]
    assert [r["contrast"] for r in fam["coupling"]] == ["nl_coupled-lin (sync)"]
    assert len(fam["exploratory_other_banks"]) == 5


def test_sign_flip_exact_and_tost():
    from resonet_sync.stats import sign_flip_p, tost_p
    d = np.array([0.01] * 14)
    assert sign_flip_p(d) == pytest.approx(2 / 2 ** 14)            # only the all-+ and all-- patterns
    assert sign_flip_p(np.r_[0.01, -0.01]) == 1.0
    assert tost_p(np.full(14, 0.0) + np.linspace(-0.01, 0.01, 14), 0.05) < 1e-6
    assert tost_p(np.full(14, 0.2) + np.linspace(-0.01, 0.01, 14), 0.05) > 0.99


def test_phase1_rows_outcomes():
    rng = np.random.default_rng(3)
    per = [{"s": 0.6 + 0.02 * rng.standard_normal(), "z": 0.5 + 0.02 * rng.standard_normal(),
            "e": 0.5 + 0.005 * rng.standard_normal()} for _ in range(14)]
    rows = contrast_table(per, [("s-z", "s", "z"), ("e-z", "e", "z")], min_n=14, phase1=True)
    assert rows[0]["outcome"] == "superior" and rows[0]["holm_m"] == 2
    assert rows[1]["outcome"] in ("equivalent", "inconclusive") and rows[1]["p_tost"] is not None
    assert "_d" not in rows[0]
    short = contrast_table(per[:9], [("s-z", "s", "z")], min_n=14, phase1=True)[0]
    assert short["incomplete"] and short["outcome"] is None and np.isnan(short["p"])
    # an older notebook copy formats these fields; NaN must not crash it
    f"{short['ci95'][0]:+.3f} {short['dz']:+.2f} {short['p_holm']:.4f}"
