"""Fixes from the review board of 2026-11-17 (docs/FIX_BRIEF.md): control analysis rule, surrogate retention,
all-subject forensics, the nonlinear branch of the null test, and the Colab notebook."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT.parent / "results_hg"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- bandwidth_analysis (DEVIATION 29, addendum 1)
def _cells(band, rate, n=14, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    BA = _load("bandwidth_analysis")
    base = rng.uniform(0.35, 0.45, n)
    mk = lambda off: {s + 1: {a: float(base[s] + off + rng.normal(0, noise)) for a in BA.ARMS} for s in range(n)}
    return BA, mk(0.0), mk(band + rate), mk(rate)                      # R, F, A


@pytest.mark.parametrize("band,rate,expect", [(0.14, 0.0, "mostly band"), (0.0, 0.14, "mostly sampling rate"),
                                              (0.0, 0.0, "no gain to attribute")])
def test_reading_rule(band, rate, expect):
    BA, R, F, A = _cells(band, rate)
    d = BA.decompose(R, F, A, "nl_sync")
    assert d["reading"] == expect
    assert abs(d["T"]["mean"] - d["Bn"]["mean"] - d["Rt"]["mean"]) < 1e-12        # T = Bn + Rt exactly


def test_reading_rule_mixed_and_missing_cells(tmp_path):
    BA, R, F, A = _cells(0.03, 0.06, noise=0.06, seed=3)
    d = BA.decompose(R, F, A, "nl_sync")
    assert d["reading"] in ("mixed or unresolved", "mostly sampling rate", "mostly band", "no gain to attribute")
    T = dict(mean=0.10, ci95=[0.05, 0.15]); half = dict(mean=0.05, ci95=[0.02, 0.08]); small = dict(mean=0.04, ci95=[0.01, 0.07])
    assert BA.reading(T, small, dict(mean=0.06, ci95=[0.03, 0.09])) == "mixed or unresolved"   # band real but < half, rate real
    assert BA.reading(T, half, half) == "mostly band"
    res = BA.analyse(R, F, A)                                                       # no B, F', R', cleaned: must run
    assert set(res["arms"]) == set(BA.ARMS) and res["pipeline_check"] == {}
    Rp = {s: R[s] for s in (1, 2, 3, 4)}
    res = BA.analyse(R, F, A, B=A, Fp={s: F[s] for s in (1, 2)}, Rp=Rp, clean=F)
    assert res["arms"]["nl_sync"]["with_rerun_restricted"]["T"]["n"] == 4
    assert res["pipeline_check"]["repro_full_decim2 - published full"]["n_cells"] == 12
    assert res["pipeline_check"]["repro_full_decim2 - published full"]["comparable"]
    for tag, cell in (("lp48_decim2", A),):                                         # file round trip
        (tmp_path / tag).mkdir()
        for s, acc in cell.items():
            (tmp_path / tag / f"subj_{s:02d}.json").write_text(json.dumps(dict(subject=s, acc=acc)))
    assert BA.load_new(tmp_path / "lp48_decim2") == A
    assert BA.main(["--published", str(tmp_path / "nowhere"), "--new", str(tmp_path)]) is not None


# ---------------------------------------------------------------- surrogate_retention (DEVIATION 31)
def test_retention_zero_shift_is_one_and_far_shift_removes_alignment():
    SR = _load("surrogate_retention")
    g = torch.Generator().manual_seed(0)
    ph = torch.cumsum(0.3 * torch.randn(8, 1000, 1, generator=g), 1)               # one common random-walk phase
    u = torch.exp(1j * (ph + 0.1 * torch.randn(8, 1000, 4, generator=g)))          # 4 oscillators locked to it
    r0 = SR.retention_from_phasors(u, torch.zeros(4, dtype=torch.long))
    assert abs(r0["ratio"] - 1) < 1e-6 and r0["rms_unshifted"] > 0.9
    r1 = SR.retention_from_phasors(u, torch.tensor([0, 250, 500, 750]))
    assert r1["ratio"] < 0.5 and sum(b["n_pairs"] for b in r1["by_relative_shift"].values()) == 6


@pytest.mark.skipif(not (CKPT / "RNsync_S1.pt").exists(), reason="published checkpoints not present")
def test_forensics_cover_all_checkpoints():
    PF = _load("published_forensics")
    ck = PF.checkpoints(CKPT)
    assert len(ck) == 14
    assert min(r["min_correlation_between_gain_curves"] for r in ck) > 0.9995
    assert max(r["max_hermitian_eig"] for r in ck) < -0.25                          # linear part contractive, all 14
    assert all(1.8 < r["plus_minus_selectivity_db_10hz"] < 2.0 for r in ck)
    stored = json.loads((ROOT / "results" / "published_forensics.json").read_text())
    assert abs(stored["all_subjects"]["min_correlation_between_gain_curves"] - 0.99957) < 1e-4


# ---------------------------------------------------------------- published_null_test --model nl (DEVIATION 30)
def test_null_test_model_switch(monkeypatch):
    NT = _load("published_null_test")
    assert set(NT.ARMS_NL) == {"power", "sync", "surrogate"} and "sync_imag" in NT.ARMS
    built = []

    import resonet_sync._vendor as V
    real = V.rn()

    class Fake:
        def __getattr__(self, name):
            if name not in ("ResonanceNetwork", "FastResonanceNetwork"):
                return getattr(real, name)

            def ctor(**kw):
                built.append((name, kw["readout"])); raise StopIteration
            return ctor
    monkeypatch.setattr(V, "rn", lambda: Fake())
    diff = json.loads(NT.CP.DIFF_FILE.read_text())["chosen"]
    for model, cls in (("nl", "ResonanceNetwork"), ("lin", "FastResonanceNetwork")):
        with pytest.raises(StopIteration):
            NT.run_one(("G0", 3200, diff, model))
        assert built[-1] == (cls, "power")


# ---------------------------------------------------------------- notebook
def test_colab_notebook_compiles_and_has_the_new_sections():
    nb = json.loads((ROOT / "notebooks" / "resonet_synchrony_full_colab.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ("## 4a", "## 4b", "## 4c", "repro_full_decim2", "repro_lp48_decim4", "bandwidth_analysis.py",
                   "RUN_SURROGATE_RETENTION", "'bandwidth_control'", "all_subjects"):
        assert needle in text, needle
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = re.sub(r"^(\s*)[!%].*$", r"\1pass", "".join(c["source"]), flags=re.M)
            compile(src, "cell", "exec")
