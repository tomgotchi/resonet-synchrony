"""scripts/bandwidth_control.py: the 48 Hz low-passed copy of a cache (the manuscript's missing control)."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("bandwidth_control", ROOT / "scripts" / "bandwidth_control.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_lowpass_copy_removes_the_high_band_and_keeps_everything_else(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 4)
    X = rng.normal(size=(16, 3, 2250)).astype(np.float16)
    np.savez(tmp_path / "subj_1.npz", X_tr=X, y_tr=y, X_te=X[:8], y_te=y[:8])
    _load().make_lowpass(tmp_path, tmp_path / "lp", cutoff=48.0)
    a, b = np.load(tmp_path / "subj_1.npz"), np.load(tmp_path / "lp" / "subj_1.npz")
    assert sorted(a.files) == sorted(b.files)
    assert b["X_tr"].shape == a["X_tr"].shape and b["X_te"].shape == a["X_te"].shape and b["X_tr"].dtype == a["X_tr"].dtype
    assert np.array_equal(a["y_tr"], b["y_tr"]) and np.array_equal(a["y_te"], b["y_te"])
    for k in ("X_tr", "X_te"):
        f, Pa = signal.welch(a[k].astype(np.float32), fs=500, nperseg=500, axis=-1)
        _, Pb = signal.welch(b[k].astype(np.float32), fs=500, nperseg=500, axis=-1)
        hi, lo = f >= 60, (f >= 10) & (f <= 40)
        assert Pb[..., hi].mean() < 1e-3 * Pa[..., hi].mean()                 # at least 30 dB down above 60 Hz
        assert Pb[..., lo].mean() > 0.8 * Pa[..., lo].mean()                   # the pass band is kept
