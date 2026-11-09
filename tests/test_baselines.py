"""Board 2026-11-09 / DEVIATION 23-24: baselines and frozen-bank analyses."""
import numpy as np
import torch

from resonet_sync import baselines as BL
from resonet_sync.screens import calib_p1 as CP


def test_fb_definitions_match_the_calibrated_ones():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((24, 16, 400)).astype(np.float32)
    y = np.repeat(np.arange(4), 6)
    assert np.allclose(BL.fblv(X), CP.fblv(X))
    Wa, Wb = BL.fbcsp_fit(X, y), CP.fbcsp_fit(X, y)
    assert all(np.allclose(a, b) for a, b in zip(Wa, Wb))
    assert np.allclose(BL.fbcsp(X, Wa), CP.fbcsp(X, Wb))


def test_baselines_and_frozen_bank_on_power_data():
    """Class = band power of one channel group: B1/B2 must decode; the frozen-bank analysis runs end to end."""
    from resonet_sync.data_hgd import train_val_split
    from resonet_sync.run import P1, build
    rng = np.random.default_rng(1)
    def make(n):
        y = np.repeat(np.arange(4), n)
        X = rng.standard_normal((len(y), 8, 500))
        from scipy import signal
        alpha = signal.sosfilt(signal.butter(2, (9, 11), "band", fs=250, output="sos"), rng.standard_normal((len(y), 500)))
        X[:, :4] += (1 + y)[:, None, None] * alpha[:, None, :] * 2
        return X.astype(np.float32), y
    Xtr, ytr = make(20); Xte, yte = make(10)
    vmask = train_val_split(ytr, 0.15, 42)
    b = BL.baselines(Xtr, ytr, Xte, yte, vmask, tangent=True)
    assert b["B1_FBLV_LDA"] > 0.5 and b["B2_FBCSP_LDA"] > 0.5 and b["B3_tangent_logreg"] > 0.4
    m = build("lin:sync+power", 8, 500, 250.0, dict(P1, n_osc=6))
    r = BL.frozen_bank_analyses(m, Xtr, ytr, Xte, yte, vmask, n_osc=6)
    assert {"FBCSP+sync", "FBCSP+surrogate_random", "FBLV+sync", "probe:surrogate"} <= set(r)
    assert all(0 <= v <= 1 for v in r.values())
