"""Validity of the evoked-vs-induced decomposition on data where the answer is known."""
import numpy as np
import torch

from resonet_sync.banks import HzOscBank
from resonet_sync.decomposition import cv_discriminability, magnitudes, same_class_partners, trajectories


def _data(kind, n_per=40, C=6, T=250, fs=250.0, seed=0):
    """evoked: class sets channel phase lags AND the absolute phase is fixed (time-locked).
    induced: class sets channel phase lags, absolute phase random per trial (not time-locked)."""
    rng = np.random.default_rng(seed)
    lags = rng.uniform(0, 2 * np.pi, (4, C))
    t = np.arange(T) / fs
    X, y = [], []
    for k in range(4):
        for _ in range(n_per):
            ph0 = 0.0 if kind == "evoked" else rng.uniform(0, 2 * np.pi)
            X.append(np.cos(2 * np.pi * 10 * t + ph0 + lags[k][:, None]) + 0.8 * rng.standard_normal((C, T)))
            y.append(k)
    return np.asarray(X, np.float32), np.asarray(y)


def _u(X):
    torch.manual_seed(0)
    b = HzOscBank(n_channels=X.shape[1], n_osc=6, fs=250.0, band=(4.0, 30.0), kind="lin")
    return trajectories(b, X)


def test_partners_same_class_within_index_set():
    y = np.repeat(np.arange(4), 10)
    idx = np.arange(0, 40, 2)
    p = same_class_partners(y, idx, np.random.default_rng(0))
    assert set(p) <= set(idx) and np.all(y[p] == y[idx]) and np.all(p != idx)


def test_induced_only_data_has_no_evoked_information():
    X, y = _data("induced")
    u = _u(X)
    cv = cv_discriminability(u, y)
    assert cv["cv_bal_acc_genuine"] > 0.6
    assert cv["cv_bal_acc_shuffled"] < 0.45            # near chance (0.25)
    m = magnitudes(u, y, same_class_partners(y, np.arange(len(y)), np.random.default_rng(0)))
    assert m["evoked_fraction"] < 0.3


def test_evoked_data_is_captured_by_shuffled_coherence():
    X, y = _data("evoked")
    u = _u(X)
    cv = cv_discriminability(u, y)
    assert cv["cv_bal_acc_shuffled"] > 0.6
    m = magnitudes(u, y, same_class_partners(y, np.arange(len(y)), np.random.default_rng(0)))
    assert m["evoked_fraction"] > 0.8
