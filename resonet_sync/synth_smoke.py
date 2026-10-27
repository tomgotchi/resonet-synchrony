"""Tiny synthetic data for smoke tests only (NOT the Phase 2 generators, which go in synth.py).
Class k sets a channel-wise phase-lag pattern of a common 10 Hz rhythm; amplitude is class-invariant."""
import numpy as np


def tiny_dataset(n_per_class_tr=24, n_per_class_te=8, C=8, T=250, fs=250.0, seed=0):
    rng = np.random.default_rng(seed)
    lags = rng.uniform(0, 2 * np.pi, (4, C))
    t = np.arange(T) / fs

    def make(n):
        X, y = [], []
        for k in range(4):
            for _ in range(n):
                ph = 2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)
                X.append(np.cos(ph[None, :] + lags[k][:, None]) + 0.5 * rng.standard_normal((C, T)))
                y.append(k)
        return np.asarray(X, np.float32), np.asarray(y)

    Xtr, ytr = make(n_per_class_tr)
    Xte, yte = make(n_per_class_te)
    return Xtr, ytr, Xte, yte
