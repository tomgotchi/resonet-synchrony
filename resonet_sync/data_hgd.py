"""HGD loading via the published `load_subject` (train-only z-scoring, balanced train cap, FIR
decimation), plus a band check that tells the low-passed cache from the full-band one.

Two caches exist (see reso-net/build_hg_cache.py and PAPER_DRAFT.md S4.2):
  subj_cache/     upstream preprocessing, LOW-PASSED to ~48 Hz (no high-gamma)
  subj_cache_hg/  rebuilt from raw EDFs, 4-125 Hz FIR zero-phase  <- Pitch 1 uses this (full band)
Both: keys X_tr, y_tr, X_te, y_te; 128 EEG channels; 2250 samples (4.5 s) @ 500 Hz; official split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import signal

from ._vendor import rn, set_cache_dir

FS_NATIVE = 500.0


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def band_check(cache_dir, subject=1, n_trials=40, fs=FS_NATIVE):
    """Fraction of (train) power above 60 Hz vs 8-30 Hz on raw cached trials. The low-passed
    cache has essentially nothing above ~48 Hz; the full-band cache keeps 4-125 Hz."""
    with np.load(Path(cache_dir) / f"subj_{subject}.npz") as d:
        X = d["X_tr"][:n_trials].astype(np.float64)
    f, P = signal.welch(X, fs=fs, nperseg=512, axis=-1)
    P = P.mean(axis=(0, 1))
    hi = P[(f >= 60) & (f <= 120)].mean()
    mid = P[(f >= 8) & (f <= 30)].mean()
    ratio_db = float(10 * np.log10(hi / mid))
    return dict(hi_over_mid_db=ratio_db, full_band=bool(ratio_db > -40),
                verdict="full-band (high-gamma present)" if ratio_db > -40 else "LOW-PASSED (no high-gamma)")


def load(cache_dir, subject, decim=2, cap=80, seed=42):
    set_cache_dir(cache_dir)
    R = rn()
    Xtr, ytr, Xte, yte = R.load_subject(subject, decim=decim, max_train_per_class=cap, seed=seed)
    return Xtr, ytr, Xte, yte


def train_val_split(y_tr, val_frac=0.15, seed=42):
    """Published stratified val mask (same one load_subject used to exclude val from z-scoring)."""
    return rn().stratified_trial_split(np.arange(len(y_tr)), y_tr, val_frac, seed)
