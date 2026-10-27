"""Data sanitization helpers: non-finite values, flat channels, clipping, line noise, fs checks.

All functions take X shaped (n_channels, n_samples) or (n_samples,) and return plain dicts / arrays
so they can be logged into result JSONs.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _2d(X):
    X = np.asarray(X)
    return X[None, :] if X.ndim == 1 else X


def nonfinite_report(X):
    X = _2d(X)
    bad = ~np.isfinite(X)
    return dict(n_nonfinite=int(bad.sum()), channels=np.flatnonzero(bad.any(1)).tolist(),
                frac=float(bad.mean()))


def fill_nonfinite(X):
    """Replace NaN/inf by linear interpolation along time (per channel); all-bad channels -> 0.
    Uses past and future neighbours: for OFFLINE cleaning only, never inside an estimator."""
    X = _2d(np.array(X, dtype=float, copy=True))
    t = np.arange(X.shape[1])
    for c in range(X.shape[0]):
        good = np.isfinite(X[c])
        if not good.any():
            X[c] = 0.0
        elif not good.all():
            X[c, ~good] = np.interp(t[~good], t[good], X[c, good])
    return X


def flat_channels(X, rel_tol=1e-6, max_const_run_s=None, fs=None):
    """Channels whose std is < rel_tol * median channel std, or (if max_const_run_s and fs given)
    that contain a run of identical consecutive values longer than max_const_run_s."""
    X = _2d(X)
    sd = np.nanstd(X, axis=1)
    ref = np.nanmedian(sd[sd > 0]) if np.any(sd > 0) else 1.0
    flat = set(np.flatnonzero(sd < rel_tol * ref).tolist())
    if max_const_run_s and fs:
        lim = int(max_const_run_s * fs)
        for c in range(X.shape[0]):
            if longest_const_run(X[c]) > lim:
                flat.add(c)
    return sorted(flat)


def longest_const_run(x):
    d = np.diff(np.asarray(x)) == 0
    if not d.any():
        return 1
    idx = np.flatnonzero(np.diff(np.concatenate([[0], d.astype(int), [0]])))
    return int((idx[1::2] - idx[::2]).max() + 1)


def clipping_report(X, rail=None, min_run=3, frac_thresh=1e-3):
    """Detect clipping: samples equal to the channel's extreme value (or to +/-rail, e.g. int16
    32767/-32768) occurring in runs >= min_run. Flags channels whose clipped fraction > frac_thresh."""
    X = _2d(X)
    out = []
    for c in range(X.shape[0]):
        x = X[c]
        hi, lo = (rail, -rail - 1) if rail is not None else (np.nanmax(x), np.nanmin(x))
        at = (x >= hi) | (x <= lo)
        runs = _runs(at)
        clipped = sum(r for r in runs if r >= min_run)
        out.append(clipped / x.size)
    out = np.asarray(out)
    return dict(clipped_frac=out.tolist(), flagged=np.flatnonzero(out > frac_thresh).tolist())


def _runs(b):
    b = np.asarray(b, int)
    if not b.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], b, [0]])))
    return (idx[1::2] - idx[::2]).tolist()


def line_noise(X, fs, candidates=(50.0, 60.0), half_band=1.0, nperseg_s=4.0):
    """Line-noise prominence (dB) at each candidate frequency: peak PSD in [f +/- 0.5 Hz] vs. median
    PSD in the flanks [f-half_band-2, f-half_band] U [f+half_band, f+half_band+2]. Averaged over
    channels. Returns the dominant candidate."""
    X = _2d(X)
    f, P = signal.welch(X, fs=fs, nperseg=min(X.shape[1], int(nperseg_s * fs)), axis=-1)
    P = P.mean(0)
    res = {}
    for f0 in candidates:
        if f0 + half_band + 2 >= fs / 2:
            res[f0] = np.nan
            continue
        pk = P[(f >= f0 - 0.5) & (f <= f0 + 0.5)].max()
        fl = P[((f >= f0 - half_band - 2) & (f <= f0 - half_band)) |
               ((f >= f0 + half_band) & (f <= f0 + half_band + 2))]
        res[f0] = float(10 * np.log10(pk / np.median(fl)))
    valid = {k: v for k, v in res.items() if np.isfinite(v)}
    dom = max(valid, key=valid.get) if valid else None
    return dict(prominence_db={str(k): v for k, v in res.items()}, dominant=dom)


def verify_fs(declared_fs, n_samples=None, duration_s=None, expected_fs=None, rel_tol=1e-3):
    """Check a declared sampling rate against (n_samples / duration) and/or an expected value
    (e.g. from an .xml header). Returns dict(ok=bool, ...). Raises nothing; caller decides."""
    checks = {}
    if n_samples is not None and duration_s:
        implied = n_samples / duration_s
        checks["implied_fs"] = implied
        checks["implied_ok"] = abs(implied - declared_fs) <= rel_tol * declared_fs
    if expected_fs is not None:
        checks["expected_fs"] = expected_fs
        checks["expected_ok"] = abs(expected_fs - declared_fs) <= rel_tol * declared_fs
    ok = all(v for k, v in checks.items() if k.endswith("_ok"))
    return dict(declared_fs=declared_fs, ok=bool(ok), **checks)


def line_noise_fs_check(X, declared_fs, mains=(50.0, 60.0), search_hz=3.0):
    """Sanity check of fs from the line-noise peak: if the strongest narrow peak near a mains
    frequency sits at f_obs instead of f_mains, the true fs is ~ declared_fs * f_mains / f_obs."""
    X = _2d(X)
    f, P = signal.welch(X, fs=declared_fs, nperseg=min(X.shape[1], int(8 * declared_fs)), axis=-1)
    P = P.mean(0)
    out = {}
    for m in mains:
        sel = (f >= m - search_hz) & (f <= m + search_hz)
        if sel.sum() < 3:
            continue
        f_obs = float(f[sel][np.argmax(P[sel])])
        out[str(m)] = dict(f_obs=f_obs, implied_fs=declared_fs * m / f_obs)
    return out


def sanitize_report(X, fs, rail=None):
    return dict(nonfinite=nonfinite_report(X), flat=flat_channels(X, fs=fs, max_const_run_s=1.0),
                clipping=clipping_report(X, rail=rail), line_noise=line_noise(X, fs))
