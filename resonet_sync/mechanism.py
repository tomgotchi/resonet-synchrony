"""Mechanism statistics screen (DEVIATIONS 11; synthetic, exploratory). Definitions are frozen there.

All functions take X: (n_trials, M, T) real observed channels at fs and pool over trials. No labels used.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

BAND = (6.0, 10.0)
EDGE_S = 0.1
TE_BINS, TE_DELAY = 8, 8


def analytic(X, fs=250.0, band=BAND, edge_s=EDGE_S):
    sos = signal.butter(4, band, btype="band", fs=fs, output="sos")
    a = signal.hilbert(signal.sosfiltfilt(sos, X, axis=-1), axis=-1)
    e = int(round(edge_s * fs))
    return a[..., e:-e]                                     # (n, M, T')


def s_plv(a):
    u = a / np.abs(a)
    C = np.einsum("nit,njt->nij", u, u.conj()) / u.shape[-1]
    iu, ju = np.triu_indices(a.shape[1], 1)
    return float(np.abs(C[:, iu, ju]).mean())


def s_couple(a, fs=250.0):
    """Multivariate Kuramoto-type regression per unit; mean over ordered pairs of sqrt(a^2+b^2), rad/s."""
    phi = np.angle(a)                                        # (n, M, T')
    n, M, T = phi.shape
    dphi = np.angle(np.exp(1j * np.diff(phi, axis=-1))) * fs  # (n, M, T'-1)
    ph = phi[..., :-1]
    strengths = []
    for i in range(M):
        others = [j for j in range(M) if j != i]
        d = ph[:, others, :] - ph[:, [i], :]                 # (n, M-1, T'-1)
        R = np.concatenate([np.ones((n, 1, T - 1)), np.sin(d), np.cos(d)], axis=1)
        R = R.transpose(0, 2, 1).reshape(-1, 1 + 2 * (M - 1))
        y = dphi[:, i, :].reshape(-1)
        coef, *_ = np.linalg.lstsq(R, y, rcond=None)
        a_, b_ = coef[1:M], coef[M:]
        strengths.extend(np.hypot(a_, b_).tolist())
    return float(np.mean(strengths))


def _H(*cols, bins=TE_BINS):
    idx = np.zeros(cols[0].shape, dtype=np.int64)
    for c in cols:
        idx = idx * bins + c
    p = np.bincount(idx, minlength=bins ** len(cols)).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def te_matrix(a, bins=TE_BINS, delay=TE_DELAY):
    """TE[j, i] = TE_{j -> i} = I(phi_i(t+d); phi_j(t) | phi_i(t)), plug-in, bits; samples pooled within trials."""
    phi = np.angle(a)
    q = np.clip(((phi + np.pi) / (2 * np.pi) * bins).astype(np.int64), 0, bins - 1)   # (n, M, T')
    past, fut = q[..., :-delay], q[..., delay:]
    M = q.shape[1]
    TE = np.zeros((M, M))
    for i in range(M):
        xf, xp = fut[:, i].ravel(), past[:, i].ravel()
        h_fp, h_p = _H(xf, xp, bins=bins), _H(xp, bins=bins)
        for j in range(M):
            if j == i:
                continue
            yp = past[:, j].ravel()
            TE[j, i] = h_fp + _H(xp, yp, bins=bins) - h_p - _H(xf, xp, yp, bins=bins)
    return TE


def te_matrix_cross_trial(a, bins=TE_BINS, delay=TE_DELAY, seed=0):
    """As te_matrix, but the source unit j comes from a DIFFERENT trial (fixed random derangement): the
    bias / own-phase-predictability floor of the plug-in TE (DEVIATIONS 12, secondary)."""
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    phi = np.angle(a)
    q = np.clip(((phi + np.pi) / (2 * np.pi) * bins).astype(np.int64), 0, bins - 1)
    past, fut = q[..., :-delay], q[..., delay:]
    M = q.shape[1]
    TE = np.zeros((M, M))
    for i in range(M):
        xf, xp = fut[:, i].ravel(), past[:, i].ravel()
        h_fp, h_p = _H(xf, xp, bins=bins), _H(xp, bins=bins)
        for j in range(M):
            if j == i:
                continue
            yp = past[perm, j].ravel()
            TE[j, i] = h_fp + _H(xp, yp, bins=bins) - h_p - _H(xf, xp, yp, bins=bins)
    return TE


def s_te(TE):
    M = TE.shape[0]
    return float(TE[~np.eye(M, dtype=bool)].mean())


def s_asym(TE):
    iu, ju = np.triu_indices(TE.shape[0], 1)
    a, b = TE[iu, ju], TE[ju, iu]
    return float(np.mean(np.abs(a - b) / np.maximum(a + b, 1e-12)))


def s_resid(a):
    """Per trial: remove the first complex principal component (common-input estimate) from the analytic
    signals (= complex least-squares regression of each channel on it), then mean pairwise PLV of residuals."""
    out = []
    iu, ju = np.triu_indices(a.shape[1], 1)
    for A in a:                                              # (M, T')
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        R = A - s[0] * np.outer(U[:, 0], Vh[0])
        u = R / (np.abs(R) + 1e-12)
        C = u @ u.conj().T / u.shape[1]
        out.append(np.abs(C[iu, ju]).mean())
    return float(np.mean(out))


def all_stats(X, fs=250.0):
    a = analytic(X, fs)
    TE = te_matrix(a)
    return dict(S_plv=s_plv(a), S_couple=s_couple(a, fs), S_TE=s_te(TE), S_asym=s_asym(TE), S_resid=s_resid(a))
