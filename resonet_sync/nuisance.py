"""Nuisance adjustments for the within-session rule (DEVIATIONS 20; synthetic, exploratory).

Screen 9 showed two nuisance confounds of Delta S_TE (both raise S_TE without any change in coupling):
  - theta frequency: the TE delay is fixed in samples, so it spans more of a cycle at a higher frequency;
  - SNR: cleaner phase estimates raise S_TE (and lower S_inst_dn, so the relative guard passes).

Fixes, both label-free and computed per regime per session:
  (a) cycle-normalised TE: delay = DELAY_CYC cycles of the regime's own spectral peak f_c, applied as a fractional
      delay on the unwrapped phase (no integer rounding);
  (b) noise matching: the regime with the smaller latent-LDS noise fraction (diagonal R of the EM fit on z-scored
      data) gets white noise added per channel until its noise fraction equals the other regime's; then both
      statistics are recomputed on the degraded data.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import mechanism as Mx

DELAY_CYC = 8 * 8.0 / 250.0          # 0.256 cycles = the old 8-sample delay at 8 Hz / 250 Hz


def peak_freq(X, fs=250.0, band=Mx.BAND):
    """Welch peak (pooled over trials and channels) inside the band; parabolic interpolation between bins."""
    f, P = signal.welch(np.asarray(X, float), fs=fs, nperseg=min(X.shape[-1], int(fs)), nfft=int(8 * fs), axis=-1)
    P = P.reshape(-1, P.shape[-1]).mean(0)
    m = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    k = m[np.argmax(P[m])]
    if 0 < k < len(P) - 1:
        a, b, c = np.log(P[k - 1:k + 2])
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(f[k] + d * (f[1] - f[0]))
    return float(f[k])


def te_matrix_frac(a, delay, bins=Mx.TE_BINS):
    """As mechanism.te_matrix, with a FRACTIONAL delay (samples): the future phase is read from the unwrapped phase
    by linear interpolation at t + delay."""
    phi = np.unwrap(np.angle(a), axis=-1)
    T = phi.shape[-1]
    k = int(np.floor(delay)); w = delay - k
    if w == 0:
        n_out = T - k
        fut_phi = phi[..., k:]
    else:
        n_out = T - k - 1
        fut_phi = (1 - w) * phi[..., k:k + n_out] + w * phi[..., k + 1:k + 1 + n_out]
    q = lambda p: np.clip((((np.angle(np.exp(1j * p)) + np.pi) / (2 * np.pi)) * bins).astype(np.int64), 0, bins - 1)
    past, fut = q(phi[..., :n_out]), q(fut_phi)
    M = phi.shape[1]
    TE = np.zeros((M, M))
    for i in range(M):
        xf, xp = fut[:, i].ravel(), past[:, i].ravel()
        h_fp, h_p = Mx._H(xf, xp, bins=bins), Mx._H(xp, bins=bins)
        for j in range(M):
            if j != i:
                yp = past[:, j].ravel()
                TE[j, i] = h_fp + Mx._H(xp, yp, bins=bins) - h_p - Mx._H(xf, xp, yp, bins=bins)
    return TE


def s_te_cyc(X, fs=250.0, delay_cyc=DELAY_CYC):
    f_c = peak_freq(X, fs)
    return Mx.s_te(te_matrix_frac(Mx.analytic(X, fs), delay_cyc * fs / f_c)), f_c


def zscore(X):
    X = np.asarray(X, float)
    return (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-12)


def degrade_to(X, r_have, r_target, seed=0):
    """Add white noise per channel to z-scored X so the noise fraction goes from r_have to r_target (>= r_have):
    (r + v) / (1 + v) = r_target  ->  v = (r_target - r) / (1 - r_target)."""
    Z = zscore(X)
    r_have, r_target = np.asarray(r_have, float), np.asarray(r_target, float)
    v = np.clip((r_target - r_have) / np.maximum(1 - r_target, 1e-6), 0.0, None)
    rng = np.random.default_rng(seed)
    return Z + np.sqrt(v)[None, :, None] * rng.standard_normal(Z.shape), v
