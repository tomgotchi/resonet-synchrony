"""Latent linear dynamical system with explicit observation noise, fit by EM (Shumway & Stoffer 1982).

    x_t = A x_{t-1} + w_t,   w_t ~ N(0, Q)      latent: one 2-D oscillator state per channel (dim 2M)
    y_t = C x_t + v_t,       v_t ~ N(0, R)      C picks each oscillator's first coordinate; R DIAGONAL

Coupling between units lives in the off-diagonal 2x2 blocks of A (lagged influence). Shared (common) input lives in
the off-diagonal blocks of Q (instantaneous state noise). Independent sensor noise is absorbed by the diagonal R,
which is the point: the previous instantaneous statistic was diluted by sensor noise.

All trials have the same length and parameters, so the Kalman gains and covariances are identical across trials and
are computed once per step. Only the state means are batched over trials.
"""
from __future__ import annotations

import numpy as np


def _init(M, fs, f0=8.0, r=0.98, q=0.05, rv=0.5):
    th = 2 * np.pi * f0 / fs
    rot = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = np.kron(np.eye(M), rot)
    Q = q * np.eye(2 * M)
    R = rv * np.ones(M)
    C = np.zeros((M, 2 * M)); C[np.arange(M), 2 * np.arange(M)] = 1.0
    return A, Q, R, C


def em_fit(X, fs=250.0, n_iter=60, tol=1e-5, max_trials=None, seed=0, verbose=False):
    """X: (n_trials, M, T) z-scored observations. Returns dict(A, Q, R, C, loglik history)."""
    X = np.asarray(X, float)
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-12)
    if max_trials and len(X) > max_trials:
        X = X[np.random.default_rng(seed).choice(len(X), max_trials, replace=False)]
    n, M, T = X.shape
    D = 2 * M
    A, Q, R, C = _init(M, fs)
    m0, P0 = np.zeros(D), np.eye(D)
    Y = X.transpose(2, 0, 1)                                       # (T, n, M)
    hist = []
    for it in range(n_iter):
        # ---- forward (covariances shared across trials)
        Pp = np.zeros((T, D, D)); Pf = np.zeros((T, D, D))
        mp = np.zeros((T, n, D)); mf = np.zeros((T, n, D))
        ll = 0.0
        m, P = np.tile(m0, (n, 1)), P0.copy()
        for t in range(T):
            if t > 0:
                m = m @ A.T; P = A @ P @ A.T + Q
            mp[t], Pp[t] = m, P
            S = C @ P @ C.T + np.diag(R)
            Si = np.linalg.inv(S)
            K = P @ C.T @ Si
            innov = Y[t] - m @ C.T                                 # (n, M)
            m = m + innov @ K.T
            P = P - K @ C @ P
            P = (P + P.T) / 2
            mf[t], Pf[t] = m, P
            _, logdet = np.linalg.slogdet(S)
            ll += -0.5 * (n * logdet + np.einsum("ni,ij,nj->", innov, Si, innov) + n * M * np.log(2 * np.pi))
        hist.append(ll / (n * T))
        if verbose:
            print(f"  EM it {it:3d}  ll/sample {hist[-1]:.5f}")
        if it > 2 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):
            break
        # ---- RTS smoother
        ms = mf.copy(); Ps = Pf.copy(); J = np.zeros((T, D, D))
        for t in range(T - 2, -1, -1):
            J[t] = Pf[t] @ A.T @ np.linalg.inv(Pp[t + 1])
            ms[t] = mf[t] + (ms[t + 1] - mp[t + 1]) @ J[t].T
            Ps[t] = Pf[t] + J[t] @ (Ps[t + 1] - Pp[t + 1]) @ J[t].T
        # lag-one smoothed cross-covariance Cov(x_t, x_{t-1}) = Ps[t] J[t-1]^T
        # ---- M-step (sums over trials and time)
        S11 = np.zeros((D, D)); S10 = np.zeros((D, D)); S00 = np.zeros((D, D))
        for t in range(1, T):
            S11 += n * Ps[t] + ms[t].T @ ms[t]
            S10 += n * Ps[t] @ J[t - 1].T + ms[t].T @ ms[t - 1]
            S00 += n * Ps[t - 1] + ms[t - 1].T @ ms[t - 1]
        A = S10 @ np.linalg.inv(S00)
        Q = (S11 - A @ S10.T) / (n * (T - 1))
        Q = (Q + Q.T) / 2 + 1e-8 * np.eye(D)
        r = np.zeros(M)
        for t in range(T):
            e = Y[t] - ms[t] @ C.T
            r += (e ** 2).sum(0) + n * np.einsum("ij,jk,ik->i", C, Ps[t], C)
        R = np.maximum(r / (n * T), 1e-6)
    return dict(A=A, Q=Q, R=R, C=C, loglik=hist, n_trials=n, iterations=len(hist))


def block_stats(fit):
    """S_A: mean over unit pairs of ||A_ij||_F / sqrt(||A_ii|| ||A_jj||)   (lagged coupling)
    S_Q: mean over unit pairs of ||Q_ij||_F / sqrt(||Q_ii|| ||Q_jj||)      (instantaneous shared input)
    noise_frac: mean R / (R + C P_ss C^T) is not needed; reported R mean instead."""
    A, Q = fit["A"], fit["Q"]
    M = A.shape[0] // 2
    nb = lambda Mx, i, j: np.linalg.norm(Mx[2 * i:2 * i + 2, 2 * j:2 * j + 2])
    sa, sq = [], []
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            sa.append(nb(A, i, j) / np.sqrt(nb(A, i, i) * nb(A, j, j)))
            if i < j:
                sq.append(nb(Q, i, j) / np.sqrt(nb(Q, i, i) * nb(Q, j, j)))
    return dict(S_A=float(np.mean(sa)), S_Q=float(np.mean(sq)), R_mean=float(np.mean(fit["R"])))


def simulate_denoised(fit, n=200, T=250, burn=100, seed=0):
    """Noise-free observable process implied by the fit: z_t = C x_t with x_t = A x_{t-1} + w_t (no R).
    Invariant to the latent basis (unlike block norms of A or Q). Returns (n, M, T)."""
    A, Q, C = fit["A"], fit["Q"], fit["C"]
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Q + 1e-10 * np.eye(len(Q)))
    D, M = A.shape[0], C.shape[0]
    x = np.zeros((n, D))
    out = np.zeros((n, M, T))
    for t in range(T + burn):
        x = x @ A.T + rng.standard_normal((n, D)) @ L.T
        if t >= burn:
            out[:, :, t - burn] = x @ C.T
    return out


def denoised_stats(fit, p=25, n=200, T=250, seed=0):
    """VAR statistics (DEVIATIONS 13 definitions) on the model-implied noise-free process:
    S_GC_dn (lagged, conditional Granger) and S_inst_dn (instantaneous residual correlation)."""
    from . import mechanism as Mx
    Z = simulate_denoised(fit, n, T, seed=seed)
    if not np.all(np.isfinite(Z)) or np.abs(Z).max() > 1e6:
        return dict(S_GC_dn=np.nan, S_inst_dn=np.nan, unstable=True)
    GC, C = Mx.gc_inst(Z, p)
    return dict(S_GC_dn=Mx.s_gc(GC), S_inst_dn=Mx.s_inst(C), unstable=False,
                GC=GC.tolist(), inst=C.tolist())
