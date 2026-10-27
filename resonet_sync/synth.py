"""Phase 2 (H0) generators: class information that is DRIVEN by a common input vs EMERGENT from coupling.

Both: M = 8 units observed as 8 channels, simulated by Euler-Maruyama at 1000 Hz, 1 s burn-in discarded,
then a 1 s window decimated to 250 Hz (offline data generation; zero-phase decimation is fine here, as in
TASK_PITCH1 S5 Phase 3). Every channel of every trial is z-scored, so per-channel power carries no class
information by construction (power-matched). 4 classes; a seed fixes the class templates and all trials.

DRIVEN (units uncoupled; class lives only in the input):
    dz_j = (-g + i w_j) z_j dt + kappa * s(t) * exp(i phi_{k,j}) dt + sigma dW_j
  s(t) is a narrowband (8 Hz, ~2 Hz wide) complex noise process drawn fresh per trial and shared by all
  units; phi_k is the class's fixed phase pattern. No coupling between units.

EMERGENT (input carries no class information; class lives only in the coupling matrix):
    dz_j = (lam + i w_j - |z_j|^2) z_j dt + (K / deg) sum_l A^(k)_{jl} (z_l - z_j) dt + sigma dW_j
  Noise-driven Stuart-Landau units (lam < 0: damped, so oscillations are sustained by noise).
  A^(k): class-specific random symmetric 0/1 graph (density 0.35, connected), fixed per seed.
  Inputs are independent white noise, identical in distribution across classes.

Natural frequencies w_j ~ 8 Hz +/- 1 Hz (fixed per seed, shared by all classes).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FS_SIM, FS_OUT, M = 1000.0, 250.0, 8
DEFAULTS = dict(
    driven=dict(g=2 * np.pi * 1.0, kappa=6.5, sigma=6.0, input_bw_hz=2.0, input_f_hz=8.0),
    emergent=dict(lam=-2 * np.pi * 1.0, K=40.0, sigma=6.0, density=0.35),
)


def _graph(rng, density):
    while True:
        A = (rng.random((M, M)) < density).astype(float)
        A = np.triu(A, 1); A = A + A.T
        L = np.diag(A.sum(1)) - A
        if np.sort(np.linalg.eigvalsh(L))[1] > 1e-6:      # connected
            return A


def templates(seed, kind, p):
    rng = np.random.default_rng([seed, 7])
    w = 2 * np.pi * (8.0 + rng.uniform(-1, 1, M))
    if kind == "driven":
        return dict(w=w, phi=rng.uniform(0, 2 * np.pi, (4, M)))
    return dict(w=w, A=np.stack([_graph(rng, p["density"]) for _ in range(4)]))


def _trial(kind, k, tpl, p, rng, burn_s=1.0, win_s=1.0):
    dt = 1 / FS_SIM
    n = int((burn_s + win_s) * FS_SIM)
    z = 0.1 * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    out = np.empty((n, M), complex)
    sq = np.sqrt(dt)
    dW = (rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M))) * sq / np.sqrt(2)
    w = tpl["w"]
    if kind == "driven":
        a = np.exp((-2 * np.pi * p["input_bw_hz"] + 1j * 2 * np.pi * p["input_f_hz"]) * dt)
        e = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        s = signal.lfilter([np.sqrt(1 - abs(a) ** 2)], [1, -a], e)        # unit-variance narrowband input
        drive = np.exp(1j * tpl["phi"][k])
        for t in range(n):
            z = z + ((-p["g"] + 1j * w) * z + p["kappa"] * s[t] * drive) * dt + p["sigma"] * dW[t]
            out[t] = z
    else:
        A = tpl["A"][k]
        deg = max(A.sum(1).mean(), 1.0)
        for t in range(n):
            cpl = (A @ z - A.sum(1) * z) * (p["K"] / deg)
            z = z + ((p["lam"] + 1j * w - np.abs(z) ** 2) * z + cpl) * dt + p["sigma"] * dW[t]
            out[t] = z
    x = out[int(burn_s * FS_SIM):].real.T                                    # (M, T_sim) observed
    x = signal.decimate(x, int(FS_SIM / FS_OUT), ftype="fir", zero_phase=True, axis=1)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)   # power-matched
    return x.astype(np.float32)


def make_dataset(seed, kind, n_train_per_class=100, n_test_per_class=40, params=None):
    p = dict(DEFAULTS[kind], **(params or {}))
    tpl = templates(seed, kind, p)
    rng = np.random.default_rng([seed, 11, 0 if kind == "driven" else 1])

    def block(n):
        X, y = [], []
        for k in range(4):
            for _ in range(n):
                X.append(_trial(kind, k, tpl, p, rng)); y.append(k)
        X, y = np.stack(X), np.asarray(y)
        o = rng.permutation(len(y))
        return X[o], y[o]

    Xtr, ytr = block(n_train_per_class)
    Xte, yte = block(n_test_per_class)
    return Xtr, ytr, Xte, yte, dict(seed=seed, kind=kind, params=p, fs=FS_OUT, M=M)
