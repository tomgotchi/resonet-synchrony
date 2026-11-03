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

DRIVEN_COMMON (DEVIATIONS 11; no coupling, no oscillatory drive):
    dz_j = (lam + i w_j - |z_j|^2) z_j dt + sigma*sqrt(1-rho^2) dW_j + sigma*rho*exp(i psi_{k,j}) dW_c
  EMERGENT's units with the coupling replaced by an UNOBSERVED shared white-noise source W_c, entering each
  unit with a class-specific phase psi_{k,j}; total noise variance per unit equals EMERGENT's.

Natural frequencies w_j ~ 8 Hz +/- 1 Hz (fixed per seed, shared by all classes).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FS_SIM, FS_OUT, M = 1000.0, 250.0, 8
DEFAULTS = dict(
    driven=dict(g=2 * np.pi * 1.0, kappa=6.5, sigma=6.0, input_bw_hz=2.0, input_f_hz=8.0),
    emergent=dict(lam=-2 * np.pi * 1.0, K=40.0, sigma=6.0, density=0.35),
    # DEVIATIONS 11: emergent's units with coupling replaced by an unobserved shared white-noise source.
    # rho = shared-noise amplitude fraction; calibrated on pilot seed 999 (see DEVIATIONS 11).
    driven_common=dict(lam=-2 * np.pi * 1.0, sigma=6.0, rho=0.85),
)
KINDS = ("emergent", "driven", "driven_common")
# DEVIATIONS 17: coupling AND unobserved common input in one network (within-session contrast tests)
DEFAULTS["mixed"] = dict(lam=-2 * np.pi * 1.0, K=40.0, sigma=6.0, density=0.35, rho=0.5)


def _graph(rng, density, m=M):
    while True:
        A = (rng.random((m, m)) < density).astype(float)
        A = np.triu(A, 1); A = A + A.T
        L = np.diag(A.sum(1)) - A
        if np.sort(np.linalg.eigvalsh(L))[1] > 1e-6:      # connected
            return A


def templates(seed, kind, p):
    m = p.get("M", M)                                   # DEVIATIONS 12 variants; default M = 8 (unchanged)
    rng = np.random.default_rng([seed, 7])
    w = 2 * np.pi * (8.0 + rng.uniform(-1, 1, m))
    if p.get("f_shift_hz"):                             # DEVIATIONS 19: nuisance frequency shift (after the draw)
        w = w + 2 * np.pi * p["f_shift_hz"]
    if kind == "driven":
        return dict(w=w, phi=rng.uniform(0, 2 * np.pi, (4, m)))
    if kind == "driven_common":
        return dict(w=w, psi=rng.uniform(0, 2 * np.pi, (4, m)))
    if kind == "mixed":
        return dict(w=w, A=np.stack([_graph(rng, p["density"], m) for _ in range(4)]),
                    psi=rng.uniform(0, 2 * np.pi, (4, m)))
    return dict(w=w, A=np.stack([_graph(rng, p["density"], m) for _ in range(4)]))


def _trial(kind, k, tpl, p, rng, burn_s=1.0, win_s=None):
    win_s = p.get("win_s", 1.0) if win_s is None else win_s
    M = len(tpl["w"])
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
    elif kind == "driven_common":
        rho = p["rho"]
        load = rho * np.exp(1j * tpl["psi"][k])                          # class-specific loading
        dWc = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sq / np.sqrt(2)   # unobserved
        sp = p["sigma"] * np.sqrt(1 - rho ** 2)                           # total noise var = sigma^2
        for t in range(n):
            z = z + ((p["lam"] + 1j * w - np.abs(z) ** 2) * z) * dt + sp * dW[t] + p["sigma"] * load * dWc[t]
            out[t] = z
    elif kind == "mixed":
        A = tpl["A"][k]
        deg = max(A.sum(1).mean(), 1.0)
        rho = p["rho"]
        load = rho * np.exp(1j * tpl["psi"][k])
        dWc = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sq / np.sqrt(2)
        sp = p["sigma"] * np.sqrt(1 - rho ** 2)
        # Screen 11 options (review board), all off by default so earlier datasets are byte-identical:
        #   kappa_nb > 0          : additional RHYTHMIC common input s(t) (AR(1) complex, nb_f_hz, nb_bw_hz, unit var)
        #   extra_self_damp_K > 0 : self-damping of a diffusive coupling of that strength, WITHOUT cross terms
        #   integrator == "exact" : exponential (Lawson) step for the linear part + exact OU noise variance, so a
        #                           uniform frequency shift is an exact symmetry of the discrete-time simulator
        kap = p.get("kappa_nb", 0.0)
        if kap:
            a_nb = np.exp((-2 * np.pi * p.get("nb_bw_hz", 2.0) + 1j * 2 * np.pi * p.get("nb_f_hz", 8.0)) * dt)
            r_nb = p["_nb_rng"]                          # dedicated stream: trial noise stays paired with kappa_nb=0
            e_nb = r_nb.standard_normal(n) + 1j * r_nb.standard_normal(n)
            s_nb = signal.lfilter([np.sqrt(1 - abs(a_nb) ** 2)], [1, -a_nb], e_nb) * np.sqrt(0.5)
            drive_nb = kap * np.exp(1j * tpl["psi"][k])
        xs = p.get("extra_self_damp_K", 0.0)
        self_d = -(xs / deg) * A.sum(1) if xs else 0.0
        exact = p.get("integrator") == "exact"
        if exact:
            E = np.exp((p["lam"] + 1j * w) * dt)
            g = np.sqrt((1 - np.exp(2 * p["lam"] * dt)) / (-2 * p["lam"])) / sq     # exact OU noise scaling
        for t in range(n):
            cpl = (A @ z - A.sum(1) * z) * (p["K"] / deg)
            if kap:
                cpl = cpl + drive_nb * s_nb[t]
            if xs:
                cpl = cpl + self_d * z
            if exact:
                z = E * (z + (-np.abs(z) ** 2 * z + cpl) * dt) + g * (sp * dW[t] + p["sigma"] * load * dWc[t])
            else:
                z = z + ((p["lam"] + 1j * w - np.abs(z) ** 2) * z + cpl) * dt + sp * dW[t] + p["sigma"] * load * dWc[t]
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
    if p.get("obs_snr_db") is not None:                 # DEVIATIONS 12: white observation noise (drawn only if set)
        nv = x.var(1, keepdims=True) / 10 ** (p["obs_snr_db"] / 10)
        if p.get("obs_noise_exp"):                      # screen 11: coloured (1/f^chi) observation noise, same variance
            Nf = np.fft.rfft(rng.standard_normal(x.shape), axis=1)
            f = np.fft.rfftfreq(x.shape[1], 1 / FS_OUT); f[0] = f[1]
            nz = np.fft.irfft(Nf * f ** (-p["obs_noise_exp"] / 2), n=x.shape[1], axis=1)
            x = x + np.sqrt(nv) * nz / nz.std(1, keepdims=True)
        else:
            x = x + np.sqrt(nv) * rng.standard_normal(x.shape)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)   # power-matched
    return x.astype(np.float32)


def make_dataset(seed, kind, n_train_per_class=100, n_test_per_class=40, params=None):
    p = dict(DEFAULTS[kind], **(params or {}))
    s11 = [k for k in ("kappa_nb", "extra_self_damp_K", "integrator") if p.get(k)]
    if s11 and kind != "mixed":
        raise ValueError(f"options {s11} are implemented for kind='mixed' only")
    if p.get("kappa_nb"):
        p["_nb_rng"] = np.random.default_rng([seed, 11, 99, int(p.get("stream", 0) or 0)])
    tpl = templates(seed, kind, p)
    key = [seed, 11, {"driven": 0, "emergent": 1, "driven_common": 2, "mixed": 3}[kind]]
    if p.get("stream"):                      # DEVIATIONS 17: independent trial draws, same session templates
        key.append(int(p["stream"]))
    rng = np.random.default_rng(key)

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
    p.pop("_nb_rng", None)
    return Xtr, ytr, Xte, yte, dict(seed=seed, kind=kind, params=p, fs=FS_OUT, M=p.get("M", M))
