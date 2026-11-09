"""Numerical check of the non-identifiability results in docs/NONIDENTIFIABILITY.md (review board 2026-11-09).

Each check compares the full cross-spectral matrix S(omega) of a COUPLED model with that of an UNCOUPLED model
driven by latent common rhythms. Equal S(omega) for Gaussian stationary processes means equal joint laws, so no
statistic of passive observations can tell them apart.

  python scripts/nonident_check.py
"""
import numpy as np


def spectrum(A, B, om):
    """Complex OU dz = A z dt + B dW: S(omega) = G B B^H G^H, G = (i omega - A)^-1."""
    out = []
    for o in om:
        G = np.linalg.inv(1j * o * np.eye(len(A)) - A)
        out.append(G @ B @ B.conj().T @ G.conj().T)
    return np.array(out)


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(a).max())


om = 2 * np.pi * np.linspace(-30, 30, 3001)
lam, w0, sig = -2 * np.pi * 1.0, 2 * np.pi * 8.0, 6.0
lp = lam + 1j * w0

# 1. Symmetric pair (Volkov): diffusive coupling k  ==  units with pole lp - 2k + one common rhythm (pole lp)
k = 20.0
Ac = np.array([[lp - k, k], [k, lp - k]])
c = np.sqrt(2 * sig ** 2 * k * (k + abs(lam)))
Au = np.array([[lp - 2 * k, 0, c], [0, lp - 2 * k, c], [0, 0, lp]])
Bu = np.diag([sig, sig, 1.0]).astype(complex)
print("1. symmetric pair            max rel |S_coupled - S_common|:",
      f"{rel(spectrum(Ac, sig * np.eye(2), om), spectrum(Au, Bu, om)[:, :2, :2]):.1e}")

# 2. N identical units on any symmetric weighted graph (Laplacian L, eigenpairs mu_m, v_m)
rng = np.random.default_rng(0)
N, kap = 5, 4.0
Wg = np.triu(rng.uniform(0, 1, (N, N)), 1); Wg = Wg + Wg.T
L = np.diag(Wg.sum(1)) - Wg
mu, V = np.linalg.eigh(L)
Ac = lp * np.eye(N) - kap * L
mmax = mu.max()
gains = np.sqrt(sig ** 2 * kap * (mmax - mu) * (kap * (mmax + mu) + 2 * abs(lam)))
keep = gains > 1e-9                                    # the top mode needs no common rhythm: N - 1 rhythms
M = int(keep.sum())
Au = np.zeros((N + M, N + M), complex)
Au[:N, :N] = (lp - kap * mmax) * np.eye(N)
Au[:N, N:] = V[:, keep] * gains[keep]
Au[N:, N:] = np.diag(lp - kap * mu[keep])
Bu = np.diag(np.r_[np.full(N, sig), np.ones(M)]).astype(complex)
print(f"2. N={N} units, random graph   max rel |S_coupled - S_common|:",
      f"{rel(spectrum(Ac, sig * np.eye(N).astype(complex), om), spectrum(Au, Bu, om)[:, :N, :N]):.1e}",
      f"({M} common rhythms)")

# 3. Directed pair (Zhao): AR(2) unit 1 -> unit 2 with gain b, observation noise; vs a latent driver L feeding
#    unit 1 (with a private, non-propagating share alpha) and unit 2 (gain b / (1 - alpha), compensating private
#    spectrum). Discrete time, 250 Hz.
fs, f1, bw1 = 250.0, 8.0, 2.0
r = np.exp(-np.pi * bw1 / fs); th = 2 * np.pi * f1 / fs
a1 = np.array([1, -2 * r * np.cos(th), r * r])
r2, th2 = np.exp(-np.pi * 3.0 / fs), 2 * np.pi * 8.5 / fs
a2 = np.array([1, -2 * r2 * np.cos(th2), r2 * r2])
b, s2, nu1, nu2 = 0.007, 1.0, 20.0, 20.0
w = np.linspace(0, np.pi, 4096)
A1 = np.polyval(a1[::-1], np.exp(-1j * w)); A2 = np.polyval(a2[::-1], np.exp(-1j * w))
K = b * np.exp(-1j * w) / A2
SA = np.array([[1 / abs(A1) ** 2 + nu1, (1 / abs(A1) ** 2) * np.conj(K)],
               [(1 / abs(A1) ** 2) * K, abs(K) ** 2 / abs(A1) ** 2 + s2 / abs(A2) ** 2 + nu2]])
base, excess = s2 / abs(A2) ** 2 + nu2, b ** 2 / (abs(A1) ** 2 * abs(A2) ** 2)
al_max = 1 / (1 + (excess / base).max()); al = 0.9 * al_max
Kb = K / (1 - al); Sq = base - excess * al / (1 - al)
SB = np.array([[1 / abs(A1) ** 2 + nu1, ((1 - al) / abs(A1) ** 2) * np.conj(Kb)],
               [((1 - al) / abs(A1) ** 2) * Kb, abs(Kb) ** 2 * (1 - al) / abs(A1) ** 2 + Sq]])
print("3. directed pair, latent L    max rel |S_coupled - S_common|:",
      f"{rel(SA, SB):.1e}  (alpha = {al:.3f} of unit 1's rhythm private; min private spectrum {Sq.min():.2f} > 0)")
