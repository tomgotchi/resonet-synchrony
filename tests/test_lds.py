"""LDS denoising recovers lagged coupling and shared input under sensor noise, where raw VAR statistics fail."""
import numpy as np

from resonet_sync import lds


def _toy(seed=0):
    rng = np.random.default_rng(seed); M = 3
    A0, _, _, C = lds._init(M, 250.0, r=0.97)
    A0[2:4, 0:2] = 0.15 * np.eye(2)                                  # 0 -> 1 coupling
    Q0 = 0.05 * np.eye(6); Q0[2:4, 4:6] = Q0[4:6, 2:4] = 0.04 * np.eye(2)   # shared input to 1 & 2
    ch = np.linalg.cholesky(Q0)
    X = np.zeros((150, M, 250))
    for k in range(150):
        x = np.zeros(6)
        for t in range(350):
            x = A0 @ x + ch @ rng.standard_normal(6)
            if t >= 100:
                X[k, :, t - 100] = C @ x + rng.standard_normal(M)   # heavy sensor noise
    return X


def test_denoised_stats_recover_structure():
    fit = lds.em_fit(_toy(), n_iter=200, tol=1e-7)
    d = lds.denoised_stats(fit, p=10)
    G, Cc = np.array(d["GC"]), np.abs(np.array(d["inst"]))
    assert G[0, 1] > 5 * G[1, 0] and G[0, 1] > 20 * G[0, 2]
    assert Cc[1, 2] > 5 * max(Cc[0, 1], Cc[0, 2])
