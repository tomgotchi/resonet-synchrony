"""Each mechanism statistic behaves as intended on toy systems where the answer is known."""
import numpy as np

from resonet_sync import mechanism as Mx

FS = 250.0


def _kuramoto(K, n_tr=60, T=300, directed=False, seed=0):
    """Two noisy phase oscillators at ~8 Hz; K couples 0 -> 1 (and 1 -> 0 unless directed). Returns cos(phase)."""
    rng = np.random.default_rng(seed)
    dt = 1 / FS
    X = []
    for _ in range(n_tr):
        ph = rng.uniform(0, 2 * np.pi, 2)
        w = 2 * np.pi * np.array([7.6, 8.4])
        out = np.empty((2, T))
        for t in range(T):
            d = ph[0] - ph[1]
            dph = w.copy()
            dph[1] += K * np.sin(d)
            if not directed:
                dph[0] += K * np.sin(-d)
            ph = ph + dph * dt + 1.5 * np.sqrt(dt) * rng.standard_normal(2)
            out[:, t] = np.cos(ph)
        X.append(out + 0.1 * rng.standard_normal((2, T)))
    return np.asarray(X)


def test_coupling_strength_tracks_K():
    s0 = Mx.s_couple(Mx.analytic(_kuramoto(0.0)))
    s1 = Mx.s_couple(Mx.analytic(_kuramoto(8.0)))
    # monotone in K; the absolute value is attenuated (6-10 Hz filtering smooths dphi/dt: regression dilution),
    # which is common to all generators in the screen, so only relative comparisons are used
    s2 = Mx.s_couple(Mx.analytic(_kuramoto(16.0)))
    assert s2 > s1 > 2.5 * s0, (s0, s1, s2)


def test_te_detects_direction():
    TE = Mx.te_matrix(Mx.analytic(_kuramoto(8.0, directed=True)))
    assert TE[0, 1] > 2 * TE[1, 0], TE                     # 0 -> 1 dominates
    assert Mx.s_asym(TE) > 0.3


def test_residual_coherence_removes_a_common_mode():
    """1-s 6-10 Hz windows have a PLV bias floor (~0.45 for independent noise). Removing the common mode must
    bring a rank-1 (common-input) system down to that floor, but leave a second, weaker independent cluster
    above it."""
    rng = np.random.default_rng(0)
    t = np.arange(300) / FS
    X_common, X_two, X_noise = [], [], []
    for _ in range(60):
        c = np.cos(2 * np.pi * 8 * t + rng.uniform(0, 6.3))
        c2 = 0.6 * np.cos(2 * np.pi * 8.5 * t + rng.uniform(0, 6.3))
        X_common.append(np.stack([np.roll(c, k * 3) for k in range(6)]) + 0.3 * rng.standard_normal((6, 300)))
        X_two.append(np.stack([c] * 3 + [c2] * 3) + 0.3 * rng.standard_normal((6, 300)))
        X_noise.append(rng.standard_normal((6, 300)))
    A = lambda X: Mx.analytic(np.asarray(X))
    floor = Mx.s_plv(A(X_noise))
    p1, r1, r2 = Mx.s_plv(A(X_common)), Mx.s_resid(A(X_common)), Mx.s_resid(A(X_two))
    assert p1 > 0.8                       # raw coherence is high for the common-input system
    assert abs(r1 - floor) < 0.06         # ...and falls to the noise floor after removing the common mode
    assert r2 > floor + 0.05, (floor, r1, r2)   # a second independent cluster survives
