import numpy as np

from resonet_sync import lds, nuisance as N, mechanism as Mx, synth


def _X(seed=999, **p):
    X, *_ = synth.make_dataset(seed, "mixed", 30, 1, dict(dict(K=25.0, rho=0.5, obs_snr_db=10.0), **p))
    return X


def test_integer_delay_matches_mechanism():
    a = Mx.analytic(_X())
    assert np.allclose(N.te_matrix_frac(a, 8.0), Mx.te_matrix(a))


def test_peak_freq_tracks_shift():
    f0, f1 = N.peak_freq(_X()), N.peak_freq(_X(f_shift_hz=1.0))
    assert 0.7 < f1 - f0 < 1.3


def test_degrade_hits_target_fraction():
    X = _X()
    r = lds.em_fit(X, n_iter=150, tol=1e-6)["R"]
    Y, v = N.degrade_to(X, r, np.minimum(r + 0.3, 0.95), seed=1)
    r2 = lds.em_fit(Y, n_iter=150, tol=1e-6)["R"]
    assert np.all(v > 0)
    assert abs(np.mean(r2) - np.mean(np.minimum(r + 0.3, 0.95))) < 0.08


def test_causal_free_of_labels():
    # the functions take only X; no labels in the signature
    import inspect
    for f in (N.peak_freq, N.s_te_cyc, N.degrade_to):
        assert "y" not in inspect.signature(f).parameters
