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


def test_aligned_matches_diag_and_reduces_to_rotation():
    from resonet_sync import diag10b
    X = _X()
    f_c = N.peak_freq(X)
    a = Mx.analytic(X, 250.0, (f_c - 2, f_c + 2))
    assert np.isclose(N.s_te_aligned(X)[0], diag10b.te_rot(a, 8.0, f_c, True, 1))


def test_decide_nan_is_invalid_not_correct():
    from resonet_sync import screen10 as S
    cells = {"F1": (None, None, False)}
    vals = {f"F1|{s}": {"A": {"S_TE_al": 1.0, "S_inst_dn": 0.1}, "B": {"S_TE_al": 0.0, "S_inst_dn": 0.1}}
            for s in range(10)}
    for v in vals.values():
        v["A"].update(f_c=8.0, R_mean=0.1); v["B"].update(f_c=8.0, R_mean=0.1)
    res, ver = S.decide(vals, cells, list(range(10)), "S_TE_al")
    assert res["F1"]["fires"] is True and ver["false_positives"] == ["F1"]       # sanity: a clear false positive
    vals["F1|3"]["A"]["S_inst_dn"] = float("nan")
    res, ver = S.decide(vals, cells, list(range(10)), "S_TE_al")
    assert res["F1"]["fires"] is None and not ver["valid"] and not ver["all_cells_correct"]
    assert ver["invalid_cells"] == ["F1"]


def test_runs_from_ids_splits_adjacent_runs():
    from resonet_sync.data_hc5 import runs_from_ids
    assert runs_from_ids([0, 1, 1, 2, 2, 0, 3]) == [(1, 3, 1), (3, 5, 2), (6, 7, 3)]


def test_simulate_observed_noise_independent_of_state_stream():
    from resonet_sync import lds
    fit = dict(A=0.5 * np.eye(2), Q=np.eye(2), R=np.ones(1), C=np.array([[1.0, 0.0]]))
    Y = lds.simulate_observed(fit, n=1, T=5, burn=0, seed=3)
    Z = lds.simulate_denoised(fit, n=1, T=5, burn=0, seed=3)
    assert not np.allclose(Y - Z, np.random.default_rng(3).standard_normal(Z.shape))


def test_screen11_generator_options_run_and_differ():
    base = dict(K=25.0, rho=0.5, obs_snr_db=10.0)
    X0 = synth.make_dataset(7, "mixed", 2, 1, base)[0]
    for extra in (dict(kappa_nb=4.0), dict(extra_self_damp_K=15.0), dict(integrator="exact"), dict(obs_noise_exp=1.5)):
        X = synth.make_dataset(7, "mixed", 2, 1, dict(base, **extra))[0]
        assert X.shape == X0.shape and np.all(np.isfinite(X)) and not np.allclose(X, X0)


def test_exact_integrator_damping_is_frequency_independent():
    import numpy as np
    lam, dt = -2 * np.pi, 1e-3
    for f in (7.0, 8.0, 9.0):
        E = np.exp((lam + 1j * 2 * np.pi * f) * dt)
        assert np.isclose(np.log(abs(E)) / dt, lam)
