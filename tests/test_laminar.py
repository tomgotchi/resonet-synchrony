import numpy as np

from resonet_sync import laminar as L

FS, SP = 1250.0, 50e-6


def _mix(**kw):
    phi, z = L.simulate_laminar(T_s=30.0, fs=FS, spacing=SP, n_sites=16, **kw)
    i_rad, i_lm = L.sink_sites(z)
    return L.generator_mix(phi, FS, SP, i_rad, i_lm)


def test_csd_recovers_sink_depths():
    phi, z = L.simulate_laminar(T_s=20.0, fs=FS, spacing=SP, w_ec=0.0, ripple_amp=0.0, noise_rel=0.0, seed=1)
    c = L.band(L.csd(phi, SP), FS, 6, 10)
    i_rad, _ = L.sink_sites(z)
    prof = np.abs(c).mean(-1)
    i_pyr = int(np.argmin(np.abs(z - L.DEPTHS["pyramidal"])))
    top2 = set(np.argsort(prof[1:-1])[-2:] + 1)                       # interior sites only
    assert top2 == {i_rad, i_pyr}                                     # the dipole's sink and return source
    assert np.corrcoef(c[i_rad], c[i_pyr])[0, 1] < -0.99               # opposite sign: sink vs source


def test_layers_and_coverage_gate():
    phi, z = L.simulate_laminar(T_s=30.0, fs=FS, spacing=SP, n_sites=16, seed=2)
    lay = L.localise_layers(phi, FS)
    assert abs(z[lay["pyramidal"]] - L.DEPTHS["pyramidal"]) <= 1.5 * SP
    assert L.coverage_ok(lay, SP)
    short, _ = L.simulate_laminar(T_s=30.0, fs=FS, spacing=20e-6, n_sites=8, z0=100e-6, seed=2)
    assert not L.coverage_ok(L.localise_layers(short, FS), 20e-6)  # 8 x 20 um shank cannot separate the sinks


def test_generator_mix_tracks_ec_weight_not_nuisances():
    base = _mix(seed=3)["log_power_ratio_lm_over_rad"]
    more_ec = _mix(seed=3, w_ec=1.5)["log_power_ratio_lm_over_rad"]
    assert more_ec - base > 0.4                                       # the target quantity moves
    for nuis in (dict(f0=7.0), dict(asym=0.4), dict(drift_hz=0.5), dict(noise_exp=1.6)):
        d = _mix(seed=3, **nuis)["log_power_ratio_lm_over_rad"] - base
        assert abs(d) < 0.15, (nuis, d)                               # nuisances do not


def test_v2_ripple_peaks_at_pyramidal_layer_and_fit_recovers_generators():
    """Gate 1 v2 (board 2026-11-09): the ripple LFP peaks AT the pyramidal layer, and the template fit recovers
    both theta sinks within one site; doubling EC drive raises G by about log 4."""
    import numpy as np
    from resonet_sync import laminar as L
    pa, z, tr = L.simulate_laminar_v2(T_s=30.0, seed=3, return_truth=True, gains=np.ones(16))
    lay = L.localise_layers(pa, 1250.0)
    assert lay["pyramidal"] == int(np.argmin(np.abs(z - L.PYR_DEPTH)))
    P, dep = L.fit_generators_refined(L.theta_csd_cov(pa, 1250.0, 50e-6), z - z[0], 50e-6)
    assert np.allclose(dep[:, 0], [tr["ca3"] - z[0], tr["ec"] - z[0]], atol=50e-6)
    pb, _ = L.simulate_laminar_v2(T_s=30.0, seed=4, w_ec=2.0, gains=np.ones(16))
    g = [np.log(L.generator_powers(L.theta_csd_cov(p, 1250.0, 50e-6), P, (0, 1))[[1, 0]]).dot([1, -1]) for p in (pa, pb)]
    assert abs((g[1] - g[0]) - np.log(4)) < 0.4


def test_v2_amplitude_change_is_not_renormalised_away():
    import numpy as np
    from resonet_sync import laminar as L
    a, _ = L.simulate_laminar_v2(T_s=5.0, seed=1, ripple_amp=0, noise_rel=0, gains=np.ones(16))
    b, _ = L.simulate_laminar_v2(T_s=5.0, seed=1, ripple_amp=0, noise_rel=0, gains=np.ones(16), w_ca3=1.3, w_ec=1.3)
    assert np.allclose(b, 1.3 * a)
