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
