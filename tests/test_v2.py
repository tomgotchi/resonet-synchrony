"""ResoNet v2 (resonet_sync/v2.py): exact properties of the cross-spectral bank and readout."""
import numpy as np
import pytest
import torch

from resonet_sync.v2 import CrossSpectralBank, V2Readout, build_v2


def _x(B=4, C=6, T=500, seed=0):
    return torch.randn(B, C, T, generator=torch.Generator().manual_seed(seed))


def test_bank_tiles_the_band_in_hz():
    b = CrossSpectralBank(6, n_bands=8, n_spatial=3, fs=250.0, band=(4.0, 40.0))
    f0, bw = b._band_params()
    assert np.allclose(f0.detach().numpy(), np.geomspace(4, 40, 8), rtol=1e-5)
    mid = (f0[:-1] + f0[1:]) / 2                                   # upper -3 dB edge = midpoint to the next centre
    assert torch.allclose(f0[:-1] + bw[:-1], mid, rtol=1e-5)
    assert b.f0_hz().shape == (24,) and b.bw_hz().shape == (24,)


def test_bank_is_linear_causal_and_matches_direct_filtering():
    b = CrossSpectralBank(6, n_bands=4, n_spatial=2, fs=250.0, band=(4.0, 40.0), kernel_tau=40.0)
    x = _x()
    with torch.no_grad():
        z = b(x)
        x2 = x.clone(); x2[:, :, 300:] += 5.0
        assert torch.allclose(b(x2)[:, :300], z[:, :300], atol=1e-5)                  # causal
        assert torch.allclose(b(2.5 * x), 2.5 * z, atol=1e-4)                         # linear
        f0, bw = b._band_params()
        lam = torch.complex(-2 * np.pi * bw / 250.0, 2 * np.pi * f0 / 250.0)
        y = torch.einsum("fsc,bct->btfs", b.W, x).to(torch.complex64)
        ref = torch.zeros_like(y); acc = torch.zeros_like(y[:, 0])
        g = (1 - torch.exp(-2 * np.pi * bw / 250.0))[None, :, None]
        for t in range(y.shape[1]):                                                    # z[t] = e^lam z[t-1] + g y[t]
            acc = torch.exp(lam)[None, :, None] * acc + g * y[:, t]
            ref[:, t] = acc
        assert torch.allclose(z, ref, atol=2e-4)


def test_unit_gain_at_resonance():
    b = CrossSpectralBank(1, n_bands=3, n_spatial=1, fs=250.0, band=(8.0, 32.0))
    with torch.no_grad():
        b.W.fill_(1.0)
        t = torch.arange(2500) / 250.0
        for k, f in enumerate((8.0, 16.0, 32.0)):
            z = b(torch.cos(2 * np.pi * f * t)[None, None])
            assert abs(float(z[0, 1500:, k, 0].abs().mean()) - 0.5) < 0.03            # analytic half of a unit cosine


def test_im_block_is_exactly_blind_to_a_single_source_and_sees_a_lag():
    """One source seen through any real mixing: every projection is the same waveform, so coherency is +-1 and
    its imaginary part vanishes whatever the source's power. A lagged second source gives Im != 0."""
    torch.manual_seed(0)
    T, fs = 1500, 250.0
    s = torch.randn(8, 1, T)
    a = torch.randn(6)
    ro = V2Readout(4, 3, fs, kind="cov+lag", skip_s=0.5)
    b = CrossSpectralBank(6, n_bands=4, n_spatial=3, fs=fs, band=(4.0, 40.0))
    with torch.no_grad():
        for scale in (1.0, 7.0):
            p = ro.parts(b(scale * a[None, :, None] * s))
            assert p["im"].abs().max() < 1e-4 and torch.allclose(p["re"].abs(), torch.ones_like(p["re"]), atol=1e-4)
        lagged = torch.roll(s, 6, dims=-1)                                             # 24 ms lag
        x = a[None, :, None] * s + torch.randn(6)[None, :, None] * lagged
        assert ro.parts(b(x))["im"].abs().max() > 0.05


@pytest.mark.parametrize("kind,dim", [("power", 12), ("cov", 24), ("cov+lag", 36), ("cov+lag_null", 36), ("lag", 12)])
def test_readout_dims_and_null_keeps_power_and_re(kind, dim):
    m = build_v2(kind, 6, 250.0, dict(n_bands=4, n_spatial=3))
    x = _x()
    assert m.head.dim == dim and m(x).shape == (4, 4)
    if kind == "cov+lag_null":
        ref = build_v2("cov+lag", 6, 250.0, dict(n_bands=4, n_spatial=3))
        with torch.no_grad():
            f_null, f_ref = m.head.features(m.bank(x)), ref.head.features(ref.bank(x))
        assert torch.allclose(f_null[:, :24], f_ref[:, :24], atol=1e-6)               # power and Re untouched
        assert not torch.allclose(f_null[:, 24:], f_ref[:, 24:], atol=1e-3)           # Im replaced by the null


def test_gradients_reach_spatial_filters_and_resonators():
    m = build_v2("cov+lag", 6, 250.0, dict(n_bands=4, n_spatial=3))
    m(_x()).sum().backward()
    for n in ("W", "log_f0", "log_bw"):
        g = getattr(m.bank, n).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n


def test_hgd_dev_script_runs_cv_without_touching_test(tmp_path, monkeypatch):
    """scripts/v2_hgd_dev.py: k-fold CV inside the training split; the test arrays are never used."""
    import importlib.util
    import json
    from resonet_sync import data_hgd, run as R
    from resonet_sync.synth_smoke import tiny_dataset
    spec = importlib.util.spec_from_file_location("v2_hgd_dev", R.ROOT / "scripts" / "v2_hgd_dev.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    Xtr, ytr, _, _ = tiny_dataset(n_per_class_tr=10, T=500, seed=0)
    monkeypatch.setattr(data_hgd, "load", lambda c, s, d, cap, seed: (Xtr, ytr, None, None))   # no test split at all
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "P1", dict(R.P1, batch=16))
    mod.main(["--subjects", "1", "--folds", "2", "--epochs", "1", "--bands", "3", "--spatial", "2", "--fmax", "40",
              "--cache", str(tmp_path)])
    s = json.loads((tmp_path / "results" / "v2_hgd_dev" / "b3s2_4-40Hz" / "summary.json").read_text())
    assert set(s["mean_over_subjects"]) == {"v2:power", "v2:cov", "v2:cov+lag", "v2:cov+lag_null", "FBLV_LDA", "FBCSP_LDA"}
    assert all(0 <= v <= 1 for v in s["mean_over_subjects"].values())


def test_null_arm_shares_the_global_rng_stream_with_cov_lag():
    """The null's shifts use a private generator: both arms leave the global RNG in the same state."""
    x = _x()
    states = []
    for kind in ("cov+lag", "cov+lag_null"):
        torch.manual_seed(3)
        m = build_v2(kind, 6, 250.0, dict(n_bands=4, n_spatial=3)); m.train()
        torch.manual_seed(5); m(x)
        states.append(torch.rand(1).item())
    assert states[0] == states[1]
