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
    s = json.loads((tmp_path / "results" / "v2_hgd_dev" / "b3s2_4-40Hz_ep1" / "summary.json").read_text())
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


def test_csp_init_finds_the_class_specific_source():
    """csp_init_: with one 10 Hz source whose power depends on the class, the first filter of the 10 Hz band
    isolates it far better than a random projection; shapes and unit projected variance hold."""
    from scipy import signal
    from resonet_sync.v2 import CrossSpectralBank, csp_filters, csp_init_
    rng = np.random.default_rng(0)
    n, C, T, fs = 80, 10, 750, 250.0
    y = np.repeat(np.arange(4), n // 4)
    a = rng.standard_normal(C)
    src = signal.sosfilt(signal.butter(2, (9, 11), "band", fs=fs, output="sos"), rng.standard_normal((n, T)))
    X = (1 + 2.0 * (y == 0))[:, None, None] * a[None, :, None] * src[:, None, :] * 3 + rng.standard_normal((n, C, T))
    b = CrossSpectralBank(C, n_bands=4, n_spatial=4, fs=fs, band=(4.0, 40.0))
    w_rand = b.W.detach().clone()
    csp_init_(b, X, y)
    assert b.W.shape == (4, 4, C) and not torch.allclose(b.W, w_rand)
    f0 = b._band_params()[0].detach().numpy()
    k = int(np.argmin(np.abs(f0 - 10)))
    def ratio(w):                                   # class-0 vs rest power of the projection, 8-12 Hz
        xb = signal.sosfilt(signal.butter(4, (8, 12), "band", fs=fs, output="sos"), np.einsum("c,nct->nt", w, X), axis=-1)
        p = (xb ** 2).mean(-1)
        return p[y == 0].mean() / p[y != 0].mean()
    assert max(ratio(b.W[k, j].detach().numpy()) for j in range(4)) > 8.0          # ideal 9 (3x amplitude)
    Wk = b.W[k].detach().numpy().astype(np.float64)
    tot = np.einsum("nct,ndt->cd", X, X) / (n * T)
    G = Wk @ tot @ Wk.T
    corr = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
    assert np.abs(corr - np.eye(4)).max() < 0.97                                    # no duplicated filters
    W = csp_filters(X, y, [(8.0, 12.0)], fs, 4)
    assert W.shape == (1, 4, C)


def test_unnormalised_lag_block_ignores_unrelated_power_but_coherency_does_not():
    """Review board 2026-11-13 (Echeverria): with a class-INDEPENDENT lagged pair present, the power of an unrelated
    source changes the imaginary COHERENCY (through its normalisation) but not the unnormalised imaginary
    cross-spectrum."""
    from scipy import signal, stats
    from resonet_sync.v2_probe import blocks
    rng = np.random.default_rng(0)
    n, C, T, fs = 80, 6, 1500, 250.0
    bp = lambda x: signal.sosfilt(signal.butter(2, (8.5, 11.5), "band", fs=fs, output="sos"), x, axis=-1)
    a = bp(rng.standard_normal((n, T + 500)))[:, 500:]
    b = 0.9 * -np.imag(signal.hilbert(a, axis=-1)) + np.sqrt(1 - 0.81) * bp(rng.standard_normal((n, T + 500)))[:, 500:]
    u = bp(rng.standard_normal((n, T + 500)))[:, 500:]
    y = np.repeat([0, 1], n // 2)
    A = rng.standard_normal((C, 3))
    S = np.stack([a, b, u * (1 + 2.0 * y)[:, None]], 1)              # only the UNRELATED source depends on y
    X = (np.einsum("ck,nkt->nct", A, S) * 8).astype(np.float32)
    bank = CrossSpectralBank(C, n_bands=2, n_spatial=3, fs=fs, band=(9.0, 11.0)).eval()
    head = V2Readout(2, 3, fs, kind="cov+lag", skip_s=0.5)
    f = blocks(bank, head, X, y, seed=1)
    t = lambda v: np.abs(stats.ttest_ind(v[y == 0], v[y == 1], axis=0).statistic)
    assert t(f["im"]).max() > 6.0                                    # coherency: strongly "class"-dependent
    assert t(f["im_raw"]).max() < 3.0                                # unnormalised: not
    t0 = lambda v: np.abs(stats.ttest_1samp(v, 0.0, axis=0).statistic)
    assert t0(f["im_raw"]).max() > 10.0 and t0(f["im_raw_null"]).max() < 4.0    # the lag structure itself is there


def test_higher_order_resonator_is_a_cascade_with_the_same_3db_width_and_steeper_skirts():
    """order = n: n equal poles in series, each widened so that the cascade keeps the -3 dB half-width bw_hz()."""
    fs, n = 250.0, 4
    b1 = CrossSpectralBank(1, n_bands=5, n_spatial=1, fs=fs, band=(20.0, 40.0), kernel_tau=40.0)
    b4 = CrossSpectralBank(1, n_bands=5, n_spatial=1, fs=fs, band=(20.0, 40.0), kernel_tau=40.0, order=n)
    f0, bw = [v.detach().numpy() for v in b4._band_params()]
    with torch.no_grad():
        b1.W.fill_(1.0); b4.W.fill_(1.0)
        x = _x(B=2, C=1, T=600, seed=3)
        z = b4(x)
        x2 = x.clone(); x2[:, :, 400:] += 3.0
        assert torch.allclose(b4(x2)[:, :400], z[:, :400], atol=1e-5)                 # still causal
        dec = 2 * np.pi * bw / fs / np.sqrt(2 ** (1 / n) - 1)                         # the widened pole
        lam = torch.tensor(np.exp(-dec + 2j * np.pi * f0 / fs), dtype=torch.complex64)
        g = torch.tensor(1 - np.exp(-dec), dtype=torch.complex64)
        ref = x[:, 0, :, None].to(torch.complex64).repeat(1, 1, 5)                    # (B, T, F)
        for _ in range(n):                                                            # n one-pole recursions in series
            out = torch.zeros_like(ref); acc = torch.zeros_like(ref[:, 0])
            for t in range(ref.shape[1]):
                acc = lam[None] * acc + g[None] * ref[:, t]
                out[:, t] = acc
            ref = out
        assert torch.allclose(z[..., 0], ref, atol=5e-4)

        def gain(bank, f):                                                            # steady-state amplitude at f Hz
            t = torch.arange(5000) / fs
            return bank(torch.cos(2 * np.pi * f * t)[None, None])[0, -1].abs().numpy().ravel() * 2

        for k in (0, 4):
            g0, gup, gdn = gain(b4, f0[k])[k], gain(b4, f0[k] + bw[k])[k], gain(b4, f0[k] - bw[k])[k]
            assert abs(g0 - 1) < 0.02                                                 # unit gain at resonance
            assert abs(gup ** 2 - 0.5) < 0.03 and abs(gdn ** 2 - 0.5) < 0.03          # -3 dB at f0 +/- bw
        far1, far4 = gain(b1, 5.0)[0], gain(b4, 5.0)[0]                               # two octaves below the 20 Hz band
        assert far4 < 0.2 * far1                                                      # much steeper skirt


def test_central44_is_a_valid_sensor_subset():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "v2_probe_run.py"
    spec = importlib.util.spec_from_file_location("v2_probe_run", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    idx = m.CENTRAL44
    assert len(idx) == 44 and len(set(idx)) == 44 and min(idx) >= 0 and max(idx) < 128
    assert 15 not in idx                                    # Cz (index 15 in the HGD order) is not in the selection
