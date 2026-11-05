"""Smoke tests required by TASK_PITCH1.md S7 for every new arm, plus Hz-parameterization,
causality and capacity checks."""
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from resonet_sync.banks import BANKS, HzOscBank
from resonet_sync.readouts import READOUTS, Readout, RNModel

FS = 250.0
ROOT = Path(__file__).resolve().parent.parent


def tiny(seed=0, B=8, C=6, T=200):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, T, generator=g), torch.arange(B) % 4


# ------------------------------------------------------------------ parameterization
def test_initial_frequencies_logspaced_in_hz():
    b = HzOscBank(n_channels=4, n_osc=8, fs=FS, band=(1.0, 40.0), kind="lin")
    assert np.allclose(b.f0_hz().detach().numpy(), np.geomspace(1, 40, 8), rtol=1e-5)
    mu_s, om_s = b.per_sample()
    assert torch.allclose(om_s, 2 * math.pi * b.f0_hz() / FS)
    assert torch.allclose(mu_s, -2 * math.pi * b.bw_hz() / FS)
    # published damping prior carried over: per-sample decay 0.004..0.01
    assert (-mu_s >= 0.004 - 1e-6).all() and (-mu_s <= 0.01 + 1e-6).all()


def test_nyquist_guard():
    b = HzOscBank(n_channels=4, n_osc=8, fs=FS, band=(2.0, 150.0), kind="lin")
    assert b.f0_hz().max() <= 0.45 * FS + 1e-4


@pytest.mark.parametrize("kind", BANKS)
def test_free_rotation_matches_f0(kind):
    """Drive with a single impulse, then measure per-sample phase advance of each oscillator."""
    b = HzOscBank(n_channels=1, n_osc=6, fs=FS, band=(2.0, 40.0), kind=kind)
    with torch.no_grad():
        b.input_gain.fill_(0.2)       # stay in the near-linear regime for nl banks
        x = torch.zeros(1, 1, 400); x[0, 0, 5] = 1.0
        z = b(x)[0]                    # (T', N)
        adv = torch.angle(z[60:300] * z[59:299].conj()).median(0).values
        f_meas = adv.numpy() * FS / (2 * np.pi)
    assert np.allclose(f_meas, b.f0_hz().detach().numpy(), rtol=0.02, atol=0.05), (f_meas, b.f0_hz())


# ------------------------------------------------------------------ causality (reused idea from phasetrack)
@pytest.mark.parametrize("kind", BANKS)
def test_bank_causal(kind):
    torch.manual_seed(0)
    b = HzOscBank(n_channels=6, n_osc=5, fs=FS, kind=kind)
    x, _ = tiny()
    with torch.no_grad():
        z0 = b(x)
        for n in (0, 50, 150):
            x2 = x.clone(); x2[..., n + 1:] = torch.randn_like(x2[..., n + 1:]) * 5
            z1 = b(x2)
            k = n + 1 if kind == "lin" else n   # nl trajectory index t holds time t+1
            if kind == "lin":  # FFT convolution: causal up to float32 round-off
                assert torch.allclose(z0[:, :k], z1[:, :k], atol=1e-4, rtol=1e-4)
            else:              # RK4 recursion: bit-exact
                assert torch.equal(z0[:, :k], z1[:, :k])
            assert not torch.allclose(z0[:, k + 1:], z1[:, k + 1:])


# ------------------------------------------------------------------ smoke: gradients + loss decrease
@pytest.mark.parametrize("kind", BANKS)
@pytest.mark.parametrize("ro", READOUTS)
def test_gradients_reach_every_parameter_group(kind, ro):
    torch.manual_seed(0)
    m = RNModel(HzOscBank(n_channels=6, n_osc=5, fs=FS, kind=kind), Readout(5, 4, ro))
    x, y = tiny()
    xp = tiny(seed=1)[0] if ro == "trial_shuffle" else None
    loss = torch.nn.functional.cross_entropy(m(x, xp), y)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{kind}:{ro} no grad for {name}"
        assert torch.isfinite(p.grad.abs()).all(), name
        assert p.grad.abs().sum() > 0, f"{kind}:{ro} zero grad for {name}"
    names = {n for n, _ in m.named_parameters()}
    assert ("bank.W" in names) == (kind == "nl_coupled")
    assert ("bank.M_enc" in names) == (kind == "nl_uncoupled_wide")


@pytest.mark.parametrize("kind", BANKS)
@pytest.mark.parametrize("ro", READOUTS)
def test_loss_decreases(kind, ro):
    from resonet_sync.synth_smoke import tiny_dataset
    from resonet_sync.train import Split
    torch.manual_seed(0)
    Xtr, ytr, _, _ = tiny_dataset(n_per_class_tr=8, C=6, T=150)
    tr = Split(Xtr, ytr, 0)
    m = RNModel(HzOscBank(n_channels=6, n_osc=5, fs=FS, kind=kind), Readout(5, 4, ro, dropout=0.0))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    idx = np.arange(len(tr))
    xp = tr.X[tr.sample_partners(idx)] if ro == "trial_shuffle" else None
    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(tr.X, xp), tr.y)
        loss.backward(); opt.step(); losses.append(loss.item())
    assert min(losses[-5:]) < losses[0] - 0.05, losses


# ------------------------------------------------------------------ readout controls
def test_surrogate_and_shuffle_change_features():
    torch.manual_seed(0)
    b = HzOscBank(n_channels=6, n_osc=5, fs=FS, kind="nl_coupled")
    x, _ = tiny()
    with torch.no_grad():
        z, zp = b(x), b(tiny(seed=3)[0])
        sync = Readout(5, 4, "sync").features(z)
        surr = Readout(5, 4, "surrogate").features(z)
        shuf = Readout(5, 4, "trial_shuffle").features(z, zp)
        shuf_self = Readout(5, 4, "trial_shuffle").features(z, z)
    assert sync.shape == surr.shape == shuf.shape          # capacity matched (2P dims)
    assert (sync - surr).abs().mean() > 1e-2
    assert (sync - shuf).abs().mean() > 1e-2
    assert torch.allclose(shuf_self, sync)                 # partner = self reduces to sync
    sp = Readout(5, 4, "sync+power").features(z)
    assert sp.shape[1] == 5 + sync.shape[1] and torch.allclose(sp[:, 5:], sync)


def test_surrogate_preserves_each_oscillator_spectrum():
    """Circular roll keeps every oscillator's own |FFT|; only cross-oscillator alignment changes."""
    torch.manual_seed(0)
    b = HzOscBank(n_channels=6, n_osc=5, fs=FS, kind="lin")
    x, _ = tiny()
    with torch.no_grad():
        u = b(x); u = u / (u.abs() + 1e-6)
        r = Readout(5, 4, "surrogate")
        sh = (r.surr_frac * u.shape[1]).long()
        ur = torch.stack([torch.roll(u[:, :, i], int(sh[i]), dims=1) for i in range(5)], dim=2)
        assert torch.allclose(torch.fft.fft(u, dim=1).abs(), torch.fft.fft(ur, dim=1).abs(), atol=1e-3)


# ------------------------------------------------------------------ capacity control + shared init
def test_wide_matches_coupling_param_count():
    N = 16
    c = HzOscBank(n_channels=128, n_osc=N, kind="nl_coupled").extra_param_count()
    w = HzOscBank(n_channels=128, n_osc=N, kind="nl_uncoupled_wide").extra_param_count()
    assert c == 2 * N * N and abs(w - c) <= 2 * N


def test_same_seed_same_encoder_across_banks():
    encs = [HzOscBank(n_channels=8, n_osc=6, kind=k, seed=7).W_enc for k in BANKS]
    assert all(torch.equal(encs[0], e) for e in encs[1:])


def test_vendor_untouched():
    readme = (ROOT / "vendor" / "README.md").read_text()
    for f in ("resonance_network.py", "hc2_loader.py"):
        h = hashlib.sha256((ROOT / "vendor" / f).read_bytes()).hexdigest()
        assert h in readme, f"vendor/{f} was modified"


def test_power_surrogate_is_exact_concatenation():
    """DEVIATIONS 22: power+surrogate = [log power, surrogate coherence], the same size as sync+power."""
    torch.manual_seed(0)
    z = torch.randn(3, 50, 6, dtype=torch.complex64)
    ps, p, s, sp = (Readout(6, 4, kind=k, seed=5) for k in ("power+surrogate", "power", "surrogate", "sync+power"))
    assert ps.dim == sp.dim
    assert torch.allclose(ps.features(z), torch.cat([p.features(z), s.features(z)], dim=1))
