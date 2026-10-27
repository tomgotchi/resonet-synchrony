"""With k = 1 sub-step, the Hz-parameterized banks must reproduce the published (vendored) forward
pass exactly once given equivalent parameters: omega = omega_per_sample / h, mu = mu_per_sample / h.
This is the "keep everything else identical" check of TASK_PITCH1.md S2."""
import torch

from resonet_sync._vendor import rn
from resonet_sync.banks import HzOscBank


def _copy_common(ours, pub):
    mu_s, om_s = ours.per_sample()
    with torch.no_grad():
        pub.omega.copy_(om_s / ours.h)
        pub.mu.copy_(mu_s / ours.h)
        pub.W_enc.copy_(ours.W_enc); pub.b_enc.copy_(ours.b_enc); pub.input_gain.copy_(ours.input_gain)


def test_nl_coupled_equals_published_resonance_network():
    torch.manual_seed(0)
    ours = HzOscBank(n_channels=6, n_osc=5, fs=250.0, band=(2.0, 30.0), kind="nl_coupled", seed=3)
    pub = rn().ResonanceNetwork(n_channels=6, n_osc=5, readout="power")
    _copy_common(ours, pub)
    with torch.no_grad():
        pub.W.copy_(ours.W); pub.beta.copy_(ours.beta); pub.gamma.copy_(ours.gamma)
        x = torch.randn(3, 6, 120)
        _, z_pub = pub(x, return_traj=True)
        z = ours(x)
    assert torch.allclose(z, z_pub, atol=1e-6, rtol=1e-5)


def test_lin_equals_published_fast_resonance_network():
    torch.manual_seed(0)
    ours = HzOscBank(n_channels=6, n_osc=5, fs=250.0, band=(2.0, 30.0), kind="lin", seed=3)
    pub = rn().FastResonanceNetwork(n_channels=6, n_osc=5, readout="power")
    _copy_common(ours, pub)
    with torch.no_grad():
        x = torch.randn(3, 6, 120)
        _, z_pub = pub(x, return_traj=True)
        z = ours(x)
    assert torch.allclose(z, z_pub, atol=1e-5, rtol=1e-4)


def test_published_sync_and_surrogate_features_identical():
    """Our sync/surrogate readout features equal the published in-forward computation."""
    from resonet_sync.readouts import Readout
    torch.manual_seed(0)
    ours = HzOscBank(n_channels=6, n_osc=5, fs=250.0, kind="lin", seed=3)
    for ro, pub_ro in (("sync", "sync"), ("surrogate", "sync_surrogate")):
        pub = rn().FastResonanceNetwork(n_channels=6, n_osc=5, readout=pub_ro, seed=42)
        _copy_common(ours, pub)
        head = Readout(5, 4, ro, seed=42)
        assert torch.equal(head.surr_frac, pub.surr_frac)
        with torch.no_grad():
            pub.readout.weight.copy_(head.readout.weight); pub.readout.bias.copy_(head.readout.bias)
            pub.eval(); head.eval()
            x = torch.randn(3, 6, 120)
            assert torch.allclose(pub(x), head(ours(x)), atol=1e-4)
