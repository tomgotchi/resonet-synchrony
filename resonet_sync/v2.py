"""ResoNet v2 ("cross-spectral ResoNet"): a design study, separate from the frozen Phase 1 arms.

Why (critique of 2026-11-09, docs/CRITIQUE_AND_V2.md):
  * In the published model every oscillator sits at 0.2-1.6 Hz while the data are 4-125 Hz, so nothing resonates;
    the Hz-parameterised bank fixed the frequencies but kept 0.16-0.40 Hz bandwidths, i.e. 16 narrow lines that
    cover a few percent of the band.
  * Each oscillator has its own frequency AND its own (complex, tanh) spatial filter, so a "phase coherence"
    between two oscillators mixes a frequency difference with a spatial difference. It is a second-order
    (cross-spectral) feature of the sensors, not a synchrony between rhythms, and it responds to source power.

v2 keeps what works (learned spatial filters + resonators + second-order readout) and makes it interpretable:
  * F resonators tile the band (log-spaced f0 in Hz, constant-Q bandwidths so neighbours meet at -3 dB);
  * each band has S REAL spatial filters (no tanh), so z[f, s](t) = resonator_f(w[f, s] . x)(t) is the analytic
    band-f signal of a real spatial projection;
  * the readout is the within-band cross-spectral matrix of those S projections, split into
        power  log <|z_s|^2>                         (F*S)        band power of each projection
        re     Re coherency(s, s'), s < s'           (F*S(S-1)/2) zero-lag covariance (what CSP uses)
        im     Im coherency(s, s'), s < s'           (F*S(S-1)/2) LAGGED coupling
    For real weights, Re C = w^T Re(S_x) w' and Im C = w^T Im(S_x) w'. Instantaneous mixing of independent
    sources has Im(S_x) = 0 whatever the source powers (Nolte et al. 2004), so `im` cannot respond to source
    power or volume conduction; it responds only to time-lagged interactions.
  * `cov+lag_null` replaces `im` by the same quantity after a fresh circular shift per trial and projection: the
    dimension-matched control for "lagged coupling adds beyond band covariance".
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

V2_READOUTS = ("power", "cov", "cov+lag", "cov+lag_null", "lag")
V2_DEFAULT = dict(n_bands=8, n_spatial=4, band=(4.0, 40.0), skip_s=0.5)


class CrossSpectralBank(nn.Module):
    """x (B, C, T) real -> z (B, T, F, S) complex: S real spatial projections per band, each through that band's
    resonator (exact one-pole complex kernel, causal, unit gain at resonance)."""
    kind = "v2"

    def __init__(self, n_channels, n_bands=8, n_spatial=4, fs=250.0, band=(4.0, 40.0), seed=42, kernel_tau=8.0):
        super().__init__()
        self.C, self.F, self.S, self.fs = n_channels, n_bands, n_spatial, float(fs)
        self.f_max, self.kernel_tau = 0.45 * self.fs, kernel_tau
        g = torch.Generator().manual_seed(seed)
        f0 = np.geomspace(band[0], min(band[1], self.f_max), n_bands)
        ratio = (f0[-1] / f0[0]) ** (1.0 / max(n_bands - 1, 1))
        bw0 = f0 * (ratio - 1.0) / 2.0                           # half-width: neighbours meet at their -3 dB points
        self.log_f0 = nn.Parameter(torch.tensor(np.log(f0), dtype=torch.float32))
        self.log_bw = nn.Parameter(torch.tensor(np.log(bw0), dtype=torch.float32))
        self.W = nn.Parameter(torch.randn(n_bands, n_spatial, n_channels, generator=g) / math.sqrt(n_channels))

    def f0_hz(self):
        return torch.exp(self.log_f0).clamp(0.5, self.f_max).repeat_interleave(self.S)      # per oscillator

    def bw_hz(self):
        return torch.exp(self.log_bw).clamp(min=0.05).repeat_interleave(self.S)

    def _band_params(self):
        return torch.exp(self.log_f0).clamp(0.5, self.f_max), torch.exp(self.log_bw).clamp(min=0.05)

    def forward(self, x):
        B, C, T = x.shape
        y = torch.einsum("fsc,bct->btfs", self.W, x)                                        # real projections
        f0, bw = self._band_params()
        decay = 2 * math.pi * bw / self.fs
        lam = torch.complex(-decay, 2 * math.pi * f0 / self.fs)                             # per sample
        kn = int(min(T, math.ceil(self.kernel_tau / float(decay.detach().min()))))                 # kernel down to e^-tau
        n = torch.arange(kn, device=x.device, dtype=torch.float32)
        k = torch.exp(lam[None, :] * n[:, None]) * (1 - torch.exp(-decay))[None, :]         # (kn, F)
        L = 1 << (T + kn - 1).bit_length()
        Z = torch.fft.ifft(torch.fft.fft(y.to(torch.complex64), n=L, dim=1) *
                           torch.fft.fft(k, n=L, dim=0)[None, :, :, None], dim=1)
        return Z[:, :T]                                                                     # (B, T, F, S)


class V2Readout(nn.Module):
    """Within-band cross-spectral readout. Head: BatchNorm -> Dropout -> Linear (as published)."""
    needs_partner = False

    def __init__(self, n_bands, n_spatial, fs=250.0, n_classes=4, kind="cov+lag", skip_s=0.5, dropout=0.25, seed=42):
        super().__init__()
        assert kind in V2_READOUTS, kind
        self.kind, self.F, self.S, self.skip = kind, n_bands, n_spatial, int(round(skip_s * fs))
        iu, ju = torch.triu_indices(n_spatial, n_spatial, offset=1)
        self.register_buffer("iu", iu); self.register_buffer("ju", ju)
        P = n_bands * iu.numel()
        self.dim = {"power": n_bands * n_spatial, "cov": n_bands * n_spatial + P,
                    "cov+lag": n_bands * n_spatial + 2 * P, "cov+lag_null": n_bands * n_spatial + 2 * P, "lag": P}[kind]
        # the null's shifts come from their own generator, so `cov+lag` and `cov+lag_null` consume the global RNG
        # identically (same augmentation and dropout streams): the pair differs only in the lag block
        self._shift_gen = torch.Generator().manual_seed(seed + 1)
        self.feat_norm = nn.BatchNorm1d(self.dim)
        self.feat_drop = nn.Dropout(dropout)
        self.readout = nn.Linear(self.dim, n_classes)

    def _coherency(self, z):
        Cm = torch.einsum("btfs,btfr->bfsr", z, z.conj()) / z.shape[1]                      # (B, F, S, S) Hermitian
        p = torch.diagonal(Cm, dim1=-2, dim2=-1).real.clamp_min(1e-12)                      # (B, F, S)
        coh = Cm[:, :, self.iu, self.ju] / torch.sqrt(p[:, :, self.iu] * p[:, :, self.ju])  # (B, F, P)
        return p, coh

    def parts(self, z):
        """dict(power, re, im, im_null) of flattened feature blocks (im_null only when the kind needs it)."""
        z = z[:, self.skip:] if z.shape[1] > 2 * self.skip else z
        B = z.shape[0]
        p, coh = self._coherency(z)
        out = dict(power=torch.log(p).reshape(B, -1), re=coh.real.reshape(B, -1), im=coh.imag.reshape(B, -1))
        if self.kind == "cov+lag_null":
            T = z.shape[1]
            sh = torch.randint(0, T, (B, 1, self.F, self.S), generator=self._shift_gen).to(z.device)
            idx = (torch.arange(T, device=z.device)[None, :, None, None] - sh) % T
            out["im_null"] = self._coherency(torch.gather(z, 1, idx))[1].imag.reshape(B, -1)
        return out

    def features(self, z, z_partner=None):
        f = self.parts(z)
        keys = {"power": ("power",), "cov": ("power", "re"), "cov+lag": ("power", "re", "im"),
                "cov+lag_null": ("power", "re", "im_null"), "lag": ("im",)}[self.kind]
        return torch.cat([f[k] for k in keys], dim=1)

    def forward(self, z, z_partner=None):
        return self.readout(self.feat_drop(self.feat_norm(self.features(z))))


def build_v2(readout, n_channels, fs, cfg=None, seed=42, n_classes=4):
    from .readouts import RNModel
    c = dict(V2_DEFAULT, **(cfg or {}))
    bank = CrossSpectralBank(n_channels, c["n_bands"], c["n_spatial"], fs, tuple(c["band"]), seed=seed)
    torch.manual_seed(seed)                                   # head init independent of the bank's generator
    return RNModel(bank, V2Readout(c["n_bands"], c["n_spatial"], fs, n_classes, readout, c["skip_s"], seed=seed))
