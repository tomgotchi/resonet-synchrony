"""Hz-parameterized oscillator banks: lin, nl_uncoupled, nl_coupled, nl_uncoupled_wide.

Identical to the published ResoNet code (vendor/resonance_network.py) except for the oscillator
frequency/damping parameterization (TASK_PITCH1.md S2):

    omega_per_sample = 2*pi*f0/fs,     mu_per_sample = -2*pi*bw/fs

with f0 (Hz) initialised log-spaced across the analysis band and trained as log f0; bw (Hz) trained
as log bw. Published code uses omega ~ U(0.5, 4), mu ~ U(-1, -0.4) with step h = 0.01, i.e.
per-sample rotation 0.005-0.04 rad. Here the continuous-time rate fed to the published RK4 step is
lam = (mu_per_sample + i*omega_per_sample) / h, so lam*h is the per-sample value above and the RK4
step, saturation, coupling, input injection (h * I) and |z| safety clamp are exactly as published.

Bandwidth initialisation (a choice, logged in DEVIATIONS.md): bw is drawn from the published mu
prior converted to Hz at the model's fs, bw = -mu*h*fs/(2*pi), mu ~ U(-1, -0.4). Only the
frequency axis changes; per-sample damping at init is identical to the published bank.

Encoder (complex linear C->N, split-tanh, learnable input gain) is the published one. All four
banks draw their random tensors in the same order from one seeded generator, so for a given seed
they share the exact same encoder initialisation and differ only in dynamics.

Discretization caveat (found in Phase 0): the published integrator takes ONE RK4 step per sample.
With Hz-correct frequencies the per-step rotation reaches ~1 rad at 40 Hz / 250 Hz, where RK4 adds
artificial damping (|R(i w)| < 1): ~2.3x the target bandwidth at 40 Hz, and oscillators above
~60 Hz decay to zero. `rk4_substeps=k` takes k RK4 sub-steps per sample (input linearly
interpolated); k = 1 is the published scheme and the default.

Frequencies are clamped to [0.5 Hz, 0.45*fs] (Nyquist guard; TASK asks for 2-150 Hz in
hippocampus, but at fs = 250 Hz anything above 125 Hz aliases -> DEVIATIONS.md).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ._vendor import rn

BANKS = ("lin", "nl_uncoupled", "nl_coupled", "nl_uncoupled_wide")


def logspaced_f0(n_osc, band):
    return np.geomspace(band[0], band[1], n_osc)


class HzOscBank(nn.Module):
    def __init__(self, n_channels=128, n_osc=16, fs=250.0, band=(1.0, 40.0), kind="nl_coupled",
                 h=0.01, seed=42, bw_min_hz=0.02, rk4_substeps=1):
        super().__init__()
        self.substeps = int(rk4_substeps)   # 1 = published (one RK4 step per sample)
        assert kind in BANKS, kind
        self.kind, self.N, self.C, self.fs, self.h = kind, n_osc, n_channels, float(fs), h
        self.f_max = 0.45 * self.fs
        self.bw_min = bw_min_hz
        g = torch.Generator().manual_seed(seed)

        f0 = np.clip(logspaced_f0(n_osc, band), 0.5, self.f_max)
        self.log_f0 = nn.Parameter(torch.tensor(np.log(f0), dtype=torch.float32))
        mu_pub = torch.empty(n_osc).uniform_(-1.0, -0.4, generator=g)      # published prior
        bw0 = (-mu_pub * h * self.fs / (2 * math.pi)).clamp(min=bw_min_hz)
        self.log_bw = nn.Parameter(torch.log(bw0))

        # coupling W: drawn for every bank (same RNG stream); trainable only in nl_coupled
        Wr = torch.randn(n_osc, n_osc, generator=g) * 0.05
        Wi = torch.randn(n_osc, n_osc, generator=g) * 0.05
        W = torch.complex(Wr, Wi)
        if kind == "nl_coupled":
            self.W = nn.Parameter(W)
        else:
            self.register_buffer("W", torch.zeros_like(W))                 # frozen at 0

        self.nonlinear = kind != "lin"
        if self.nonlinear:
            self.beta = nn.Parameter(torch.ones(n_osc))
            self.gamma = nn.Parameter(torch.zeros(n_osc))
            self.amp_max = 5.0
            self.register_buffer("z0", torch.full((n_osc,), 0.01 + 0.01j, dtype=torch.complex64))
        self.input_gain = nn.Parameter(torch.tensor(3.0))

        s = 0.5 / math.sqrt(n_channels)
        Er = torch.randn(n_osc, n_channels, generator=g) * s
        Ei = torch.randn(n_osc, n_channels, generator=g) * s
        self.W_enc = nn.Parameter(torch.complex(Er, Ei))
        self.b_enc = nn.Parameter(torch.zeros(n_osc, dtype=torch.complex64))

        # nl_uncoupled_wide: the N x N complex parameters nl_coupled spends on W are spent in the
        # encoder instead, as a second complex layer I = ctanh(M ctanh(W_enc x + b) + b2).
        # M starts at identity + the same 0.05 noise scale as W, so at init the encoder is close to
        # the single-layer one. Extra real params: 2N^2 + 2N (W has 2N^2) -> matched to within 2N.
        Mr = torch.randn(n_osc, n_osc, generator=g) * 0.05
        Mi = torch.randn(n_osc, n_osc, generator=g) * 0.05
        if kind == "nl_uncoupled_wide":
            self.M_enc = nn.Parameter(torch.complex(Mr + torch.eye(n_osc), Mi))
            self.b2_enc = nn.Parameter(torch.zeros(n_osc, dtype=torch.complex64))

    # ---------------------------------------------------------------- parameters in physical units
    def f0_hz(self):
        return torch.exp(self.log_f0).clamp(0.5, self.f_max)

    def bw_hz(self):
        return torch.exp(self.log_bw).clamp(min=self.bw_min)

    def per_sample(self):
        """(mu_per_sample, omega_per_sample)."""
        return -2 * math.pi * self.bw_hz() / self.fs, 2 * math.pi * self.f0_hz() / self.fs

    # ---------------------------------------------------------------- encoder (published)
    def encode(self, x):
        ctanh = rn().complex_tanh
        gain = self.input_gain.clamp(0.1, 10.0)
        xt = x.transpose(1, 2).to(torch.complex64)                         # (B,T,C)
        e = ctanh(xt @ self.W_enc.t() + self.b_enc)
        if self.kind == "nl_uncoupled_wide":
            e = ctanh(e @ self.M_enc.t() + self.b2_enc)
        return gain * e                                                    # (B,T,N)

    # ---------------------------------------------------------------- dynamics
    def forward(self, x):
        """x: (B, C, T) real -> z: (B, T', N) complex trajectory (T' = T for lin, T-1 for nl,
        exactly as the published banks)."""
        I = self.encode(x)
        mu_s, om_s = self.per_sample()
        if not self.nonlinear:
            return self._linear(I, mu_s, om_s)
        return self._nonlinear(I, mu_s, om_s)

    def _linear(self, I, mu_s, om_s):
        # published FastResonanceNetwork: kernel exp(lam*h*n), causal FFT convolution
        B, T, N = I.shape
        lamh = torch.complex(mu_s, om_s).to(torch.complex64)
        n = torch.arange(T, device=I.device, dtype=torch.float32)
        kernel = torch.exp(lamh.unsqueeze(0) * n.unsqueeze(1))             # (T,N)
        L = 1
        while L < 2 * T:
            L *= 2
        Z = torch.fft.ifft(torch.fft.fft(I, n=L, dim=1) * torch.fft.fft(kernel, n=L, dim=0).unsqueeze(0), dim=1)
        return Z[:, :T, :]

    def _nonlinear(self, I, mu_s, om_s):
        # published ResonanceNetwork._f / rk4_step / safety clamp, with lam = (mu_s + i om_s)/h
        rk4 = rn().rk4_step
        lam = torch.complex(mu_s, om_s).to(torch.complex64) / self.h
        sat = torch.complex(self.beta, self.gamma)
        W = self.W

        def f(z, Ii):
            amp2 = z.real ** 2 + z.imag ** 2
            return lam * z - sat * amp2 * z + z @ W.t() + Ii

        B, T, N = I.shape
        z = self.z0.unsqueeze(0).expand(B, N).contiguous()
        traj = []
        k = self.substeps
        for t in range(T - 1):
            I0, I1 = I[:, t, :], I[:, t + 1, :]
            for j in range(k):  # k = 1 reproduces the published step exactly
                a0, a1 = I0 + (I1 - I0) * (j / k), I0 + (I1 - I0) * ((j + 1) / k)
                z = rk4(f, z, self.h / k, a0, 0.5 * (a0 + a1), a1)
                amp = z.abs()
                z = z * (torch.clamp(amp, max=self.amp_max) / (amp + 1e-8))
            traj.append(z)
        return torch.stack(traj, dim=1)

    def extra_param_count(self):
        """Real parameters beyond the shared core (for the capacity-control check)."""
        n = 0
        if self.kind == "nl_coupled":
            n += 2 * self.W.numel()
        if self.kind == "nl_uncoupled_wide":
            n += 2 * self.M_enc.numel() + 2 * self.b2_enc.numel()
        return n
