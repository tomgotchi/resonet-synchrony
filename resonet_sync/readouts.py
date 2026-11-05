"""Readouts: sync, power, surrogate, sync+power, trial_shuffle (+ the published head).

sync / power / surrogate are the published formulas (vendor/resonance_network.py):
  power     : log mean |z|^2 per oscillator                               (N dims)
  sync      : C_ij = <u_i conj(u_j)>_t, u = z/(|z|+1e-6), i<j, [Re, Im]    (2P dims)
  surrogate : same after rolling each oscillator's u by a fixed fraction of T
              (surr_frac from a generator seeded seed+1, as published)    (2P dims)
New:
  sync+power    : concatenation                                           (N + 2P dims)
  power+surrogate: power + surrogate coherence (capacity-matched control for sync+power; DEVIATIONS 22)
  trial_shuffle : C_ij = <u_i^(n) conj(u_j^(m))>_t with m a different trial of the SAME class
                  (shift-predictor logic: keeps class-locked/evoked phase structure, destroys
                  trial-specific coupling). Partner selection needs the label, including at test
                  time, so this arm is an analysis control, not a deployable decoder: its accuracy
                  bounds what evoked (class-locked) phase alone can give. Per Tom (2026-10-27) it is
                  NOT trained as a decoder arm; see decomposition.py for the evoked-vs-induced analysis.
Head (published): BatchNorm1d -> Dropout (0.2 for power, 0.5 otherwise) -> Linear.
"""
from __future__ import annotations

import torch
import torch.nn as nn

READOUTS = ("sync", "power", "surrogate", "sync+power", "power+surrogate", "trial_shuffle")


def unit(z):
    return z / (z.abs() + 1e-6)


def coherence_upper(u_a, u_b, iu, ju):
    T = u_a.shape[1]
    C = torch.einsum("bti,btj->bij", u_a, u_b.conj()) / T
    Cu = C[:, iu, ju]
    return torch.cat([Cu.real, Cu.imag], dim=1)


class Readout(nn.Module):
    def __init__(self, n_osc, n_classes=4, kind="sync", seed=42, dropout=None):
        super().__init__()
        assert kind in READOUTS, kind
        self.kind, self.N = kind, n_osc
        iu, ju = torch.triu_indices(n_osc, n_osc, offset=1)
        self.register_buffer("iu", iu)
        self.register_buffer("ju", ju)
        P = iu.numel()
        dim = {"power": n_osc, "sync": 2 * P, "surrogate": 2 * P, "sync+power": n_osc + 2 * P,
               "power+surrogate": n_osc + 2 * P, "trial_shuffle": 2 * P}[kind]
        self.register_buffer("surr_frac",
                             torch.rand(n_osc, generator=torch.Generator().manual_seed(seed + 1)))
        drop = dropout if dropout is not None else (0.2 if kind == "power" else 0.5)
        self.feat_norm = nn.BatchNorm1d(dim)
        self.feat_drop = nn.Dropout(drop)
        self.readout = nn.Linear(dim, n_classes)
        self.dim = dim

    @property
    def needs_partner(self):
        return self.kind == "trial_shuffle"

    def features(self, z, z_partner=None):
        if self.kind == "power":
            return torch.log((z.abs() ** 2).mean(dim=1) + 1e-6)
        u = unit(z)
        if self.kind == "trial_shuffle":
            assert z_partner is not None, "trial_shuffle needs a same-class partner trajectory"
            return coherence_upper(u, unit(z_partner), self.iu, self.ju)
        if self.kind in ("surrogate", "power+surrogate"):
            T = u.shape[1]
            sh = (self.surr_frac * T).long()
            u = torch.stack([torch.roll(u[:, :, i], int(sh[i]), dims=1) for i in range(self.N)], dim=2)
        f = coherence_upper(u, u, self.iu, self.ju)
        if self.kind in ("sync+power", "power+surrogate"):     # power+surrogate: capacity-matched control (DEVIATIONS 22)
            f = torch.cat([torch.log((z.abs() ** 2).mean(dim=1) + 1e-6), f], dim=1)
        return f

    def forward(self, z, z_partner=None):
        return self.readout(self.feat_drop(self.feat_norm(self.features(z, z_partner))))


class RNModel(nn.Module):
    """Bank + readout. forward(x, x_partner=None) -> logits."""

    def __init__(self, bank, readout):
        super().__init__()
        self.bank, self.head = bank, readout

    @property
    def needs_partner(self):
        return self.head.needs_partner

    def forward(self, x, x_partner=None):
        z = self.bank(x)
        zp = self.bank(x_partner) if self.head.needs_partner else None
        return self.head(z, zp)
