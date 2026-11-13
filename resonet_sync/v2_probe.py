"""ResoNet v2, closed-form path ("probe"): CSP-initialised cross-spectral bank -> feature blocks -> shrinkage LDA.

No gradient training: the spatial filters come from CSP on the training trials (v2.csp_init_), the resonators stay
at their constant-Q initialisation, and each feature set is classified by shrinkage LDA fitted on the training
trials. It is fast (seconds per fold), so it is the path used to look at feature blocks on many folds and
subjects, with two controls:

  im_null    Im coherency after a fresh circular shift per trial and projection (seeded): the dimension-matched
             null for the lag block. Destroys every cross-term; keeps each projection's own series.
  im_evoked  Im coherency between projection s of trial n and projection s' of ANOTHER trial of the same class
             (shift predictor). Only structure that is time-locked to the trial start survives. Partner selection
             needs the label, also in the held-out fold, so this is an analysis control, not a decoder.

Contrasts (per fold or dataset):
  L   = acc(power + re + im) - acc(power + re + im_null)      lagged coupling beyond band covariance
  R   = acc(power + re) - acc(power)                          zero-lag covariance beyond power
  im, im_null, im_evoked alone                                is there class information in the lag block at all,
                                                              and is it time-locked?"""
from __future__ import annotations

import numpy as np
import torch

from . import baselines as BL
from .v2 import CrossSpectralBank, V2Readout, csp_init_

FEATURE_SETS = {"power": ("power",), "power+re": ("power", "re"), "power+re+im": ("power", "re", "im"),
                "power+re+im_null": ("power", "re", "im_null"), "im": ("im",), "im_null": ("im_null",),
                "im_evoked": ("im_evoked",),
                # board 2026-11-13: UNNORMALISED imaginary cross-spectrum. The coherency normalisation divides by the
                # projections' powers, so `im` responds to the power of unrelated sources whenever any lagged
                # structure exists, even a class-independent one. `im_raw` does not.
                "im_raw": ("im_raw",), "im_raw_null": ("im_raw_null",)}


def _same_class_partner(y, rng):
    out = np.arange(len(y))
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > 1:
            out[idx] = idx[(np.arange(len(idx)) + rng.integers(1, len(idx))) % len(idx)]     # never itself
    return out


def blocks(bank, head, X, y, seed, batch=64):
    """Feature blocks for trials X (n, C, T): power, re, im, im_null (fresh roll), im_evoked (same-class partner)."""
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(int(seed))
    partner = _same_class_partner(np.asarray(y), rng)
    out = {k: [] for k in ("power", "re", "im", "im_null", "im_evoked", "im_raw", "im_raw_null")}
    X = torch.as_tensor(X, dtype=torch.float32)
    skip = head.skip
    with torch.no_grad():
        for s in range(0, len(X), batch):
            z = bank(X[s:s + batch])
            z = z[:, skip:] if z.shape[1] > 2 * skip else z
            B, T = z.shape[:2]
            p, coh = head._coherency(z)
            out["power"].append(torch.log(p).reshape(B, -1)); out["re"].append(coh.real.reshape(B, -1))
            out["im"].append(coh.imag.reshape(B, -1))
            raw = lambda zz: (torch.einsum("btfs,btfr->bfsr", zz, zz.conj()) / T)[:, :, head.iu, head.ju].imag.reshape(B, -1)
            out["im_raw"].append(raw(z))
            sh = torch.randint(0, T, (B, 1, bank.F, bank.S), generator=g)
            idx = (torch.arange(T)[None, :, None, None] - sh) % T
            zr = torch.gather(z, 1, idx)
            out["im_null"].append(head._coherency(zr)[1].imag.reshape(B, -1))
            out["im_raw_null"].append(raw(zr))
            zp = bank(X[partner[s:s + batch]])
            zp = zp[:, skip:] if zp.shape[1] > 2 * skip else zp
            Cx = torch.einsum("btfs,btfr->bfsr", z, zp.conj()) / T                 # projection s (trial n) x r (partner)
            pp = torch.diagonal(torch.einsum("btfs,btfr->bfsr", zp, zp.conj()) / T, dim1=-2, dim2=-1).real.clamp_min(1e-12)
            cx = Cx[:, :, head.iu, head.ju] / torch.sqrt(p[:, :, head.iu] * pp[:, :, head.ju])
            out["im_evoked"].append(cx.imag.reshape(B, -1))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def probe(Xtr, ytr, Xte, yte, vmask, fs, n_bands, n_spatial, band, skip_s=0.5, seed=42, csp=True):
    """CSP filters and LDA are fitted on Xtr[~vmask] only. Returns {feature set: balanced accuracy on (Xte, yte)}."""
    bank = CrossSpectralBank(Xtr.shape[1], n_bands, n_spatial, fs, tuple(band), seed=seed).eval()
    if csp:
        csp_init_(bank, Xtr[~vmask], ytr[~vmask])
    head = V2Readout(n_bands, n_spatial, fs, kind="cov+lag", skip_s=skip_s)
    a, b = blocks(bank, head, Xtr, ytr, seed), blocks(bank, head, Xte, yte, seed + 1)
    cat = lambda blk, ks: np.concatenate([blk[k] for k in ks], 1)
    acc = {name: BL.lda(cat(a, ks), ytr, cat(b, ks), yte, vmask) for name, ks in FEATURE_SETS.items()}
    acc["L"] = acc["power+re+im"] - acc["power+re+im_null"]
    acc["R"] = acc["power+re"] - acc["power"]
    return acc
