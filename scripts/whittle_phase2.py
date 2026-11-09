"""Whittle (Gaussian, purely spectral) plug-in classifier on the Phase 2 synthetic datasets (review board
2026-11-09, Volkov). If it reaches ceiling, the class information is second order, and no decoder can show a
non-Gaussian (nonlinear-dynamics) advantage on these data.

  python scripts/whittle_phase2.py      -> results/phase2/whittle_oracle.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal.windows import dpss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync import synth  # noqa: E402


def tapered_fft(X, K=3, NW=2):
    tp = dpss(X.shape[-1], NW, K)
    return np.fft.rfft(X[:, None, :, :] * tp[None, :, None, :], axis=-1)          # (n, K, C, F)


def fit(F, y, nb, shrink=0.05):
    S = {}
    for k in np.unique(y):
        Fk = F[y == k][..., :nb]
        Sk = np.einsum("nkcf,nkdf->fcd", Fk, Fk.conj()) / (Fk.shape[0] * Fk.shape[1])
        tr = np.trace(Sk, axis1=1, axis2=2).real / Sk.shape[1]
        S[k] = (1 - shrink) * Sk + shrink * tr[:, None, None] * np.eye(Sk.shape[1])[None]
    return S


def loglik(F, S, nb):
    out = []
    for k in sorted(S):
        Si, ld = np.linalg.inv(S[k]), np.linalg.slogdet(S[k])[1]
        Ft = F[..., :nb]
        q = np.einsum("nkcf,fcd,nkdf->n", Ft.conj(), Si, Ft).real
        out.append(-(q + Ft.shape[1] * ld.sum()))
    return np.stack(out, 1)


rows = []
for kind in ("emergent", "driven"):
    for seed in range(10):
        Xtr, ytr, Xte, yte, _ = synth.make_dataset(seed, kind, 100, 40)
        nb = 41                                                     # 0-40 Hz, 1 Hz bins (1 s at 250 Hz)
        pred = loglik(tapered_fft(Xte), fit(tapered_fft(Xtr), ytr, nb), nb).argmax(1)
        acc = float(np.mean([(pred[yte == k] == k).mean() for k in np.unique(yte)]))
        arms = json.loads((ROOT / "results" / "phase2" / f"{kind}_seed{seed:02d}.json").read_text())["arms"]
        rows.append(dict(kind=kind, seed=seed, whittle_bal_acc=acc,
                         **{a: arms[a]["test_bal_acc"] for a in ("lin:sync", "nl_coupled:sync", "nl_uncoupled_wide:sync")}))
        print(rows[-1], flush=True)
summ = {k: {c: float(np.mean([r[c] for r in rows if r["kind"] == k]))
            for c in ("whittle_bal_acc", "lin:sync", "nl_coupled:sync", "nl_uncoupled_wide:sync")}
        for k in ("emergent", "driven")}
(ROOT / "results" / "phase2" / "whittle_oracle.json").write_text(json.dumps(dict(
    note="Gaussian spectral plug-in classifier (multitaper K=3, NW=2, 0-40 Hz, 5% shrinkage); balanced accuracy",
    means=summ, rows=rows), indent=1))
print(json.dumps(summ, indent=1))
