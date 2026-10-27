"""Evoked-vs-induced decomposition of oscillator coherence (replaces `trial_shuffle` as a decoder arm).

For a trained bank and one split S (train or test, never mixed):
  genuine   C_ij(n)    = < u_i^(n) conj u_j^(n) >_t                 (what the `sync` readout sees)
  shuffled  C_ij(n, m) = < u_i^(n) conj u_j^(m) >_t,  m != n, y_m = y_n, m in S
            -> keeps class-locked (evoked) phase structure, destroys trial-specific coupling
  induced   = genuine - shuffled
Reported: class-consistent magnitude of each, |mean over same-class trials of C_ij| averaged over
pairs and classes (per-trial |C_ij| is high for any narrowband pair and cannot separate evoked from
induced); evoked_fraction = shuffled / genuine; the cross-trial same-oscillator PLV (class-averaged
diagonal of the shuffled matrix, a direct evoked-locking index); and **train-only** stratified 5-fold CV balanced
accuracy of a logistic-regression readout on each feature set. Inside the CV, shuffle partners are
drawn within the same fold (training fold with training fold, held-out fold with held-out fold).

Partner selection uses class labels by construction (shift-predictor logic), so these numbers
describe *where* class information lives (evoked vs induced). They are NOT decoder accuracies and
are kept out of the main accuracy table. Test-split values are descriptive magnitudes only.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def trajectories(bank, X, device="cpu", batch=64):
    bank.eval()
    out = []
    for s in range(0, len(X), batch):
        out.append(bank(torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device)).cpu())
    z = torch.cat(out).numpy()
    return z / (np.abs(z) + 1e-6)


def same_class_partners(y, idx, rng):
    """For each i in idx, a different index of the same class drawn from idx itself."""
    idx = np.asarray(idx)
    out = np.empty(len(idx), int)
    for k, i in enumerate(idx):
        pool = idx[(y[idx] == y[i]) & (idx != i)]
        out[k] = pool[rng.integers(len(pool))] if len(pool) else i
    return out


def coh(u_a, u_b):
    """(n, T, N) x (n, T, N) -> (n, N, N) complex coherence."""
    return np.einsum("nti,ntj->nij", u_a, u_b.conj()) / u_a.shape[1]


def _feat(C):
    iu, ju = np.triu_indices(C.shape[1], 1)
    c = C[:, iu, ju]
    return np.concatenate([c.real, c.imag], axis=1)


def _class_avg_abs(C, y, iu, ju):
    """mean over classes and pairs of |mean_{n in class} C_ij(n)|: class-consistent coherence.
    (Per-trial |C_ij| is high for any narrowband pair, so it cannot separate evoked from induced.)"""
    return float(np.mean([np.abs(C[y == c][:, iu, ju].mean(0)).mean() for c in np.unique(y)]))


def magnitudes(u, y, partners):
    G = coh(u, u)
    S = coh(u, u[partners])
    N = u.shape[2]
    iu, ju = np.triu_indices(N, 1)
    di = np.arange(N)
    g, s_ = _class_avg_abs(G, y, iu, ju), _class_avg_abs(S, y, iu, ju)
    return dict(genuine_classavg_abs=g, shuffled_classavg_abs=s_,
                induced_classavg_abs=_class_avg_abs(G - S, y, iu, ju),
                evoked_fraction=float(s_ / max(g, 1e-12)),
                cross_trial_same_osc_plv=_class_avg_abs(S, y, di, di))


def cv_discriminability(u, y, seed=42, k=5, C=1.0):
    """Train-split-only CV. Returns balanced accuracy for genuine / shuffled / induced features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    G = _feat(coh(u, u))
    scores = {"genuine": [], "shuffled": [], "induced": []}
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(G, y):
        p_tr, p_va = same_class_partners(y, tr, rng), same_class_partners(y, va, rng)
        S_tr, S_va = _feat(coh(u[tr], u[p_tr])), _feat(coh(u[va], u[p_va]))
        feats = {"genuine": (G[tr], G[va]), "shuffled": (S_tr, S_va),
                 "induced": (G[tr] - S_tr, G[va] - S_va)}
        for name, (a, b) in feats.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            clf.fit(a, y[tr])
            scores[name].append(balanced_accuracy_score(y[va], clf.predict(b)))
    return {f"cv_bal_acc_{k_}": float(np.mean(v)) for k_, v in scores.items()}


def decompose(bank, tr_split, te_split, device="cpu", seed=42):
    """tr_split / te_split: train.Split objects (their .fixed partners are same-split, same-class)."""
    out = {}
    u_tr = trajectories(bank, tr_split.X.numpy(), device)
    y_tr = tr_split.y.numpy()
    out["train"] = magnitudes(u_tr, y_tr, tr_split.fixed)
    out["train"].update(cv_discriminability(u_tr, y_tr, seed))
    u_te = trajectories(bank, te_split.X.numpy(), device)
    out["test_descriptive"] = magnitudes(u_te, te_split.y.numpy(), te_split.fixed)
    return out
