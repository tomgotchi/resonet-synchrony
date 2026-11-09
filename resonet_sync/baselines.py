"""Standard band-power baselines and the frozen-bank analyses of DEVIATION 23 (review board 2026-11-09).

B1  FBLV  + shrinkage LDA         filter-bank log-variance (9 causal 4th-order Butterworth bands, 4-40 Hz)
B2  FBCSP + shrinkage LDA         one-vs-rest CSP per band, 2 filter pairs per class, 10% shrinkage
B3  FB tangent space + logistic   OAS covariances per band, log-Euclidean reference, upper-triangle tangent vectors
Same split as Phase 1: cap 80/class, the seed-42 15% validation carve inside train (used for hyperparameters only),
fit on train minus the carve, scored on test (balanced accuracy). The first 0.5 s of each band-passed trial is
dropped (causal filter transient). FBLV and FBCSP definitions are identical to screens/calib_p1.py (DEVIATION 24;
tests/test_baselines.py checks this), so F5 uses exactly the calibrated baseline.

F5 (DEVIATION 23): (B + sync) - (B + surrogate_random) on the frozen bank of the {bank}:sync+power arm.
D2: logistic-regression probes on the frozen bank's sync / fixed-roll / fresh-roll surrogate features."""
from __future__ import annotations

import numpy as np
from scipy import signal

FB_BANDS = [(lo, lo + 4.0) for lo in range(4, 40, 4)]
HG_BANDS = [(40.0, 48.0), (52.0, 60.0), (60.0, 80.0), (80.0, 100.0), (100.0, 124.0)]   # descriptive; 48-52 Hz (mains) skipped
DROP = 125


def bal_acc(pred, y):
    return float(np.mean([(pred[y == k] == k).mean() for k in np.unique(y)]))


def filter_bank(X, fs=250.0, bands=FB_BANDS):
    return [signal.sosfilt(signal.butter(4, b, btype="band", fs=fs, output="sos"), X, axis=-1)[..., DROP:]
            for b in bands]


def fblv(X, fs=250.0, bands=FB_BANDS):
    return np.concatenate([np.log(x.var(-1)) for x in filter_bank(X, fs, bands)], 1)


def fbcsp_fit(Xtr, ytr, fs=250.0, bands=FB_BANDS, n_pairs=2, shrink=0.1):
    from scipy.linalg import eigh
    C = Xtr.shape[1]
    W = []
    for x in filter_bank(Xtr, fs, bands):
        cov = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
        cov = cov / np.trace(cov, axis1=1, axis2=2)[:, None, None]
        Wb = []
        for k in np.unique(ytr):
            Ca, Cb = cov[ytr == k].mean(0), cov[ytr != k].mean(0)
            Ca, Cb = [(1 - shrink) * M + shrink * np.trace(M) / C * np.eye(C) for M in (Ca, Cb)]
            _, V = eigh(Ca, Ca + Cb)
            Wb.append(np.c_[V[:, :n_pairs], V[:, -n_pairs:]])
        W.append(np.concatenate(Wb, 1))
    return W


def fbcsp(X, W, fs=250.0, bands=FB_BANDS):
    return np.concatenate([np.log(np.einsum("cf,nct->nft", w, x).var(-1)) for w, x in zip(W, filter_bank(X, fs, bands))], 1)


def _sym_fun(M, f):
    w, V = np.linalg.eigh(M)
    return (V * f(w)) @ V.T


def tangent_space(Xtr, Xte, fs=250.0, bands=FB_BANDS):
    """OAS covariance per band and trial; log-Euclidean mean of train as reference; tangent vectors (upper triangle,
    off-diagonals x sqrt 2) of logm(R^-1/2 C R^-1/2)."""
    from sklearn.covariance import OAS
    out_tr, out_te = [], []
    for xtr, xte in zip(filter_bank(Xtr, fs, bands), filter_bank(Xte, fs, bands)):
        covs = [np.array([OAS(assume_centered=True).fit(t.T).covariance_ for t in x]) for x in (xtr, xte)]
        ref = _sym_fun(np.mean([_sym_fun(c, np.log) for c in covs[0]], 0), np.exp)
        isq = _sym_fun(ref, lambda w: 1 / np.sqrt(w))
        iu = np.triu_indices(Xtr.shape[1])
        wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
        for cs, out in zip(covs, (out_tr, out_te)):
            out.append(np.array([_sym_fun(isq @ c @ isq, np.log)[iu] * wgt for c in cs]))
    return np.concatenate(out_tr, 1), np.concatenate(out_te, 1)


def logreg_val(Ftr, ytr, Fte, yte, vmask, Cs=np.logspace(-3, 2, 11)):
    """L2 logistic regression; C chosen on the validation carve; refit on train minus the carve."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    a, b = ~vmask, vmask
    sc = StandardScaler().fit(Ftr[a])
    A, B, Te = sc.transform(Ftr[a]), sc.transform(Ftr[b]), sc.transform(Fte)
    best = max(Cs, key=lambda c: (LogisticRegression(C=c, max_iter=3000).fit(A, ytr[a]).predict(B) == ytr[b]).mean())
    return bal_acc(LogisticRegression(C=best, max_iter=3000).fit(A, ytr[a]).predict(Te), yte)


def lda(Ftr, ytr, Fte, yte, vmask):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr[~vmask])
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(Ftr[~vmask]), ytr[~vmask])
    return bal_acc(m.predict(sc.transform(Fte)), yte)


def baselines(Xtr, ytr, Xte, yte, vmask, fs=250.0, tangent=True):
    """B1-B3 (+ the high-gamma band set for B1, descriptive)."""
    W = fbcsp_fit(Xtr[~vmask], ytr[~vmask], fs)
    out = dict(B1_FBLV_LDA=lda(fblv(Xtr, fs), ytr, fblv(Xte, fs), yte, vmask),
               B2_FBCSP_LDA=lda(fbcsp(Xtr, W, fs), ytr, fbcsp(Xte, W, fs), yte, vmask),
               B1hg_FBLV_LDA_40_124Hz=lda(fblv(Xtr, fs, HG_BANDS), ytr, fblv(Xte, fs, HG_BANDS), yte, vmask)
               if fs / 2 > 124 else None)
    if tangent:
        a, b = tangent_space(Xtr, Xte, fs)
        out["B3_tangent_logreg"] = logreg_val(a, ytr, b, yte, vmask)
    return out


def frozen_bank_analyses(model, Xtr, ytr, Xte, yte, vmask, fs=250.0, n_osc=16, seed=42):
    """F5 candidates and D2 probes on a trained model's frozen bank (DEVIATION 23 / 24)."""
    import torch
    from .readouts import Readout
    model.eval()
    with torch.no_grad():
        ztr, zte = model.bank(torch.as_tensor(Xtr)), model.bank(torch.as_tensor(Xte))
    feats = {}
    for kind in ("sync", "surrogate", "surrogate_random", "power"):
        r = Readout(n_osc, kind=kind, seed=seed)
        torch.manual_seed(seed + 7)
        with torch.no_grad():
            feats[kind] = (r.features(ztr).numpy(), r.features(zte).numpy())
    W = fbcsp_fit(Xtr[~vmask], ytr[~vmask], fs)
    feats["FBLV"] = (fblv(Xtr, fs), fblv(Xte, fs))
    feats["FBCSP"] = (fbcsp(Xtr, W, fs), fbcsp(Xte, W, fs))
    cat = lambda *ks: (np.concatenate([feats[k][0] for k in ks], 1), np.concatenate([feats[k][1] for k in ks], 1))
    out = {}
    for name, ks in [("FBLV", ("FBLV",)), ("FBCSP", ("FBCSP",)),
                     ("FBLV+sync", ("FBLV", "sync")), ("FBLV+surrogate_random", ("FBLV", "surrogate_random")),
                     ("FBCSP+sync", ("FBCSP", "sync")), ("FBCSP+surrogate_random", ("FBCSP", "surrogate_random")),
                     ("probe:sync", ("sync",)), ("probe:surrogate", ("surrogate",)),
                     ("probe:surrogate_random", ("surrogate_random",)), ("probe:power", ("power",))]:
        a, b = cat(*ks)
        out[name] = logreg_val(a, ytr, b, yte, vmask)
    return out
