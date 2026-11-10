"""ResoNet v2 on HGD, DEVELOPMENT mode: stratified k-fold cross-validation INSIDE the training split.

The test split is never given to a model, so v2 can be tuned here without touching the data that a later
confirmatory test (or the frozen Phase 1 analysis) uses. Development subjects default to 1-6; subjects 7-14 are
left untouched for a later one-shot evaluation with a frozen v2 configuration.

  python scripts/v2_hgd_dev.py --subjects 1-6 [--cache DIR] [--folds 5] [--refs lin:sync+power,shallow]
                               [--bands 12 --spatial 4 --fmax 100]

Arms per fold: v2:power, v2:cov, v2:cov+lag, v2:cov+lag_null; baselines FBLV and FBCSP with shrinkage LDA;
--refs adds reference arms on the same folds, e.g. the Hz-bank lin:sync+power (v1) and the published
ShallowConvNet (`shallow`, slow on CPU). Arms can be added to finished subjects later (resumable per fold and arm).
Writes results/v2_hgd_dev/<tag>/subj_XX.json and summary.json (means only; no inference: this is development).
Caveat: the loader's channel z-scoring uses all capped training trials, so a held-out fold shares normalisation
statistics (no labels) with its training folds."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync import baselines as BL, data_hgd  # noqa: E402
from resonet_sync.run import P1, V2_HGD, build, load_config, provenance, write_json_atomic  # noqa: E402
from resonet_sync.train import Split, evaluate, train  # noqa: E402

V2_ARMS = ["v2:power", "v2:cov", "v2:cov+lag", "v2:cov+lag_null"]


def folds(y, k, seed):
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        f[idx] = np.arange(len(idx)) % k
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-6"); ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--refs", default="")
    ap.add_argument("--bands", type=int, default=V2_HGD["n_bands"]); ap.add_argument("--spatial", type=int, default=V2_HGD["n_spatial"])
    ap.add_argument("--fmin", type=float, default=V2_HGD["band"][0]); ap.add_argument("--fmax", type=float, default=V2_HGD["band"][1])
    ap.add_argument("--epochs", type=int, default=P1["epochs"]); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--csp-init", action="store_true",
                    help="start v2's spatial filters from CSP filters fitted on each fold's training trials")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(P1, epochs=a.epochs, v2=dict(n_bands=a.bands, n_spatial=a.spatial, band=[a.fmin, a.fmax], skip_s=V2_HGD["skip_s"]))
    tag = f"b{a.bands}s{a.spatial}_{a.fmin:g}-{a.fmax:g}Hz" + ("_csp" if a.csp_init else "") + \
        (f"_ep{a.epochs}" if a.epochs != P1["epochs"] else "")
    out = ROOT / "results" / "v2_hgd_dev" / tag; out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or load_config().get("paths", {}).get("hgd_cache_hg")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fs = data_hgd.FS_NATIVE / cfg["decim"]
    arms = V2_ARMS + [r for r in a.refs.split(",") if r]
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    crit = torch.nn.CrossEntropyLoss()
    for s in range(int(lo), int(hi) + 1):
        f = out / f"subj_{s:02d}.json"
        rec = json.loads(f.read_text()) if f.exists() else dict(subject=s, config=json.loads(json.dumps(cfg)), folds={},
                                                                provenance=provenance(dev))
        X, y, _, _ = data_hgd.load(cache, s, cfg["decim"], cfg["cap"], cfg["seed"])       # test split never used
        fold = folds(y, a.folds, cfg["seed"])
        for k in range(a.folds):
            fr = rec["folds"].setdefault(str(k), {})
            tr_idx, ho_idx = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            Xa, ya, Xh, yh = X[tr_idx], y[tr_idx], X[ho_idx], y[ho_idx]
            vmask = data_hgd.train_val_split(ya, cfg["val_frac"], cfg["seed"])
            if "FBLV_LDA" not in fr:
                W = BL.fbcsp_fit(Xa[~vmask], ya[~vmask], fs)
                fr["FBLV_LDA"] = BL.lda(BL.fblv(Xa, fs), ya, BL.fblv(Xh, fs), yh, vmask)
                fr["FBCSP_LDA"] = BL.lda(BL.fbcsp(Xa, W, fs), ya, BL.fbcsp(Xh, W, fs), yh, vmask)
            tr, va, ho = Split(Xa[~vmask], ya[~vmask], 42), Split(Xa[vmask], ya[vmask], 43), Split(Xh, yh, 44)
            Wcsp = None
            for arm in arms:
                if arm in fr:
                    continue
                torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
                m = build(arm, X.shape[1], X.shape[2], fs, cfg)
                if a.csp_init and arm.startswith("v2:"):
                    if Wcsp is None:                                  # once per fold, from its training trials only
                        from resonet_sync.v2 import csp_init_
                        Wcsp = csp_init_(m.bank, Xa[~vmask], ya[~vmask]).W.detach().clone()
                    with torch.no_grad():
                        m.bank.W.copy_(Wcsp)
                m, hist = train(m, tr, va, dev, max_epochs=cfg["epochs"], batch=cfg["batch"], seed=cfg["seed"])
                torch.manual_seed(cfg["seed"] + 7)
                fr[arm] = evaluate(m, ho, dev, crit, cfg["batch"])["bal_acc"]
                write_json_atomic(f, rec)
                print(f"S{s:02d} fold {k} {arm:18s} {fr[arm]:.3f} (best ep {hist['best_epoch']})", flush=True)
            write_json_atomic(f, rec)
    per = [json.loads(p.read_text()) for p in sorted(out.glob("subj_[0-9]*.json"))]
    keys = sorted({a_ for p in per for fr in p["folds"].values() for a_ in fr})
    subj_means = {p["subject"]: {k: float(np.mean([fr[k] for fr in p["folds"].values() if k in fr])) for k in keys
                                 if any(k in fr for fr in p["folds"].values())} for p in per}
    summ = dict(note="DEVELOPMENT: train-split cross-validation, descriptive means only", tag=tag, subjects=sorted(subj_means),
                mean_over_subjects={k: float(np.mean([m[k] for m in subj_means.values() if k in m])) for k in keys},
                per_subject=subj_means)
    write_json_atomic(out / "summary.json", summ)
    print(json.dumps(summ["mean_over_subjects"], indent=1))


if __name__ == "__main__":
    main()
