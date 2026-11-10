"""ResoNet v2 closed-form path (resonet_sync/v2_probe.py) on synthetic ground truth and on HGD development folds.

  python scripts/v2_probe_run.py --synthetic [--workers 3]              # seeds 3700-3709, generators of DEVIATION 27
  python scripts/v2_probe_run.py --hgd --subjects 1-6 --cache DIR       # 5-fold CV inside the TRAIN split only

Synthetic: 8 bands 4-40 Hz x 4 CSP filters (16 channels). HGD: 12 bands 4-100 Hz x 8 CSP filters (128 channels).
Outputs: results/v2_probe/synthetic.json, results/v2_probe/hgd_<tag>/subj_XX.json + summary.json.
HGD numbers are DEVELOPMENT results (descriptive; the test split is never read by a model)."""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync.stats import sign_flip_p, t_ci  # noqa: E402

OUT = ROOT / "results" / "v2_probe"
SYN = dict(n_bands=8, n_spatial=4, band=[4.0, 40.0])
HGD = dict(n_bands=12, n_spatial=8, band=[4.0, 100.0])
SYN_SEEDS = list(range(3700, 3710))
MARGIN = 0.05


def syn_one(job):
    gen, seed = job
    import torch
    torch.set_num_threads(1)
    from resonet_sync import baselines as BL, data_hgd
    from resonet_sync.screens import calib_p1 as CP, v2_study as V
    from resonet_sync.v2_probe import probe
    Xtr, ytr, Xte, yte, Str, Ste = V.make_dataset(gen, seed, V.load_diff())
    vmask = data_hgd.train_val_split(ytr, 0.15, 42)
    acc = probe(Xtr, ytr, Xte, yte, vmask, CP.FS, **SYN)
    W = BL.fbcsp_fit(Xtr[~vmask], ytr[~vmask], CP.FS)
    acc["FBCSP_LDA"] = BL.lda(BL.fbcsp(Xtr, W, CP.FS), ytr, BL.fbcsp(Xte, W, CP.FS), yte, vmask)
    acc["oracle"] = CP._logreg_val(V.oracle_features(gen, Str), ytr, V.oracle_features(gen, Ste), yte, vmask)
    return dict(gen=gen, seed=seed, acc=acc)


def synthetic(workers):
    from resonet_sync.screens import v2_study as V
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(workers) as p:
        rows = p.map(syn_one, [(g, s) for g in V.GENS for s in SYN_SEEDS])
    summ = {}
    for g in V.GENS:
        rs = [r["acc"] for r in rows if r["gen"] == g]
        L = np.array([r["L"] for r in rs])
        ci = t_ci(L, 0.90)
        summ[g] = dict(n=len(rs), means={k: float(np.mean([r[k] for r in rs])) for k in rs[0]},
                       L=dict(mean=float(L.mean()), ci90=ci, p_signflip=sign_flip_p(L), n_pos=int((L > 0).sum()),
                              valid=bool(-MARGIN < ci[0] and ci[1] < MARGIN),
                              sensitive=bool(sign_flip_p(L) < 0.05 and L.mean() >= MARGIN)))
    ok = all(summ[g]["L"]["valid"] for g in ("G0", "G1", "G1r")) and all(summ[g]["L"]["sensitive"] for g in ("G2i", "G2"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic.json").write_text(json.dumps(dict(config=SYN, seeds=SYN_SEEDS, criteria="DEVIATION 27 (same as the trained path)",
                                                        L_passes=bool(ok), summary=summ, rows=rows), indent=1))
    for g, S in summ.items():
        print(g, {k: round(v, 3) for k, v in S["means"].items() if k not in ("L", "R")})
        print(f"   L {S['L']['mean']:+.3f} CI90[{S['L']['ci90'][0]:+.3f},{S['L']['ci90'][1]:+.3f}] p {S['L']['p_signflip']:.4f} "
              f"+{S['L']['n_pos']}/{S['n']} valid={S['L']['valid']} sensitive={S['L']['sensitive']}")
    print("L passes:", ok)


def hgd(a):
    import importlib.util
    import torch
    torch.set_num_threads(a.threads)
    from resonet_sync import baselines as BL, data_hgd
    from resonet_sync.run import P1, load_config, provenance, write_json_atomic
    from resonet_sync.v2_probe import probe
    spec = importlib.util.spec_from_file_location("dev", ROOT / "scripts" / "v2_hgd_dev.py")
    dev = importlib.util.module_from_spec(spec); spec.loader.exec_module(dev)
    cfg = dict(HGD, n_bands=a.bands, n_spatial=a.spatial, band=[a.fmin, a.fmax])
    tag = f"b{a.bands}s{a.spatial}_{a.fmin:g}-{a.fmax:g}Hz"
    out = OUT / f"hgd_{tag}"; out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or load_config().get("paths", {}).get("hgd_cache_hg")
    fs = data_hgd.FS_NATIVE / P1["decim"]
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    for s in range(int(lo), int(hi) + 1):
        f = out / f"subj_{s:02d}.json"
        rec = json.loads(f.read_text()) if f.exists() else dict(subject=s, config=cfg, folds={}, provenance=provenance("cpu"))
        if len(rec["folds"]) == a.folds:
            continue
        X, y, _, _ = data_hgd.load(cache, s, P1["decim"], P1["cap"], P1["seed"])          # test split never used
        fold = dev.folds(y, a.folds, P1["seed"])
        for k in range(a.folds):
            if str(k) in rec["folds"]:
                continue
            tr, ho = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            Xa, ya, Xh, yh = X[tr], y[tr], X[ho], y[ho]
            vmask = data_hgd.train_val_split(ya, P1["val_frac"], P1["seed"])
            acc = probe(Xa, ya, Xh, yh, vmask, fs, **cfg)
            W = BL.fbcsp_fit(Xa[~vmask], ya[~vmask], fs)
            acc["FBCSP_LDA"] = BL.lda(BL.fbcsp(Xa, W, fs), ya, BL.fbcsp(Xh, W, fs), yh, vmask)
            acc["FBLV_LDA"] = BL.lda(BL.fblv(Xa, fs), ya, BL.fblv(Xh, fs), yh, vmask)
            rec["folds"][str(k)] = acc
            write_json_atomic(f, rec)
            print(f"S{s:02d} fold {k}", {q: round(v, 3) for q, v in acc.items()}, flush=True)
    per = [json.loads(p.read_text()) for p in sorted(out.glob("subj_[0-9]*.json"))]
    keys = list(per[0]["folds"]["0"]) if per else []
    subj = {p["subject"]: {q: float(np.mean([fr[q] for fr in p["folds"].values()])) for q in keys} for p in per}
    summ = dict(note="DEVELOPMENT: 5-fold CV inside the training split; descriptive means; no inference", config=cfg,
                subjects=sorted(subj), mean_over_subjects={q: float(np.mean([m[q] for m in subj.values()])) for q in keys},
                per_subject=subj, n_folds={p["subject"]: len(p["folds"]) for p in per})
    write_json_atomic(out / "summary.json", summ)
    print(json.dumps({q: round(v, 3) for q, v in summ["mean_over_subjects"].items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--hgd", action="store_true")
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--subjects", default="1-6"); ap.add_argument("--cache", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bands", type=int, default=HGD["n_bands"]); ap.add_argument("--spatial", type=int, default=HGD["n_spatial"])
    ap.add_argument("--fmin", type=float, default=HGD["band"][0]); ap.add_argument("--fmax", type=float, default=HGD["band"][1])
    a = ap.parse_args(argv)
    if a.synthetic:
        synthetic(a.workers)
    if a.hgd:
        hgd(a)


if __name__ == "__main__":
    main()
