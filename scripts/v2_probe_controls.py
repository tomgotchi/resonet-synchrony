"""Extra controls for the v2 lag block on HGD development folds (train-split CV; descriptive).

Question: is the class information in the Im (lag) block really lagged coupling, or zero-lag information that
leaked into it (e.g. a fixed timing offset between channels turns Re coherency into a little Im coherency)?

  im_resid     Im features after removing everything linearly predictable from [power, Re] (ridge regression fitted
               on the training trials), then shrinkage LDA. A leak of zero-lag information is removed; genuine lagged
               structure stays.
  im_band_k    Im block of one band at a time: where does the information live? (A timing-offset leak grows with
               frequency; mu/beta lagged coupling does not.)
  re_from_im   the converse: how much of Im is linearly predictable from [power, Re] (median R^2 over features).

  python scripts/v2_probe_controls.py --subjects 1-6 --cache DIR   -> results/v2_probe/hgd_<tag>/controls.json"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync import baselines as BL, data_hgd  # noqa: E402
from resonet_sync.run import P1, write_json_atomic  # noqa: E402
from resonet_sync.v2 import CrossSpectralBank, V2Readout, csp_init_  # noqa: E402
from resonet_sync.v2_probe import blocks  # noqa: E402

spec = importlib.util.spec_from_file_location("probe_run", ROOT / "scripts" / "v2_probe_run.py")
PR = importlib.util.module_from_spec(spec); spec.loader.exec_module(PR)
spec = importlib.util.spec_from_file_location("dev", ROOT / "scripts" / "v2_hgd_dev.py")
DEV = importlib.util.module_from_spec(spec); spec.loader.exec_module(DEV)


def residualise(A_tr, B_tr, A_te, B_te, fit, lam=1.0):
    """Remove from B what ridge regression on A (fitted on rows `fit` of the training set) predicts."""
    mu_a, sd_a, mu_b = A_tr[fit].mean(0), A_tr[fit].std(0) + 1e-9, B_tr[fit].mean(0)
    Za, Zt = (A_tr - mu_a) / sd_a, (A_te - mu_a) / sd_a
    G = Za[fit].T @ Za[fit] + lam * len(Za[fit]) * np.eye(Za.shape[1])
    W = np.linalg.solve(G, Za[fit].T @ (B_tr[fit] - mu_b))
    r_tr, r_te = B_tr - mu_b - Za @ W, B_te - mu_b - Zt @ W
    r2 = 1 - (r_te ** 2).sum(0) / (((B_te - mu_b) ** 2).sum(0) + 1e-12)
    return r_tr, r_te, r2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-6"); ap.add_argument("--cache", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    cfg = dict(PR.HGD)
    tag = f"b{cfg['n_bands']}s{cfg['n_spatial']}_{cfg['band'][0]:g}-{cfg['band'][1]:g}Hz"
    out = PR.OUT / f"hgd_{tag}" / "controls.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    fs = data_hgd.FS_NATIVE / P1["decim"]
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    for s in range(int(lo), int(hi) + 1):
        if str(s) in res and len(res[str(s)]) == a.folds:
            continue
        X, y, _, _ = data_hgd.load(a.cache, s, P1["decim"], P1["cap"], P1["seed"])
        fold = DEV.folds(y, a.folds, P1["seed"])
        res.setdefault(str(s), {})
        for k in range(a.folds):
            if str(k) in res[str(s)]:
                continue
            tr, ho = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            Xa, ya, Xh, yh = X[tr], y[tr], X[ho], y[ho]
            vmask = data_hgd.train_val_split(ya, P1["val_frac"], P1["seed"])
            bank = CrossSpectralBank(X.shape[1], cfg["n_bands"], cfg["n_spatial"], fs, tuple(cfg["band"]), seed=42).eval()
            csp_init_(bank, Xa[~vmask], ya[~vmask])
            head = V2Readout(cfg["n_bands"], cfg["n_spatial"], fs, kind="cov+lag", skip_s=0.5)
            A, B = blocks(bank, head, Xa, ya, 42), blocks(bank, head, Xh, yh, 43)
            zl_tr, zl_te = np.c_[A["power"], A["re"]], np.c_[B["power"], B["re"]]
            r_tr, r_te, r2 = residualise(zl_tr, A["im"], zl_te, B["im"], ~vmask)
            n_tr, n_te, r2n = residualise(zl_tr, A["im_null"], zl_te, B["im_null"], ~vmask)
            P = A["im"].shape[1] // cfg["n_bands"]
            f0 = bank._band_params()[0].detach().numpy()
            rec = dict(im=BL.lda(A["im"], ya, B["im"], yh, vmask), im_resid=BL.lda(r_tr, ya, r_te, yh, vmask),
                       im_null_resid=BL.lda(n_tr, ya, n_te, yh, vmask),
                       im_r2_from_zero_lag_median=float(np.median(r2)), im_r2_from_zero_lag_p90=float(np.quantile(r2, 0.9)),
                       im_null_r2_median=float(np.median(r2n)),
                       im_by_band={f"{f0[b]:.1f}": BL.lda(A["im"][:, b * P:(b + 1) * P], ya, B["im"][:, b * P:(b + 1) * P], yh, vmask)
                                   for b in range(cfg["n_bands"])},
                       power_by_band={f"{f0[b]:.1f}": BL.lda(A["power"][:, b * cfg["n_spatial"]:(b + 1) * cfg["n_spatial"]], ya,
                                                            B["power"][:, b * cfg["n_spatial"]:(b + 1) * cfg["n_spatial"]], yh, vmask)
                                      for b in range(cfg["n_bands"])})
            res[str(s)][str(k)] = rec
            write_json_atomic(out, res)
            print(f"S{s:02d} fold {k} im {rec['im']:.3f} im_resid {rec['im_resid']:.3f} null_resid {rec['im_null_resid']:.3f} "
                  f"R2 {rec['im_r2_from_zero_lag_median']:+.3f}", flush=True)
    keys = ("im", "im_resid", "im_null_resid", "im_r2_from_zero_lag_median")
    print({s: {q: round(float(np.mean([fr[q] for fr in v.values()])), 3) for q in keys} for s, v in res.items()})


if __name__ == "__main__":
    main()
