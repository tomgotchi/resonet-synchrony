"""DEVIATIONS 16: confirmatory test of  emergent := S_TE > t_TE and S_inst_dn < t_dn  on fresh seeds 50-59.
python -m resonet_sync.screen6 [--workers 4]."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import lds
from . import mechanism as Mx
from . import synth
from .screen2 import _params

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(50, 60))
T_TE, T_DN = 0.015295, 0.115964                       # frozen (DEVIATIONS 14 / 16)
# name -> (screen2 condition for common params, rho, obs SNR or None)
CONDS = {"rho95_snr20": ("base", 0.95, 20.0), "rho95_snr10": ("base", 0.95, 10.0),
         "rho95_snr5": ("base", 0.95, 5.0), "rho95_snr0": ("base", 0.95, 0.0),
         "base": ("base", 0.85, None), "rho70": ("base", 0.70, None), "K25": ("K25", 0.70, None),
         "M16": ("M16", 0.70, None), "win05": ("win05", 0.85, None)}
PRIMARY = ("rho95_snr20", "rho95_snr10", "rho95_snr5", "rho95_snr0")


def _job(args):
    cond, kind, seed = args
    pc, rho, snr = CONDS[cond]
    p = _params(pc, kind, rho)
    if snr is not None:
        p["obs_snr_db"] = snr
    X, *_ = synth.make_dataset(seed, kind, 100, 1, p)
    te = Mx.s_te(Mx.te_matrix(Mx.analytic(X, synth.FS_OUT)))
    fit = lds.em_fit(X, n_iter=200, tol=1e-6, max_trials=150)
    dn = lds.denoised_stats(fit)["S_inst_dn"]
    return cond, kind, seed, dict(S_TE=te, S_inst_dn=dn, em_iterations=fit["iterations"],
                                  emergent=bool(te > T_TE and dn < T_DN))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = ROOT / "results" / "screen6"; out.mkdir(parents=True, exist_ok=True)
    part = out / "partial.json"
    vals = json.loads(part.read_text()) if part.exists() else {}
    jobs = [(c, k, s) for c in CONDS for k in synth.KINDS for s in SEEDS if str(s) not in vals.get(c, {}).get(k, {})]
    with Pool(a.workers) as p:
        for i, (c, k, s, st) in enumerate(p.imap_unordered(_job, jobs)):
            vals.setdefault(c, {}).setdefault(k, {})[str(s)] = st
            part.write_text(json.dumps(vals))                          # resumable
            if (i + 1) % 15 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    res = {}
    for c in CONDS:
        v = vals[c]
        ne = sum(v["emergent"][str(s)]["emergent"] for s in SEEDS)
        nd = sum(not v["driven"][str(s)]["emergent"] for s in SEEDS)
        ndc = sum(not v["driven_common"][str(s)]["emergent"] for s in SEEDS)
        res[c] = dict(emergent_correct=ne, driven_correct=nd, driven_common_correct=ndc,
                      success=bool(ne >= 9 and nd >= 9 and ndc >= 9),
                      means={k: {st: float(np.mean([v[k][str(s)][st] for s in SEEDS])) for st in ("S_TE", "S_inst_dn")}
                             for k in synth.KINDS})
    others = [c for c in CONDS if c not in PRIMARY]
    verdict = dict(primary_all=all(res[c]["success"] for c in PRIMARY),
                   others_passed=sum(res[c]["success"] for c in others), others_total=len(others))
    verdict["confirmed"] = bool(verdict["primary_all"] and verdict["others_passed"] >= 4)
    (out / "screen6_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #16", seeds=SEEDS,
        thresholds=dict(t_TE=T_TE, t_dn=T_DN), per_dataset=vals, per_condition=res, verdict=verdict), indent=1))
    print("\ncond          E ok   D ok  DC ok  success | S_TE E/D/DC                S_inst_dn E/D/DC")
    for c, r in res.items():
        m = r["means"]
        print(f"{c:12s} {r['emergent_correct']:>3d}/10 {r['driven_correct']:>3d}/10 {r['driven_common_correct']:>3d}/10  {str(r['success']):5s} | "
              f"{m['emergent']['S_TE']:.4f}/{m['driven']['S_TE']:.4f}/{m['driven_common']['S_TE']:.4f}  "
              f"{m['emergent']['S_inst_dn']:.3f}/{m['driven']['S_inst_dn']:.3f}/{m['driven_common']['S_inst_dn']:.3f}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
