"""DEVIATIONS 14: confirmatory test of the rule  emergent := S_TE > t_TE and S_inst < t_inst  on fresh seeds
30-39 (thresholds frozen from seeds 20-29). python -m resonet_sync.screen4 [--workers 4]."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import mechanism as Mx
from . import synth
from .screen2 import _params

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(30, 40))
T_TE, T_INST = 0.015295, 0.111262                     # frozen in DEVIATIONS 14
CONDS = {"base": 0.85, "rho70": 0.70, "rho95": 0.95, "K25": 0.70, "K60": 0.95, "M16": 0.70, "win05": 0.85,
         "snr0": 0.85}
PRIMARY = ("base", "rho70", "rho95")


def _job(args):
    cond, kind, seed = args
    pcond = cond if cond in ("K25", "K60", "M16", "win05", "snr0") else "base"
    X, *_ = synth.make_dataset(seed, kind, 100, 1, _params(pcond, kind, CONDS[cond]))
    a = Mx.analytic(X, synth.FS_OUT)
    te, inst = Mx.s_te(Mx.te_matrix(a)), Mx.s_inst(Mx.inst_corr(X))
    return cond, kind, seed, dict(S_TE=te, S_inst=inst, emergent=bool(te > T_TE and inst < T_INST))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = [(c, k, s) for c in CONDS for k in synth.KINDS for s in SEEDS]
    vals = {c: {k: {} for k in synth.KINDS} for c in CONDS}
    with Pool(a.workers) as p:
        for i, (c, k, s, st) in enumerate(p.imap_unordered(_job, jobs)):
            vals[c][k][s] = st
            if (i + 1) % 30 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    res = {}
    for c in CONDS:
        ne = sum(vals[c]["emergent"][s]["emergent"] for s in SEEDS)
        nd = sum(not vals[c]["driven"][s]["emergent"] for s in SEEDS)
        ndc = sum(not vals[c]["driven_common"][s]["emergent"] for s in SEEDS)
        res[c] = dict(emergent_correct=ne, driven_correct=nd, driven_common_correct=ndc,
                      success=bool(ne >= 9 and nd >= 9 and ndc >= 9),
                      means={k: {st: float(np.mean([vals[c][k][s][st] for s in SEEDS])) for st in ("S_TE", "S_inst")}
                             for k in synth.KINDS})
    others = [c for c in CONDS if c not in PRIMARY]
    verdict = dict(primary_all=all(res[c]["success"] for c in PRIMARY),
                   others_passed=sum(res[c]["success"] for c in others), others_total=len(others))
    verdict["confirmed"] = bool(verdict["primary_all"] and verdict["others_passed"] >= 4)
    out = ROOT / "results" / "screen4"; out.mkdir(parents=True, exist_ok=True)
    (out / "screen4_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #14", seeds=SEEDS,
        thresholds=dict(t_TE=T_TE, t_inst=T_INST), per_dataset={c: {k: {str(s): v for s, v in d.items()}
        for k, d in vals[c].items()} for c in CONDS}, per_condition=res, verdict=verdict), indent=1))
    print("\ncond    E correct  D correct  DC correct  success | mean S_TE E/D/DC          mean S_inst E/D/DC")
    for c, r in res.items():
        m = r["means"]
        print(f"{c:6s}  {r['emergent_correct']:>5d}/10  {r['driven_correct']:>5d}/10  {r['driven_common_correct']:>6d}/10   {str(r['success']):5s} | "
              f"{m['emergent']['S_TE']:.4f}/{m['driven']['S_TE']:.4f}/{m['driven_common']['S_TE']:.4f}  "
              f"{m['emergent']['S_inst']:.3f}/{m['driven']['S_inst']:.3f}/{m['driven_common']['S_inst']:.3f}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
