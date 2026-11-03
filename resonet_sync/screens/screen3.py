"""DEVIATIONS 13: conditional Granger (lagged) vs instantaneous dependence; fresh seeds 20-29.
python -m resonet_sync.screens.screen3 [--workers 4]. Exploratory, synthetic only; protocol frozen before this ran."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import mechanism as Mx
from .. import synth
from .screen2 import _params
from ..stats import boot_ci

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(20, 30))
CONDS = {"base": 0.85, "rho70": 0.70, "rho95": 0.95}
STATS = ("S_GC", "S_inst", "S_TE")


def _job(args):
    cond, kind, seed = args
    rho = CONDS[cond]
    X, *_ = synth.make_dataset(seed, kind, 100, 1, _params("base", kind, rho))
    GC, C = Mx.gc_inst(X)
    a = Mx.analytic(X, synth.FS_OUT)
    return cond, kind, seed, dict(S_GC=Mx.s_gc(GC), S_inst=Mx.s_inst(C), S_TE=Mx.s_te(Mx.te_matrix(a)))


def criterion(v, st):
    E = np.array([v["emergent"][s][st] for s in SEEDS])
    r = {}
    for o in ("driven", "driven_common"):
        d = E - np.array([v[o][s][st] for s in SEEDS])
        sg = np.sign(d.mean()); ci = boot_ci(d); n = int(np.sum(np.sign(d) == sg))
        r[o] = dict(mean_diff=float(d.mean()), ci95=ci, same_sign=n, sign=int(sg), ok=bool(n >= 9 and (ci[0] > 0 or ci[1] < 0)))
    r["PASS"] = bool(r["driven"]["ok"] and r["driven_common"]["ok"] and r["driven"]["sign"] == r["driven_common"]["sign"])
    r["means"] = {k: float(np.mean([v[k][s][st] for s in SEEDS])) for k in synth.KINDS}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = [(c, k, s) for c in CONDS for k in synth.KINDS for s in SEEDS]
    vals = {c: {k: {} for k in synth.KINDS} for c in CONDS}
    with Pool(a.workers) as p:
        for i, (c, k, s, st) in enumerate(p.imap_unordered(_job, jobs)):
            vals[c][k][s] = st
            if (i + 1) % 15 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    crit = {c: {st: criterion(vals[c], st) for st in STATS} for c in CONDS}
    verdict = dict(S_GC_success=all(crit[c]["S_GC"]["PASS"] for c in CONDS),
                   per_condition={c: {st: crit[c][st]["PASS"] for st in STATS} for c in CONDS})
    out = ROOT / "results" / "screen3"; out.mkdir(parents=True, exist_ok=True)
    (out / "screen3_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #13", seeds=SEEDS, rho=CONDS,
        per_dataset={c: {k: {str(s): x for s, x in d.items()} for k, d in vals[c].items()} for c in CONDS},
        criterion=crit, verdict=verdict), indent=1))
    print("\ncond   stat     E / D / DC means              E-D [CI] (n)                 E-DC [CI] (n)               PASS")
    for c in CONDS:
        for st in STATS:
            r = crit[c][st]; m = r["means"]; d, x = r["driven"], r["driven_common"]
            print(f"{c:6s} {st:7s} {m['emergent']:.4f}/{m['driven']:.4f}/{m['driven_common']:.4f}  "
                  f"{d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] ({d['same_sign']})  "
                  f"{x['mean_diff']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] ({x['same_sign']})  {r['PASS']}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
