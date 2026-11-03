"""DEVIATIONS 11 mechanism-statistic screen: python -m resonet_sync.screens.screen1 [--workers 4]
Exploratory, synthetic only. Success criterion frozen in DEVIATIONS.md before this ran."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import mechanism as Mx
from .. import synth
from ..stats import boot_ci  # noqa: F401  (moved to stats.py)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(10))
STATS = ("S_couple", "S_TE", "S_asym", "S_resid", "S_plv")
CANDIDATES = ("S_couple", "S_TE", "S_asym", "S_resid")


def _job(args):
    kind, seed = args
    X, *_ = synth.make_dataset(seed, kind, 100, 1)        # train trials only (identical to Phase 2's)
    return kind, seed, Mx.all_stats(X, synth.FS_OUT)


def criterion(vals):
    """vals[kind][seed] -> dict. Applies DEVIATIONS 11 (a)-(c) for each statistic."""
    out = {}
    for st in STATS:
        E = np.array([vals["emergent"][s][st] for s in SEEDS])
        res = {}
        for other in ("driven", "driven_common"):
            d = E - np.array([vals[other][s][st] for s in SEEDS])
            sign = np.sign(d.mean())
            ci = boot_ci(d)
            res[other] = dict(mean_diff=float(d.mean()), ci95=ci, same_sign=int(np.sum(np.sign(d) == sign)),
                              sign=int(sign), ci_excludes_0=bool(ci[0] > 0 or ci[1] < 0),
                              ok=bool(np.sum(np.sign(d) == sign) >= 9 and (ci[0] > 0 or ci[1] < 0)))
        res["same_direction"] = res["driven"]["sign"] == res["driven_common"]["sign"]
        res["PASS"] = bool(res["driven"]["ok"] and res["driven_common"]["ok"] and res["same_direction"])
        res["means"] = {k: float(np.mean([vals[k][s][st] for s in SEEDS])) for k in synth.KINDS}
        res["candidate"] = st in CANDIDATES
        out[st] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = [(k, s) for s in SEEDS for k in synth.KINDS]
    vals = {k: {} for k in synth.KINDS}
    with Pool(a.workers) as p:
        for kind, seed, st in p.imap_unordered(_job, jobs):
            vals[kind][seed] = st
            print(f"  {kind:14s} seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in st.items()), flush=True)
    crit = criterion(vals)
    out_dir = ROOT / "results" / "screen1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screen_results.json").write_text(json.dumps(dict(
        protocol="DEVIATIONS.md #11", seeds=SEEDS, params={k: synth.DEFAULTS[k] for k in synth.KINDS},
        per_dataset={k: {str(s): v for s, v in d.items()} for k, d in vals.items()}, criterion=crit), indent=1))
    print("\nstatistic   E / D / DC means            E-D [CI] (n same sign)        E-DC [CI] (n same sign)   PASS")
    for st, r in crit.items():
        m = r["means"]; d, c = r["driven"], r["driven_common"]
        print(f"{st:9s}{'' if r['candidate'] else '*'} {m['emergent']:.4f}/{m['driven']:.4f}/{m['driven_common']:.4f}  "
              f"{d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] ({d['same_sign']}/10)  "
              f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] ({c['same_sign']}/10)  {r['PASS']}")
    print("* reference only (not a candidate)")


if __name__ == "__main__":
    main()
