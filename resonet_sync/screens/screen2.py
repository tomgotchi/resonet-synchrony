"""DEVIATIONS 12: TE re-validation on fresh seeds 10-19, base + 10 one-at-a-time variants.
python -m resonet_sync.screens.screen2 [--workers 4]. Exploratory, synthetic only; protocol frozen before this ran."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import mechanism as Mx
from .. import synth
from ..stats import boot_ci

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(10, 20))
CALIB_SEED, RHO_GRID = 999, (0.3, 0.5, 0.7, 0.85, 0.95)
# id -> (params applied to every generator, params for emergent only, rho rule)
CONDITIONS = {
    "base":  ({}, {}, 0.85),
    "K25":   ({}, {"K": 25.0}, "recal"),
    "K60":   ({}, {"K": 60.0}, "recal"),
    "rho70": ({}, {}, 0.70),
    "rho95": ({}, {}, 0.95),
    "M4":    ({"M": 4}, {}, "recal"),
    "M16":   ({"M": 16}, {}, "recal"),
    "win05": ({"win_s": 0.5}, {}, "recal"),
    "win2":  ({"win_s": 2.0}, {}, "recal"),
    "snr10": ({"obs_snr_db": 10.0}, {}, "recal"),
    "snr0":  ({"obs_snr_db": 0.0}, {}, "recal"),
}


def _params(cond, kind, rho):
    common, emerg, _ = CONDITIONS[cond]
    p = dict(common)
    if kind == "emergent":
        p.update(emerg)
    if kind == "driven_common":
        p["rho"] = rho
    return p


def _stats(job):
    cond, kind, seed, rho, what = job
    X, *_ = synth.make_dataset(seed, kind, 100, 1, _params(cond, kind, rho))
    a = Mx.analytic(X, synth.FS_OUT)
    if what == "plv":
        return cond, kind, seed, rho, dict(S_plv=Mx.s_plv(a))
    TE = Mx.te_matrix(a)
    TEx = Mx.te_matrix_cross_trial(a)
    return cond, kind, seed, rho, dict(S_TE=Mx.s_te(TE), S_TE_cross=Mx.s_te(TEx), S_TE_excess=Mx.s_te(TE) - Mx.s_te(TEx),
                                       S_plv=Mx.s_plv(a))


def criterion(vals, st):
    E = np.array([vals["emergent"][s][st] for s in SEEDS])
    res = {}
    for other in ("driven", "driven_common"):
        d = E - np.array([vals[other][s][st] for s in SEEDS])
        sign = np.sign(d.mean()); ci = boot_ci(d)
        n_same = int(np.sum(np.sign(d) == sign))
        res[other] = dict(mean_diff=float(d.mean()), ci95=ci, same_sign=n_same, sign=int(sign),
                          ok=bool(n_same >= 9 and (ci[0] > 0 or ci[1] < 0)))
    res["PASS"] = bool(res["driven"]["ok"] and res["driven_common"]["ok"]
                       and res["driven"]["sign"] == res["driven_common"]["sign"])
    res["means"] = {k: float(np.mean([vals[k][s][st] for s in SEEDS])) for k in synth.KINDS}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out_dir = ROOT / "results" / "screen2"; out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        # 1) recalibrate rho on seed 999 where required (frozen rule)
        recal = [c for c, v in CONDITIONS.items() if v[2] == "recal"]
        jobs = [(c, "emergent", CALIB_SEED, None, "plv") for c in recal] + \
               [(c, "driven_common", CALIB_SEED, r, "plv") for c in recal for r in RHO_GRID]
        cal = {}
        for c, kind, _, r, st in pool.imap_unordered(_stats, jobs):
            cal.setdefault(c, {})[("E" if kind == "emergent" else r)] = st["S_plv"]
        rho = {c: (v[2] if v[2] != "recal" else min(RHO_GRID, key=lambda r: abs(cal[c][r] - cal[c]["E"])))
               for c, v in CONDITIONS.items()}
        for c in recal:
            print(f"  calib {c}: E PLV {cal[c]['E']:.3f} -> rho {rho[c]} (PLV {cal[c][rho[c]]:.3f})", flush=True)
        # 2) evaluation seeds
        jobs = [(c, k, s, rho[c], "te") for c in CONDITIONS for k in synth.KINDS for s in SEEDS]
        vals = {c: {k: {} for k in synth.KINDS} for c in CONDITIONS}
        for i, (c, k, s, _, st) in enumerate(pool.imap_unordered(_stats, jobs)):
            vals[c][k][s] = st
            if (i + 1) % 30 == 0:
                print(f"  {i + 1}/{len(jobs)} datasets", flush=True)
    crit = {c: {st: criterion(vals[c], st) for st in ("S_TE", "S_TE_excess", "S_plv")} for c in CONDITIONS}
    variants = [c for c in CONDITIONS if c != "base"]
    n_pass = sum(crit[c]["S_TE"]["PASS"] for c in variants)
    verdict = dict(primary_base_pass=crit["base"]["S_TE"]["PASS"], variants_passed=n_pass, variants_total=len(variants),
                   robust=bool(crit["base"]["S_TE"]["PASS"] and n_pass >= 8),
                   failing_variants=[c for c in variants if not crit[c]["S_TE"]["PASS"]])
    (out_dir / "screen2_results.json").write_text(json.dumps(dict(
        protocol="DEVIATIONS.md #12", seeds=SEEDS, rho=rho, calibration={c: {str(k): v for k, v in d.items()} for c, d in cal.items()},
        per_dataset={c: {k: {str(s): v for s, v in d.items()} for k, d in vals[c].items()} for c in CONDITIONS},
        criterion=crit, verdict=verdict), indent=1))
    print("\ncond    rho   S_TE E/D/DC means           E-D [CI] (n)            E-DC [CI] (n)          PASS | excess E/D/DC         PASS")
    for c in CONDITIONS:
        r, x = crit[c]["S_TE"], crit[c]["S_TE_excess"]
        m, mx = r["means"], x["means"]
        print(f"{c:6s} {rho[c]:4.2f}  {m['emergent']:.4f}/{m['driven']:.4f}/{m['driven_common']:.4f}  "
              f"{r['driven']['mean_diff']:+.4f} [{r['driven']['ci95'][0]:+.4f},{r['driven']['ci95'][1]:+.4f}] ({r['driven']['same_sign']})  "
              f"{r['driven_common']['mean_diff']:+.4f} [{r['driven_common']['ci95'][0]:+.4f},{r['driven_common']['ci95'][1]:+.4f}] ({r['driven_common']['same_sign']})  "
              f"{str(r['PASS']):5s} | {mx['emergent']:.4f}/{mx['driven']:.4f}/{mx['driven_common']:.4f} {x['PASS']}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
