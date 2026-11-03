"""DEVIATIONS 17: within-session contrast rule on fresh session seeds 60-69.
python -m resonet_sync.screens.screen7 [--workers 4]. Exploratory, synthetic; protocol frozen before this ran."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import lds
from .. import mechanism as Mx
from .. import synth
from ..stats import boot_ci

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = list(range(60, 70))
EPS = 1e-9
SCEN = {"S1_coupling_up": ((40.0, 0.5), (25.0, 0.5), True),
        "S2_common_up_trap": ((25.0, 0.9), (25.0, 0.5), False),
        "S3_null": ((25.0, 0.5), (25.0, 0.5), False),
        "S5_coupling_vs_common": ((40.0, EPS), (EPS, 0.9), True)}
SNRS = (20.0, 0.0)
MARGIN = 0.03


def _job(args):
    scen, snr, seed, reg = args
    K, rho = SCEN[scen][0 if reg == "A" else 1]
    X, *_ = synth.make_dataset(seed, "mixed", 100, 1, dict(K=K, rho=rho, obs_snr_db=snr, stream=1 if reg == "B" else 0))
    te = Mx.s_te(Mx.te_matrix(Mx.analytic(X, synth.FS_OUT)))
    dn = lds.denoised_stats(lds.em_fit(X, n_iter=200, tol=1e-6, max_trials=150))["S_inst_dn"]
    return scen, snr, seed, reg, dict(S_TE=te, S_inst_dn=dn)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = ROOT / "results" / "screen7"; out.mkdir(parents=True, exist_ok=True)
    part = out / "partial.json"
    vals = json.loads(part.read_text()) if part.exists() else {}
    key = lambda sc, sn, sd, rg: f"{sc}|{sn}|{sd}|{rg}"
    jobs = [(sc, sn, sd, rg) for sc in SCEN for sn in SNRS for sd in SESSIONS for rg in "AB" if key(sc, sn, sd, rg) not in vals]
    with Pool(a.workers) as p:
        for i, (sc, sn, sd, rg, st) in enumerate(p.imap_unordered(_job, jobs)):
            vals[key(sc, sn, sd, rg)] = st
            part.write_text(json.dumps(vals))
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    res, ok_all = {}, True
    for sc, (_, _, should) in SCEN.items():
        for sn in SNRS:
            dte = np.array([vals[key(sc, sn, s, "A")]["S_TE"] - vals[key(sc, sn, s, "B")]["S_TE"] for s in SESSIONS])
            ddn = np.array([vals[key(sc, sn, s, "A")]["S_inst_dn"] - vals[key(sc, sn, s, "B")]["S_inst_dn"] for s in SESSIONS])
            ci_te, ci_dn = boot_ci(dte), boot_ci(ddn)
            fires = bool(np.sum(dte > 0) >= 9 and ci_te[0] > 0 and ci_dn[1] < MARGIN)
            correct = fires == should
            ok_all &= correct
            res[f"{sc}@{sn:g}dB"] = dict(dTE=float(dte.mean()), dTE_ci=ci_te, dTE_pos=int(np.sum(dte > 0)),
                                         dDN=float(ddn.mean()), dDN_ci=ci_dn, fires=fires, should_fire=should, correct=correct)
    verdict = dict(all_cells_correct=bool(ok_all), false_positives=[k for k, r in res.items() if r["fires"] and not r["should_fire"]],
                   misses=[k for k, r in res.items() if not r["fires"] and r["should_fire"]])
    (out / "screen7_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #17", sessions=SESSIONS, margin=MARGIN,
                                                              per_dataset=vals, per_cell=res, verdict=verdict), indent=1))
    print("\ncell                          dTE [CI] (n>0)                 dInst_dn [CI]              fires should correct")
    for k, r in res.items():
        print(f"{k:28s} {r['dTE']:+.4f} [{r['dTE_ci'][0]:+.4f},{r['dTE_ci'][1]:+.4f}] ({r['dTE_pos']})  "
              f"{r['dDN']:+.3f} [{r['dDN_ci'][0]:+.3f},{r['dDN_ci'][1]:+.3f}]  {str(r['fires']):5s} {str(r['should_fire']):5s} {r['correct']}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
