"""DEVIATIONS 18: within-session rule v2 (scale-free relative guard) on fresh sessions 70-79 at 0 dB.
python -m resonet_sync.screen8 [--workers 4]."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import lds
from . import mechanism as Mx
from . import synth
from .screen import boot_ci

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = list(range(70, 80))
EPS = 1e-9
REL_MARGIN = 0.25
SNR = 0.0
BASE_SCEN = {"S1_coupling_up": ((40.0, 0.5), (25.0, 0.5), True),
             "S2_common_up_trap": ((25.0, 0.9), (25.0, 0.5), False),
             "S3_null": ((25.0, 0.5), (25.0, 0.5), False),
             "S5_coupling_vs_common": ((40.0, EPS), (EPS, 0.9), True)}
CELLS = {f"{s}|{v}": (BASE_SCEN[s], vp) for s in BASE_SCEN for v, vp in (("M16", {"M": 16}), ("win05", {"win_s": 0.5}))}
CELLS["S2b_mild_trap|base"] = (((25.0, 0.7), (25.0, 0.5), False), {})


def _job(args):
    cell, seed, reg = args
    (a, b, _), vp = CELLS[cell]
    K, rho = a if reg == "A" else b
    p = dict(vp, K=K, rho=rho, obs_snr_db=SNR, stream=1 if reg == "B" else 0)
    X, *_ = synth.make_dataset(seed, "mixed", 100, 1, p)
    te = Mx.s_te(Mx.te_matrix(Mx.analytic(X, synth.FS_OUT)))
    dn = lds.denoised_stats(lds.em_fit(X, n_iter=200, tol=1e-6, max_trials=150))["S_inst_dn"]
    return cell, seed, reg, dict(S_TE=te, S_inst_dn=dn)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = ROOT / "results" / "screen8"; out.mkdir(parents=True, exist_ok=True)
    part = out / "partial.json"
    vals = json.loads(part.read_text()) if part.exists() else {}
    k = lambda c, s, r: f"{c}|{s}|{r}"
    jobs = [(c, s, r) for c in CELLS for s in SESSIONS for r in "AB" if k(c, s, r) not in vals]
    with Pool(a.workers) as p:
        for i, (c, s, r, st) in enumerate(p.imap_unordered(_job, jobs)):
            vals[k(c, s, r)] = st
            part.write_text(json.dumps(vals))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    res, ok = {}, True
    for c, ((_, _, should), _) in CELLS.items():
        dte = np.array([vals[k(c, s, "A")]["S_TE"] - vals[k(c, s, "B")]["S_TE"] for s in SESSIONS])
        rel = np.array([(vals[k(c, s, "A")]["S_inst_dn"] - vals[k(c, s, "B")]["S_inst_dn"]) / vals[k(c, s, "B")]["S_inst_dn"]
                        for s in SESSIONS])
        ci_te, ci_r = boot_ci(dte), boot_ci(rel)
        fires = bool(np.sum(dte > 0) >= 9 and ci_te[0] > 0 and ci_r[1] < REL_MARGIN)
        ok &= fires == should
        res[c] = dict(dTE=float(dte.mean()), dTE_ci=ci_te, dTE_pos=int(np.sum(dte > 0)), rel=float(rel.mean()), rel_ci=ci_r,
                      fires=fires, should_fire=should, correct=fires == should)
    verdict = dict(all_cells_correct=bool(ok), false_positives=[c for c, r in res.items() if r["fires"] and not r["should_fire"]],
                   misses=[c for c, r in res.items() if not r["fires"] and r["should_fire"]])
    (out / "screen8_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #18", sessions=SESSIONS, rel_margin=REL_MARGIN,
                                                              snr_db=SNR, per_dataset=vals, per_cell=res, verdict=verdict), indent=1))
    print("\ncell                               dTE [CI] (n>0)                rel dInst [CI]          fires should correct")
    for c, r in res.items():
        print(f"{c:34s} {r['dTE']:+.4f} [{r['dTE_ci'][0]:+.4f},{r['dTE_ci'][1]:+.4f}] ({r['dTE_pos']})  "
              f"{r['rel']:+.3f} [{r['rel_ci'][0]:+.3f},{r['rel_ci'][1]:+.3f}]  {str(r['fires']):5s} {str(r['should_fire']):5s} {r['correct']}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
