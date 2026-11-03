"""DEVIATIONS 19: nuisance-confound traps (frequency, SNR) for rule v2 on fresh sessions 80-89.
python -m resonet_sync.screens.screen9 [--workers 4]."""
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
SESSIONS = list(range(80, 90))
REL_MARGIN = 0.25
# cell -> ((A params, B params, should_fire)). Mechanism K25 rho0.5 in both unless stated.
BASE = dict(K=25.0, rho=0.5)
CELLS = {"F1_freq_up": (dict(BASE, f_shift_hz=1.0, obs_snr_db=10.0), dict(BASE, obs_snr_db=10.0), False),
         "F2_freq_down": (dict(BASE, f_shift_hz=-1.0, obs_snr_db=10.0), dict(BASE, obs_snr_db=10.0), False),
         "N1_snr_up": (dict(BASE, obs_snr_db=10.0), dict(BASE, obs_snr_db=0.0), False),
         "N2_snr_down": (dict(BASE, obs_snr_db=0.0), dict(BASE, obs_snr_db=10.0), False),
         "C1_coupling_up_control": (dict(BASE, K=40.0, obs_snr_db=10.0), dict(BASE, obs_snr_db=10.0), True)}


def _job(args):
    cell, seed, reg = args
    a, b, _ = CELLS[cell]
    p = dict(a if reg == "A" else b, stream=1 if reg == "B" else 0)
    X, *_ = synth.make_dataset(seed, "mixed", 100, 1, p)
    te = Mx.s_te(Mx.te_matrix(Mx.analytic(X, synth.FS_OUT)))
    dn = lds.denoised_stats(lds.em_fit(X, n_iter=200, tol=1e-6, max_trials=150))["S_inst_dn"]
    return cell, seed, reg, dict(S_TE=te, S_inst_dn=dn)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out = ROOT / "results" / "screen9"; out.mkdir(parents=True, exist_ok=True)
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
    for c, (_, _, should) in CELLS.items():
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
    (out / "screen9_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #19", sessions=SESSIONS, rel_margin=REL_MARGIN, per_dataset=vals, per_cell=res, verdict=verdict), indent=1))
    print("\ncell                               dTE [CI] (n>0)                rel dInst [CI]          fires should correct")
    for c, r in res.items():
        print(f"{c:34s} {r['dTE']:+.4f} [{r['dTE_ci'][0]:+.4f},{r['dTE_ci'][1]:+.4f}] ({r['dTE_pos']})  "
              f"{r['rel']:+.3f} [{r['rel_ci'][0]:+.3f},{r['rel_ci'][1]:+.3f}]  {str(r['fires']):5s} {str(r['should_fire']):5s} {r['correct']}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
