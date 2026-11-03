"""DEVIATIONS 20: rule v3 (cycle-normalised TE + LDS noise-fraction matching).
python -m resonet_sync.screen10 --stage D|C [--workers 4]."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import lds
from . import mechanism as Mx
from . import nuisance as N
from . import synth
from .screen import boot_ci

ROOT = Path(__file__).resolve().parent.parent
REL_MARGIN = 0.25
B0 = dict(K=25.0, rho=0.5, obs_snr_db=10.0)
P = lambda **kw: dict(B0, **kw)
CELLS_D = {"F1_freq_up": (P(f_shift_hz=1.0), P(), False),
           "F2_freq_down": (P(f_shift_hz=-1.0), P(), False),
           "N1_snr_up": (P(), P(obs_snr_db=0.0), False),
           "N2_snr_down": (P(obs_snr_db=0.0), P(), False),
           "C1_coupling_up": (P(K=40.0), P(), True)}
CELLS_C = dict(CELLS_D,
               X1_both_nuisances_up=(P(f_shift_hz=1.0), P(obs_snr_db=0.0), False),
               T1_mild_common_trap=(P(rho=0.7), P(), False),
               C2_coupling_up_A_noisier=(P(K=40.0, obs_snr_db=0.0), P(), True),
               C3_coupling_up_A_slower=(P(K=40.0, f_shift_hz=-1.0), P(), True))
STAGES = {"D": (list(range(80, 90)), CELLS_D), "C": (list(range(90, 100)), CELLS_C)}
EM = dict(n_iter=200, tol=1e-6, max_trials=150)


def _job(args):
    stage, cell, seed = args
    a, b, _ = STAGES[stage][1][cell]
    X = {r: synth.make_dataset(seed, "mixed", 100, 1, dict(p, stream=i))[0] for i, (r, p) in enumerate((("A", a), ("B", b)))}
    out = {}
    fits = {r: lds.em_fit(X[r], **EM) for r in "AB"}
    tgt = np.maximum(fits["A"]["R"], fits["B"]["R"])
    for i, r in enumerate("AB"):
        st = dict(R_mean=float(fits[r]["R"].mean()),
                  S_TE_v2=Mx.s_te(Mx.te_matrix(Mx.analytic(X[r]))),
                  S_inst_dn_v2=lds.denoised_stats(fits[r])["S_inst_dn"])
        Y, v = N.degrade_to(X[r], fits[r]["R"], tgt, seed=1000 * seed + i)
        fit_m = lds.em_fit(Y, **EM)
        te, f_c = N.s_te_cyc(Y)
        st.update(S_TE_al=N.s_te_aligned(Y)[0], f_c=f_c, added_noise_var=float(v.mean()), R_mean_matched=float(fit_m["R"].mean()),
                  S_TE_cyc=te, S_inst_dn=lds.denoised_stats(fit_m)["S_inst_dn"])
        out[r] = st
    return cell, seed, out


def decide(vals, cells, sessions, te_key="S_TE_cyc", inst_key="S_inst_dn"):
    res = {}
    for c, (_, _, should) in cells.items():
        A = [vals[f"{c}|{s}"]["A"] for s in sessions]; B = [vals[f"{c}|{s}"]["B"] for s in sessions]
        dte = np.array([x[te_key] - y[te_key] for x, y in zip(A, B)])
        rel = np.array([(x[inst_key] - y[inst_key]) / y[inst_key] for x, y in zip(A, B)])
        ci_te, ci_r = boot_ci(dte), boot_ci(rel)
        fires = bool(np.sum(dte > 0) >= 9 and ci_te[0] > 0 and ci_r[1] < REL_MARGIN)
        res[c] = dict(dTE=float(dte.mean()), dTE_ci=ci_te, dTE_pos=int(np.sum(dte > 0)), rel=float(rel.mean()),
                      rel_ci=ci_r, fires=fires, should_fire=should, correct=fires == should,
                      d_fc=float(np.mean([x["f_c"] - y["f_c"] for x, y in zip(A, B)])),
                      d_R=float(np.mean([x["R_mean"] - y["R_mean"] for x, y in zip(A, B)])))
    verdict = dict(all_cells_correct=all(r["correct"] for r in res.values()),
                   false_positives=[c for c, r in res.items() if r["fires"] and not r["should_fire"]],
                   misses=[c for c, r in res.items() if not r["fires"] and r["should_fire"]])
    return res, verdict


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices="DC", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    sessions, cells = STAGES[a.stage]
    out = ROOT / "results" / "screen10" / f"stage_{a.stage}"; out.mkdir(parents=True, exist_ok=True)
    part = out / "partial.json"
    vals = json.loads(part.read_text()) if part.exists() else {}
    jobs = [(a.stage, c, s) for c in cells for s in sessions if f"{c}|{s}" not in vals]
    with Pool(a.workers) as p:
        for i, (c, s, st) in enumerate(p.imap_unordered(_job, jobs)):
            vals[f"{c}|{s}"] = st
            part.write_text(json.dumps(vals))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    te_key = "S_TE_al" if a.stage == "C" else "S_TE_cyc"     # DEVIATIONS 20 revision: stage C uses rule v3r
    res, verdict = decide(vals, cells, sessions, te_key)
    res_v2, verdict_v2 = decide(vals, cells, sessions, "S_TE_v2", "S_inst_dn_v2")
    (out / "results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #20", stage=a.stage, te_statistic=te_key, sessions=sessions,
                                                      per_dataset=vals, per_cell=res, verdict=verdict,
                                                      unmatched_v2=dict(per_cell=res_v2, verdict=verdict_v2)), indent=1))
    print(f"\nstage {a.stage}   cell                       dTE_cyc [CI] (n>0)             rel dInst [CI]       dfc   dR    fires should | v2 fires")
    for c, r in res.items():
        print(f"{c:30s} {r['dTE']:+.4f} [{r['dTE_ci'][0]:+.4f},{r['dTE_ci'][1]:+.4f}] ({r['dTE_pos']:2d})  "
              f"{r['rel']:+.3f} [{r['rel_ci'][0]:+.3f},{r['rel_ci'][1]:+.3f}]  {r['d_fc']:+.2f} {r['d_R']:+.3f}  "
              f"{str(r['fires']):5s} {str(r['should_fire']):5s} | {res_v2[c]['fires']}")
    print("\nVERDICT:", json.dumps(verdict))
    print("VERDICT_v2_unmatched:", json.dumps(verdict_v2))


if __name__ == "__main__":
    main()
