"""DEVIATIONS 15: operating range of the DEVIATIONS 14 rule under strong common input (rho 0.95) x observation SNR.
python -m resonet_sync.screens.screen5 [--workers 4]."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import mechanism as Mx
from .. import synth
from .screen4 import T_INST, T_TE

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(40, 50))
SNRS = (20.0, 10.0, 5.0, 0.0)
RHO = 0.95


def _job(args):
    snr, kind, seed = args
    p = {"obs_snr_db": snr}
    if kind == "driven_common":
        p["rho"] = RHO
    X, *_ = synth.make_dataset(seed, kind, 100, 1, p)
    te, inst = Mx.s_te(Mx.te_matrix(Mx.analytic(X, synth.FS_OUT))), Mx.s_inst(Mx.inst_corr(X))
    return snr, kind, seed, dict(S_TE=te, S_inst=inst, emergent=bool(te > T_TE and inst < T_INST))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = [(s, k, sd) for s in SNRS for k in synth.KINDS for sd in SEEDS]
    vals = {s: {k: {} for k in synth.KINDS} for s in SNRS}
    with Pool(a.workers) as p:
        for snr, k, sd, st in p.imap_unordered(_job, jobs):
            vals[snr][k][sd] = st
    res = {}
    for s in SNRS:
        ne = sum(vals[s]["emergent"][x]["emergent"] for x in SEEDS)
        nd = sum(not vals[s]["driven"][x]["emergent"] for x in SEEDS)
        ndc = sum(not vals[s]["driven_common"][x]["emergent"] for x in SEEDS)
        res[str(s)] = dict(emergent_correct=ne, driven_correct=nd, driven_common_correct=ndc,
                           success=bool(ne >= 9 and nd >= 9 and ndc >= 9),
                           means={k: {st: float(np.mean([vals[s][k][x][st] for x in SEEDS])) for st in ("S_TE", "S_inst")}
                                  for k in synth.KINDS})
    ok = [s for s in SNRS if res[str(s)]["success"]]
    verdict = dict(lowest_successful_snr_db=min(ok) if ok else None, succeeded=[str(s) for s in ok])
    out = ROOT / "results" / "screen5"; out.mkdir(parents=True, exist_ok=True)
    (out / "screen5_results.json").write_text(json.dumps(dict(protocol="DEVIATIONS.md #15", seeds=SEEDS, rho=RHO,
        per_dataset={str(s): {k: {str(x): v for x, v in d.items()} for k, d in vals[s].items()} for s in SNRS},
        per_condition=res, verdict=verdict), indent=1))
    print("SNR dB  E correct  D correct  DC correct  success | S_TE E/D/DC                S_inst E/D/DC")
    for s in SNRS:
        r = res[str(s)]; m = r["means"]
        print(f"{s:6.0f}  {r['emergent_correct']:>5d}/10  {r['driven_correct']:>5d}/10  {r['driven_common_correct']:>6d}/10   {str(r['success']):5s} | "
              f"{m['emergent']['S_TE']:.4f}/{m['driven']['S_TE']:.4f}/{m['driven_common']['S_TE']:.4f}  "
              f"{m['emergent']['S_inst']:.3f}/{m['driven']['S_inst']:.3f}/{m['driven_common']['S_inst']:.3f}")
    print("\nVERDICT:", json.dumps(verdict))


if __name__ == "__main__":
    main()
