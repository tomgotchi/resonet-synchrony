"""Development diagnostic for DEVIATIONS 20 revision (used sessions 80-89 only): which band/delay choice removes the
frequency confound? TE only (no noise matching needed: dR ~ 0 in F cells)."""
import json, sys
from multiprocessing import Pool
import numpy as np
from .. import mechanism as Mx, nuisance as N, synth
from .screen10 import CELLS_D

VARIANTS = {"fixband_fixdelay": (None, None), "fixband_cycdelay": (None, "cyc"),
            "relband_fixdelay": (2.0, None), "relband_cycdelay": (2.0, "cyc"),
            "relband3_cycdelay": (3.0, "cyc")}


def te(X, f_c, hw, dmode):
    band = Mx.BAND if hw is None else (f_c - hw, f_c + hw)
    a = Mx.analytic(X, 250.0, band)
    d = 8.0 if dmode is None else N.DELAY_CYC * 250.0 / f_c
    return Mx.s_te(N.te_matrix_frac(a, d))


def job(args):
    cell, seed = args
    a, b, _ = CELLS_D[cell]
    out = {}
    for i, (r, p) in enumerate((("A", a), ("B", b))):
        X = synth.make_dataset(seed, "mixed", 100, 1, dict(p, stream=i))[0]
        f_c = N.peak_freq(X)
        out[r] = {v: te(X, f_c, *spec) for v, spec in VARIANTS.items()} | {"f_c": f_c}
    return cell, seed, out


if __name__ == "__main__":
    jobs = [(c, s) for c in ("F1_freq_up", "F2_freq_down", "C1_coupling_up") for s in range(80, 90)]
    with Pool(4) as p:
        res = list(p.imap_unordered(job, jobs))
    json.dump([list(r) for r in res], open(sys.argv[1], "w"))
    for c in ("F1_freq_up", "F2_freq_down", "C1_coupling_up"):
        rows = [o for cc, s, o in res if cc == c]
        for v in VARIANTS:
            d = np.array([o["A"][v] - o["B"][v] for o in rows])
            print(f"{c:16s} {v:20s} {d.mean():+.4f}  ({(d > 0).sum():2d}/10)")
