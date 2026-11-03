"""Development diagnostic 2 for the DEVIATIONS 20 revision (used sessions 80-89 only).
Hypothesis: a uniform frequency shift is a rotating-frame change (identical physics in real time), so the delay should
stay fixed in TIME; the residual bias is plug-in-TE bin alignment (mean phase advance per delay = 2.05 bins at 8 Hz,
fractional at 7/9 Hz). Remedies: rotate the future phase by -2 pi f_c d (removes the mean advance) and/or average
TE over shifted bin grids (dither)."""
import json
import sys
from multiprocessing import Pool

import numpy as np

from . import mechanism as Mx, nuisance as N, synth
from .screen10 import CELLS_D

D = 8.0
VARIANTS = {  # (band half-width or None, rotate, n_dither)
    "fixband_fix": (None, False, 1), "fixband_fix_rot": (None, True, 1), "fixband_fix_rot_dith4": (None, True, 4),
    "relband_fix_rot": (2.0, True, 1), "relband_fix_rot_dith4": (2.0, True, 4), "fixband_fix_dith4": (None, False, 4)}


def te_rot(a, d, f_c, rotate, n_dith, fs=250.0, bins=Mx.TE_BINS):
    phi = np.angle(a); di = int(d)
    past, fut = phi[..., :-di], phi[..., di:]
    if rotate:
        fut = fut - 2 * np.pi * f_c * d / fs
    M = phi.shape[1]; TE = np.zeros((M, M))
    for o in range(n_dith):
        off = 2 * np.pi * o / (bins * n_dith)
        q = lambda p: np.clip((((np.angle(np.exp(1j * (p + off))) + np.pi) / (2 * np.pi)) * bins).astype(np.int64),
                              0, bins - 1)
        P, F = q(past), q(fut)
        for i in range(M):
            xf, xp = F[:, i].ravel(), P[:, i].ravel()
            h_fp, h_p = Mx._H(xf, xp, bins=bins), Mx._H(xp, bins=bins)
            for j in range(M):
                if j != i:
                    yp = P[:, j].ravel()
                    TE[j, i] += (h_fp + Mx._H(xp, yp, bins=bins) - h_p - Mx._H(xf, xp, yp, bins=bins)) / n_dith
    return Mx.s_te(TE)


def job(args):
    cell, seed = args
    a, b, _ = CELLS_D[cell]
    out = {}
    for i, (r, p) in enumerate((("A", a), ("B", b))):
        X = synth.make_dataset(seed, "mixed", 100, 1, dict(p, stream=i))[0]
        f_c = N.peak_freq(X)
        res = {"f_c": f_c}
        for v, (hw, rot, nd) in VARIANTS.items():
            band = Mx.BAND if hw is None else (f_c - hw, f_c + hw)
            res[v] = te_rot(Mx.analytic(X, 250.0, band), D, f_c, rot, nd)
        out[r] = res
    return cell, seed, out


if __name__ == "__main__":
    cells = ("F1_freq_up", "F2_freq_down", "C1_coupling_up")
    jobs = [(c, s) for c in cells for s in range(80, 90)]
    with Pool(4) as p:
        res = list(p.imap_unordered(job, jobs))
    json.dump([list(r) for r in res], open(sys.argv[1], "w"))
    for c in cells:
        rows = [o for cc, s, o in res if cc == c]
        for v in VARIANTS:
            d = np.array([o["A"][v] - o["B"][v] for o in rows])
            print(f"{c:16s} {v:24s} {d.mean():+.4f}  ({(d > 0).sum():2d}/10)", flush=True)
