"""DEVIATIONS 21: screen 11 = terminal calibration of rule v3r (review board rounds 1-2). Synthetic only.

  python -m resonet_sync.screens.screen11 --phase calib   # pilot sessions 990-994: rhythmic-drive levels (PLV-matched)
  python -m resonet_sync.screens.screen11 --phase T       # tier T: TE leg, every cell, sessions 100-129
  python -m resonet_sync.screens.screen11 --phase F       # tier F: full v3r, pre-specified cells, sessions 100-109
  python -m resonet_sync.screens.screen11 --phase report  # P(fire) estimates, pre-committed decisions

All regimes use the exact (Lawson / exact-OU) integrator. Run with OMP_NUM_THREADS=1 (one BLAS thread per worker)."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

from .. import lds
from .. import mechanism as Mx
from .. import nuisance as N
from .. import synth
from .screen10 import ROOT, _strict_json

OUT = ROOT / "results" / "screen11"
REL_MARGIN = 0.25
EM = dict(n_iter=200, tol=1e-6, max_trials=150)
BASE = dict(K=25.0, rho=0.5, obs_snr_db=10.0, integrator="exact")
P = lambda **kw: dict(BASE, **kw)
CALIB_SESSIONS = list(range(990, 995))
KAPPA_GRID = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
T_SESSIONS = list(range(100, 130))
F_SESSIONS = list(range(100, 110))
PINK = dict(obs_noise_exp=1.5, obs_noise_corr=0.3)


def cells(levels):
    L1, L2, L3 = levels
    c = {  # name: (A params, B params, should_fire)
        "N0_null": (P(), P(), False),
        "F1_freq_up": (P(f_shift_hz=1.0), P(), False),
        "F2_freq_down": (P(f_shift_hz=-1.0), P(), False),
        "R1_rhythm_low": (P(kappa_nb=L1), P(), False),
        "R2_rhythm_mid": (P(kappa_nb=L2), P(), False),
        "R3_rhythm_high": (P(kappa_nb=L3), P(), False),
        "W055_white_common": (P(rho=0.55), P(), False),
        "W060_white_common": (P(rho=0.60), P(), False),
        "W065_white_common": (P(rho=0.65), P(), False),
        "D075_self_damping": (P(extra_self_damp_K=7.5), P(), False),
        "D150_self_damping": (P(extra_self_damp_K=15.0), P(), False),
        "VC_volume_conduction": (P(vc_spread=1.0), P(), False),
        "Q1_pink_corr_noise_snr": (P(**PINK), P(obs_snr_db=0.0, **PINK), False),
        "K30_coupling": (P(K=30.0), P(), True),
        "K35_coupling": (P(K=35.0), P(), True),
        "K40_coupling": (P(K=40.0), P(), True),
        "K35R_coupling_with_rhythm": (P(K=35.0, kappa_nb=L2), P(kappa_nb=L2), True),
    }
    return c


FORCED_F = ["Q1_pink_corr_noise_snr", "K35_coupling", "K35R_coupling_with_rhythm"]
RETIRE_FAMILY = ("R", "W")          # rhythmic and white-common-input cells (DEVIATIONS 20 pre-commitment)


def _data(seed, p, stream, n=100):
    return synth.make_dataset(seed, "mixed", n, 1, dict(p, stream=stream))[0]


def _key(p):
    return json.dumps(p, sort_keys=True)


# ---------------------------------------------------------------- calibration
def _calib_job(args):
    seed, kind, val = args
    p = P(K=val) if kind == "K" else P(kappa_nb=val)
    X = _data(seed, p, 0 if (kind, val) != ("K", 25.0) else 1, n=50)
    return seed, kind, val, Mx.s_plv(Mx.analytic(X))


def phase_calib(workers):
    jobs = [(s, "K", v) for s in CALIB_SESSIONS for v in (25.0, 40.0)] + \
           [(s, "kappa", v) for s in CALIB_SESSIONS for v in KAPPA_GRID]
    with Pool(workers) as pool:
        res = list(pool.imap_unordered(_calib_job, jobs))
    plv = {}
    for s, kind, val, v in res:
        plv[(s, kind, val)] = v
    base = np.array([plv[(s, "K", 25.0)] for s in CALIB_SESSIONS])
    d_ref = float(np.mean(np.array([plv[(s, "K", 40.0)] for s in CALIB_SESSIONS]) - base))
    d_k = [float(np.mean(np.array([plv[(s, "kappa", k)] for s in CALIB_SESSIONS]) - base)) for k in KAPPA_GRID]
    levels, flags = [], []
    for mult in (0.5, 1.0, 2.0):
        tgt = mult * d_ref
        if tgt > max(d_k):
            levels.append(KAPPA_GRID[int(np.argmax(d_k))]); flags.append(f"x{mult}: target not reached, grid max used")
        else:
            levels.append(float(np.interp(tgt, d_k, KAPPA_GRID)) if np.all(np.diff(d_k) > 0)
                          else float(KAPPA_GRID[int(np.argmin(np.abs(np.array(d_k) - tgt)))]))
    out = dict(protocol="DEVIATIONS.md #21 calibration", sessions=CALIB_SESSIONS, dPLV_K25_to_K40=d_ref,
               kappa_grid=KAPPA_GRID, dPLV_kappa=d_k, levels=levels, flags=flags)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(_strict_json(out, indent=1))
    print(json.dumps(out, indent=1))


def _levels():
    return json.loads((OUT / "calibration.json").read_text())["levels"]


# ---------------------------------------------------------------- tier T (TE leg, unmatched, every cell)
def _t_job(args):
    seed, levels = args
    C = cells(levels)
    specs = {}
    for a, b, _ in C.values():
        specs[(_key(a), 0)] = a
        specs[(_key(b), 1)] = b
    out = {}
    for (k, stream), p in specs.items():
        X = _data(seed, p, stream)
        te, f_c = N.s_te_aligned(X)
        out[f"{stream}|{k}"] = dict(S_TE_al=te, f_c=f_c, PLV=Mx.s_plv(Mx.analytic(X)))
    return seed, out


def phase_T(workers):
    levels = _levels()
    part = OUT / "tierT_partial.json"
    vals = json.loads(part.read_text()) if part.exists() else {}
    jobs = [(s, levels) for s in T_SESSIONS if str(s) not in vals]
    with Pool(workers) as pool:
        for i, (s, o) in enumerate(pool.imap_unordered(_t_job, jobs)):
            vals[str(s)] = o
            part.write_text(json.dumps(vals))
            print(f"  T {i + 1}/{len(jobs)}", flush=True)
    (OUT / "tierT.json").write_text(_strict_json(dict(levels=levels, sessions=T_SESSIONS, per_session=vals), indent=1))


def tier_T_table(levels, vals):
    C = cells(levels)
    tab = {}
    for c, (a, b, should) in C.items():
        dte = np.array([vals[str(s)][f"0|{_key(a)}"]["S_TE_al"] - vals[str(s)][f"1|{_key(b)}"]["S_TE_al"] for s in T_SESSIONS])
        dplv = np.array([vals[str(s)][f"0|{_key(a)}"]["PLV"] - vals[str(s)][f"1|{_key(b)}"]["PLV"] for s in T_SESSIONS])
        tab[c] = dict(dTE=dte.tolist(), dPLV=dplv.tolist(), should_fire=should)
    return tab


# ---------------------------------------------------------------- P(fire) estimators
_IDX = np.random.default_rng(0).integers(0, 10, (10000, 10))          # == boot_ci(d) indices at n = 10, seed 0


def _te_leg_fires(D):
    """D: (S, 10) per-subset session differences -> bool (S,): >= 9/10 positive and boot_ci lower > 0."""
    pos = (D > 0).sum(1) >= 9
    low = np.empty(len(D))
    for i in range(0, len(D), 250):
        m = D[i:i + 250][:, _IDX].mean(2)                               # (s, 10000)
        low[i:i + 250] = np.percentile(m, 2.5, axis=1)
    return pos & (low > 0)


def p_fire_te_leg(dte, n_sub=10000, seed=1):
    """Tier T: probability that the TE leg fires on a random 10 of the available sessions (without replacement)."""
    dte = np.asarray(dte); rng = np.random.default_rng(seed)
    D = np.stack([dte[rng.choice(len(dte), 10, replace=False)] for _ in range(n_sub)])
    return float(_te_leg_fires(D).mean())


def p_fire_rule(dte, rel, n_boot=2000, seed=2):
    """Tier F: smoothed bootstrap of per-session (dTE, rel) pairs; full rule v3r on each resampled set of 10."""
    dte, rel = np.asarray(dte), np.asarray(rel); n = len(dte); rng = np.random.default_rng(seed)
    h = 1.06 * n ** (-1 / 5)
    D = np.empty((n_boot, 10)); R = np.empty((n_boot, 10))
    for b in range(n_boot):
        i = rng.integers(0, n, 10)
        D[b] = dte[i] + h * dte.std(ddof=1) * rng.standard_normal(10)
        R[b] = rel[i] + h * rel.std(ddof=1) * rng.standard_normal(10)
    te = _te_leg_fires(D)
    up = np.empty(n_boot)
    for i in range(0, n_boot, 250):
        up[i:i + 250] = np.percentile(R[i:i + 250][:, _IDX].mean(2), 97.5, axis=1)
    return float((te & (up < REL_MARGIN)).mean())


def f_cells(levels, tabT):
    """Pre-specified: forced cells + every must-not-fire cell whose tier-T TE-leg P(fire) > 0.05."""
    sel = list(FORCED_F)
    for c, r in tabT.items():
        if not r["should_fire"] and c not in sel and p_fire_te_leg(r["dTE"]) > 0.05:
            sel.append(c)
    return sel


# ---------------------------------------------------------------- tier F (full v3r)
def _f_job(args):
    cell, seed, levels = args
    a, b, _ = cells(levels)[cell]
    X = {"A": _data(seed, a, 0), "B": _data(seed, b, 1)}
    fits = {r: lds.em_fit(X[r], **EM) for r in "AB"}
    tgt = np.maximum(fits["A"]["R"], fits["B"]["R"])
    out = {}
    for i, r in enumerate("AB"):
        Y, v = N.degrade_to(X[r], fits[r]["R"], tgt, seed=1000 * seed + i)
        fit_m = lds.em_fit(Y, **EM)
        te, f_c = N.s_te_aligned(Y)
        dn = lds.denoised_stats(fit_m)
        out[r] = dict(S_TE_al=te, f_c=f_c, S_inst_dn=dn["S_inst_dn"], unstable=bool(dn.get("unstable", False)),
                      R_mean=float(fits[r]["R"].mean()), added_noise_var=float(v.mean()))
    return cell, seed, out


def phase_F(workers):
    levels = _levels()
    tabT = tier_T_table(levels, json.loads((OUT / "tierT.json").read_text())["per_session"])
    sel = f_cells(levels, tabT)
    print("tier F cells (pre-specified selection):", sel, flush=True)
    part = OUT / "tierF_partial.json"
    vals = json.loads(part.read_text()) if part.exists() else {}
    jobs = [(c, s, levels) for c in sel for s in F_SESSIONS if f"{c}|{s}" not in vals]
    with Pool(workers) as pool:
        for i, (c, s, o) in enumerate(pool.imap_unordered(_f_job, jobs)):
            vals[f"{c}|{s}"] = o
            part.write_text(json.dumps(vals))
            print(f"  F {i + 1}/{len(jobs)}", flush=True)
    (OUT / "tierF.json").write_text(_strict_json(dict(levels=levels, cells=sel, sessions=F_SESSIONS, per_dataset=vals),
                                                 indent=1))


# ---------------------------------------------------------------- report (pre-committed decisions)
def phase_report():
    T = json.loads((OUT / "tierT.json").read_text()); levels = T["levels"]
    tabT = tier_T_table(levels, T["per_session"])
    Fp = OUT / "tierF.json"
    F = json.loads(Fp.read_text()) if Fp.exists() else None
    rows = {}
    for c, r in tabT.items():
        row = dict(should_fire=r["should_fire"], dTE_mean=float(np.mean(r["dTE"])), dPLV_mean=float(np.mean(r["dPLV"])),
                   p_te_leg=p_fire_te_leg(r["dTE"]))
        if F and c in F["cells"]:
            A = [F["per_dataset"][f"{c}|{s}"]["A"] for s in F_SESSIONS]
            B = [F["per_dataset"][f"{c}|{s}"]["B"] for s in F_SESSIONS]
            dte = np.array([x["S_TE_al"] - y["S_TE_al"] for x, y in zip(A, B)])
            rel = np.array([(x["S_inst_dn"] - y["S_inst_dn"]) / y["S_inst_dn"] for x, y in zip(A, B)])
            row["tier_F_finite"] = bool(np.isfinite(dte).all() and np.isfinite(rel).all())
            row["p_rule"] = p_fire_rule(dte, rel) if row["tier_F_finite"] else None
            row["rel_mean"] = float(np.mean(rel))
        row["p_fire"] = row.get("p_rule", row["p_te_leg"])     # tier F if run, else the TE-leg upper bound
        rows[c] = row
    nulls = {c: r["p_fire"] for c, r in rows.items() if not r["should_fire"]}
    retire = [c for c, p in nulls.items() if c.startswith(RETIRE_FAMILY) and (p is None or p > 0.05)]
    power = rows["K35_coupling"].get("p_rule")
    verdict = dict(effective_alpha=max(v for v in nulls.values() if v is not None),
                   must_not_fire_over_005=[c for c, p in nulls.items() if p is None or p > 0.05],
                   power_K35=power, power_ok=power is not None and power >= 0.8,
                   calibration_pass=all(p is not None and p <= 0.05 for p in nulls.values())
                   and power is not None and power >= 0.8,
                   retire_TE_LDS_rule=bool(retire), retire_because=retire)
    out = dict(protocol="DEVIATIONS.md #21", levels=levels, cells=rows, verdict=verdict)
    (OUT / "report.json").write_text(_strict_json(out, indent=1))
    print(f"{'cell':28s} should  dPLV    dTE      P(TE leg)  P(rule)")
    for c, r in rows.items():
        pr = r.get("p_rule")
        print(f"{c:28s} {str(r['should_fire']):6s} {r['dPLV_mean']:+.3f} {r['dTE_mean']:+.5f}  {r['p_te_leg']:.3f}     "
              f"{'-' if pr is None else f'{pr:.3f}'}")
    print("\nVERDICT:", json.dumps(verdict))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calib", "T", "F", "report"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"calib": lambda: phase_calib(a.workers), "T": lambda: phase_T(a.workers), "F": lambda: phase_F(a.workers),
     "report": phase_report}[a.phase]()


if __name__ == "__main__":
    main()
