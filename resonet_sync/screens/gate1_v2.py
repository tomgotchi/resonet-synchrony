"""Gate 1 v2 for the hc-5 generator-mix plan (review board 2026-11-09; docs/PREREG_H3_GENERATOR_MIX_DRAFT.md).

  python -m resonet_sync.screens.gate1_v2 --n-sites 16 --spacing-um 50 --top-um -150 --seeds dev   [--dry-run]
  python -m resonet_sync.screens.gate1_v2 --passmap                                              # geometry grid

Forward model: laminar.simulate_laminar_v2 (disc sources, shared septal drift, CA3-EC coherence, perisomatic / DG
theta, absolute noise, per-session probe offset U(0, spacing), layer thickness x U(0.85, 1.15), site gains).
Statistic: Delta = G(A) - G(B), G = log power ratio EC / CA3 of the two generators fitted to the neutral theta
CSD (first halves of A and B), projected on each regime's second half (laminar.generator_mix_fit).
Criteria (set on dev seeds 2100-2109, frozen before the validation seeds 2200-2209):
  - must-not-move cells: 90% t-CI of Delta inside +/- 1/3 of the EC+20% effect (Halvorsen);
  - must-move cell: Delta > 0 in >= 8/10 sessions AND resampled power >= 0.8 for the exact sign-flip test at
    n = 7 (the hc-5 number of sessions);
  - identification (null, coherence, lag and target cells): both fitted sinks within one site spacing of the
    truth in >= 9/10 sessions;
  - depth-shift QC (applied to every real session): generators fitted separately on each regime's first half;
    a sink-depth difference > one site spacing flags the session. Reported: flag rate in every cell (false flags
    in null cells; detection in the radiatum-shift cell, which is a LIMITATION cell, not gating: the neutral fit
    cannot absorb a generator that moves between regimes)."""
from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool

import numpy as np

from .. import laminar as L
from ..stats import sign_flip_p, t_ci
from .screen10 import ROOT, _strict_json

FS = 1250.0
SEEDS = dict(dev=list(range(2100, 2110)), val=list(range(2200, 2210)))
TARGET = "EC_drive_plus20pct"
BASE = dict(plv=0.95)
CELLS = {   # name: (A params, B params, role)
    "null": (dict(), dict(), "null"),
    "coherence_plv_0.75": (dict(plv=0.75), dict(), "null"),
    "ec_lag_200deg": (dict(ec_phase_deg=200.0), dict(), "null"),
    "ec_lag_225deg": (dict(ec_phase_deg=225.0), dict(), "null"),
    "radiatum_sink_50um_proximal": (dict(rad_shift=50e-6), dict(), "limitation"),
    "perisomatic_theta_x1.3": (dict(w_peri=0.65), dict(w_peri=0.5), "null"),
    "all_theta_x1.3_abs_noise": (dict(w_ca3=1.3, w_ec=1.3), dict(), "null"),
    "dg_theta_x1.3": (dict(w_dg=0.65), dict(w_dg=0.5), "null"),
    "speed_f0+1Hz_shared_drift": (dict(f0=9.0, drift_hz=0.5), dict(), "null"),
    "background_1f_1.6_noise_15pct": (dict(noise_exp=1.6, noise_rel=0.15), dict(noise_exp=1.0, noise_rel=0.05), "null"),
    "ec_asymmetry_0.4_fundamental_fixed": (dict(asym_ec=0.4), dict(), "null"),
    TARGET: (dict(w_ec=1.2), dict(), "target"),
    "EC_drive_plus10pct": (dict(w_ec=1.1), dict(), "curve"),
    "EC_drive_plus30pct": (dict(w_ec=1.3), dict(), "curve"),
    "CA3_drive_minus20pct": (dict(w_ca3=0.8), dict(), "curve"),
}


ID_CELLS = {"null", "coherence_plv_0.75", "ec_lag_200deg", "ec_lag_225deg", TARGET}


def session(job):
    cell, seed, geom = job
    a, b, _ = CELLS[cell]
    rng = np.random.default_rng([seed, 7])                       # probe offset / thickness: per session, not per cell
    jit = dict(top=geom["top"] + rng.uniform(0, geom["spacing"]), thick=rng.uniform(0.85, 1.15))
    kw = dict(n_sites=geom["n_sites"], spacing=geom["spacing"], gains=1 + 0.03 * rng.standard_normal(geom["n_sites"]), **jit)
    pa, z, tr = L.simulate_laminar_v2(fs=FS, seed=seed, return_truth=True, **kw, **dict(BASE, **a))
    pb, _ = L.simulate_laminar_v2(fs=FS, seed=seed + 50_000, **kw, **dict(BASE, **b))
    h = pa.shape[1] // 2
    neutral = np.concatenate([pa[:, :h], pb[:, :h]], axis=1)
    lay = L.localise_layers(neutral, FS)
    r = L.generator_mix_fit(neutral, [pa[:, h:], pb[:, h:]], FS, geom["spacing"])
    true = [tr["ca3"] - z[0], tr["ec"] - z[0]]
    err = [abs(f - t) for f, t in zip(r["sinks_rel_first_site"], true)]
    in_span = all(0 <= t <= z[-1] - z[0] for t in true)
    zz = np.arange(pa.shape[0]) * geom["spacing"]                   # depth-shift QC: fit each regime separately
    da = L.fit_generators_refined(L.theta_csd_cov(pa[:, :h], FS, geom["spacing"]), zz, geom["spacing"])[1][:, 0]
    db = L.fit_generators_refined(L.theta_csd_cov(pb[:, :h], FS, geom["spacing"]), zz, geom["spacing"])[1][:, 0]
    qc_flag = bool(np.max(np.abs(da - db)) > geom["spacing"] + 1e-9)
    return dict(cell=cell, seed=seed, delta=r["G"][0] - r["G"][1], fit_sinks_um=[round(v * 1e6) for v in r["sinks_rel_first_site"]],
                true_sinks_um=[round(v * 1e6) for v in true], id_ok=bool(in_span and max(err) <= geom["spacing"] + 1e-9),
                sinks_in_span=in_span, qc_depth_shift_flag=qc_flag,
                pyramidal_site=lay["pyramidal"], pyramidal_true_site=int(np.argmin(np.abs(z - L.PYR_DEPTH))),
                jitter=dict(top_um=round(jit["top"] * 1e6, 1), thick=round(jit["thick"], 3)))


def resampled_power(d, n=7, n_res=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([sign_flip_p(rng.choice(d, n, replace=True)) < alpha for _ in range(n_res)]))


def evaluate(rows):
    by = {c: [r for r in rows if r["cell"] == c] for c in CELLS}
    tgt = np.array([r["delta"] for r in by[TARGET]])
    bound = abs(tgt.mean()) / 3
    out, ok = {}, True
    for c, rs in by.items():
        if not rs:
            continue
        d = np.array([r["delta"] for r in rs])
        role = CELLS[c][2]
        ci = t_ci(d, 0.90)
        idr = float(np.mean([r["id_ok"] for r in rs]))
        need_id = c in ID_CELLS
        rec = dict(role=role, mean=float(d.mean()), sd=float(d.std(ddof=1)), ci90=ci, n_pos=int((d > 0).sum()),
                   p_signflip=sign_flip_p(d), id_rate=idr, id_gating=need_id, deltas=d.round(4).tolist(),
                   qc_flag_rate=float(np.mean([r["qc_depth_shift_flag"] for r in rs])),
                   fit_sinks_um=[r["fit_sinks_um"] for r in rs], true_sinks_um=[r["true_sinks_um"] for r in rs])
        if role == "null":
            rec["passed"] = bool(-bound < ci[0] and ci[1] < bound and (idr >= 0.9 or not need_id))
        elif role == "target":
            rec["power_n7"] = resampled_power(d)
            rec["passed"] = bool((d > 0).sum() >= 8 and rec["power_n7"] >= 0.8 and idr >= 0.9)
        if role in ("null", "target"):
            ok &= rec["passed"]
        out[c] = rec
    return dict(null_bound=bound, target_mean=float(tgt.mean()), cells=out, gate1_pass=bool(ok))


def run_geometry(geom, seeds, workers):
    jobs = [(c, s, geom) for c in CELLS for s in seeds]
    with Pool(workers) as p:
        rows = p.map(session, jobs)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, default=16)
    ap.add_argument("--spacing-um", type=float, default=50.0)
    ap.add_argument("--top-um", type=float, default=-150.0, help="first site relative to the pyramidal layer")
    ap.add_argument("--seeds", choices=list(SEEDS), default="dev")
    ap.add_argument("--dry-run", action="store_true", help="hypothetical geometry (not the real Gate 1)")
    ap.add_argument("--passmap", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    out_dir = ROOT / "results" / "gate1_v2"; out_dir.mkdir(parents=True, exist_ok=True)
    if a.passmap:
        grid = list(itertools.product((20.0, 50.0, 100.0), (8, 16, 32), (-150.0, -50.0, 50.0)))
        res = []
        for sp, ns, top in grid:
            geom = dict(n_sites=ns, spacing=sp * 1e-6, top=top * 1e-6)
            ev = evaluate(run_geometry(geom, SEEDS[a.seeds], a.workers))
            span_below = (ns - 1) * sp + top
            res.append(dict(spacing_um=sp, n_sites=ns, top_um=top, span_below_pyramidal_um=span_below,
                            gate1_pass=ev["gate1_pass"], target_mean=ev["target_mean"],
                            failed=[c for c, v in ev["cells"].items() if v.get("passed") is False],
                            id_rate_min=min(v["id_rate"] for v in ev["cells"].values())))
            print(res[-1], flush=True)
        (out_dir / f"passmap_{a.seeds}.json").write_text(_strict_json(dict(seeds=SEEDS[a.seeds], grid=res), indent=1))
        return
    geom = dict(n_sites=a.n_sites, spacing=a.spacing_um * 1e-6, top=a.top_um * 1e-6)
    rows = run_geometry(geom, SEEDS[a.seeds], a.workers)
    ev = evaluate(rows)
    for c, v in ev["cells"].items():
        extra = f" power_n7 {v['power_n7']:.2f}" if "power_n7" in v else ""
        print(f"{c:36s} {v['role']:6s} mean {v['mean']:+.3f} CI90[{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}] "
              f"+{v['n_pos']}/10 id {v['id_rate']:.1f} qc {v['qc_flag_rate']:.1f}{extra} "
              f"{'PASS' if v.get('passed') else ('FAIL' if 'passed' in v else '')}")
    print(f"null bound +/-{ev['null_bound']:.3f}; GATE 1 v2", "PASS" if ev["gate1_pass"] else "FAIL",
          "(DRY RUN)" if a.dry_run else "")
    tag = f"{a.n_sites}x{int(a.spacing_um)}um_top{int(a.top_um)}_{a.seeds}" + ("_DRYRUN" if a.dry_run else "")
    (out_dir / f"gate1v2_{tag}.json").write_text(_strict_json(dict(
        protocol="docs/PREREG_H3_GENERATOR_MIX_DRAFT.md Gate 1 v2 (board 2026-11-09)", geometry=geom, seeds=SEEDS[a.seeds],
        dry_run=a.dry_run, **ev, rows=rows), indent=1))


if __name__ == "__main__":
    main()
