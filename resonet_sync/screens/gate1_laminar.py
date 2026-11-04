"""hc-5 generator-mix plan, Gate 1 (docs/PREREG_H3_GENERATOR_MIX_DRAFT.md): synthetic forward-model null through the
real analysis path. Geometry is a parameter: the real Gate 1 uses the probe geometry reported by Gate 0 (hc-5 QC);
any other geometry is a DRY RUN and is labelled as such.

  python -m resonet_sync.screens.gate1_laminar --n-sites 16 --spacing-um 50 --z0-um 0 [--dry-run]

Per simulated session (10 sessions, seeds 2000-2009), regimes A and B (120 s each, 1250 Hz):
  neutral data = first half of A + first half of B -> layer localisation and sink-site choice (theta-CSD amplitude
  peak between the pyramidal layer and the theta reversal = radiatum; below the reversal = l-m);
  Delta = G(A) - G(B), G = log theta-CSD power ratio l-m / radiatum (second halves only).
Cells: nuisances must NOT move Delta (|mean| < 0.10 and exact sign-flip p >= 0.05); EC drive +20% must be detected
(Delta > 0 in >= 8/10 sessions)."""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from .. import laminar as L
from .screen10 import ROOT, _strict_json

FS = 1250.0
SEEDS = list(range(2000, 2010))
CELLS = {  # name: (A params, B params, must_move)
    "freq_up_1Hz": (dict(f0=9.0), dict(), False),
    "speed_drift": (dict(drift_hz=0.5), dict(), False),
    "asymmetry": (dict(asym=0.4), dict(), False),
    "amplitude_x1.3": (dict(w_ca3=1.3, w_ec=1.3), dict(), False),
    "background_1f": (dict(noise_exp=1.6, noise_rel=0.15), dict(noise_exp=1.0, noise_rel=0.05), False),
    "ec_phase_lag_10deg": (dict(ec_phase_deg=190.0), dict(), False),
    "EC_drive_plus20pct": (dict(w_ec=1.2), dict(), True),
}


def sign_flip_p(d):
    d = np.asarray(d)
    stats = [abs(np.mean(d * np.array(s))) for s in itertools.product((-1, 1), repeat=len(d))]
    return float(np.mean(np.array(stats) >= abs(d.mean()) - 1e-15))


def _session(seed, a, b, geom):
    pa, z = L.simulate_laminar(T_s=120.0, fs=FS, seed=seed, **geom, **a)
    pb, _ = L.simulate_laminar(T_s=120.0, fs=FS, seed=seed + 50_000, **geom, **b)
    h = pa.shape[1] // 2
    neutral = np.concatenate([pa[:, :h], pb[:, :h]], axis=1)
    lay = L.localise_layers(neutral, FS)
    if not L.coverage_ok(lay, geom["spacing"]):
        return dict(usable=False, layers=dict(pyramidal=lay["pyramidal"], reversal=lay["theta_reversal"]))
    amp = np.abs(L.band(L.csd(neutral, geom["spacing"]), FS, 6, 10)).mean(-1)
    i_p, i_r = lay["pyramidal"], lay["theta_reversal"]
    i_rad = i_p + 1 + int(np.argmax(amp[i_p + 1:i_r])) if i_r > i_p + 1 else i_r
    i_lm = i_r + int(np.argmax(amp[i_r:-1])) if i_r < len(amp) - 1 else i_r
    ga = L.generator_mix(pa[:, h:], FS, geom["spacing"], i_rad, i_lm)["log_power_ratio_lm_over_rad"]
    gb = L.generator_mix(pb[:, h:], FS, geom["spacing"], i_rad, i_lm)["log_power_ratio_lm_over_rad"]
    return dict(usable=True, delta=ga - gb, sites=dict(pyramidal=i_p, reversal=i_r, radiatum=i_rad, lm=i_lm))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, required=True)
    ap.add_argument("--spacing-um", type=float, required=True)
    ap.add_argument("--z0-um", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true", help="hypothetical geometry (not the real Gate 1)")
    a = ap.parse_args(argv)
    geom = dict(n_sites=a.n_sites, spacing=a.spacing_um * 1e-6, z0=a.z0_um * 1e-6)
    res, ok = {}, True
    for c, (pa, pb, must_move) in CELLS.items():
        rows = [_session(s, pa, pb, geom) for s in SEEDS]
        if not all(r["usable"] for r in rows):
            res[c] = dict(usable=False, note="geometry fails the coverage gate", rows=rows); ok = False
            continue
        d = np.array([r["delta"] for r in rows])
        p = sign_flip_p(d)
        passed = bool((d > 0).sum() >= 8) if must_move else bool(abs(d.mean()) < 0.10 and p >= 0.05)
        ok &= passed
        res[c] = dict(mean_delta=float(d.mean()), deltas=d.tolist(), sign_flip_p=p, n_positive=int((d > 0).sum()),
                      must_move=must_move, passed=passed, sites=rows[0]["sites"])
        print(f"{c:22s} mean {d.mean():+.3f}  p {p:.3f}  +{(d > 0).sum():2d}/10  {'must move' if must_move else 'null'}"
              f"  {'PASS' if passed else 'FAIL'}", flush=True)
    tag = f"{a.n_sites}x{int(a.spacing_um)}um" + ("_DRYRUN" if a.dry_run else "")
    out_dir = ROOT / "results" / "gate1_laminar"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gate1_{tag}.json").write_text(_strict_json(dict(
        protocol="docs/PREREG_H3_GENERATOR_MIX_DRAFT.md Gate 1", geometry=dict(n_sites=a.n_sites, spacing_um=a.spacing_um,
        z0_um=a.z0_um), dry_run=a.dry_run, seeds=SEEDS, cells=res, gate1_pass=bool(ok)), indent=1))
    print("GATE 1", "PASS" if ok else "FAIL", "(DRY RUN, hypothetical geometry)" if a.dry_run else "")


if __name__ == "__main__":
    main()
