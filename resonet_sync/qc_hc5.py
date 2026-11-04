"""hc-5 data-quality report (no hypothesis test; Phase 3 stays stopped).

  python -m resonet_sync.qc_hc5 --session-dir DIR [--out results/qc_hc5]

The report contains:
- Format: .xml vs Par.totNch vs .eeg size, the sampling rates, and the anatomical groups.
- Channels: non-finite values, flat channels, int16-rail clipping, line noise, and theta (6-10 Hz) / delta
  (1-4 Hz) power per channel.
- Behaviour: wheel-run and maze-lap counts and durations, plus speed availability.
- Referencing: mean pairwise theta-band PLV within each anatomical group, for the raw common reference vs local
  bipolar. This is over the whole session, NOT split by wheel and maze, so no H3 contrast is computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from . import data_hc5 as H5
from . import sanitize as S

ROOT = Path(__file__).resolve().parent.parent


def _plv_within(x, fs, groups_cols, band=(6.0, 10.0), max_s=300.0):
    n = min(len(x), int(max_s * fs))
    sos = signal.butter(4, band, btype="band", fs=fs, output="sos")
    out = []
    for cols in groups_cols:
        if len(cols) < 2:
            continue
        a = signal.hilbert(signal.sosfiltfilt(sos, np.asarray(x[:n][:, cols], float), axis=0), axis=0)
        u = a / (np.abs(a) + 1e-12)
        C = np.abs(u.conj().T @ u) / n
        iu = np.triu_indices(len(cols), 1)
        out.append(float(C[iu].mean()))
    return float(np.mean(out)) if out else None


def _laminar_coverage(X, fs, groups, max_s=120.0):
    """Feasibility gate for the generator-mix hypothesis (docs/PREREG_H3_GENERATOR_MIX_DRAFT.md): per anatomical
    group (shank, in xml order = assumed depth order), the ripple-power peak (pyramidal layer) and whether a theta
    phase reversal exists below it. Site spacing is not in the .xml, so the span is reported in sites."""
    from . import laminar as L
    n = min(X.shape[1], int(max_s * fs))
    out = []
    for gi, g in enumerate(groups):
        if len(g) < 3:
            out.append(dict(group=gi, n_sites=len(g), usable=False)); continue
        lay = L.localise_layers(X[g, :n], fs)
        out.append(dict(group=gi, n_sites=len(g), pyramidal_site=lay["pyramidal"], theta_reversal_site=lay["theta_reversal"],
                        sites_below_pyramidal=len(g) - 1 - lay["pyramidal"],
                        theta_phase_deg=[round(v, 1) for v in lay["theta_phase_deg"]],
                        usable=lay["theta_reversal"] is not None))
    return out


def qc_session(session_dir, max_power_s=600.0):
    base = H5.find_base(session_dir)
    name = base.name
    xmlp = H5.parse_xml(str(base) + ".xml")
    eeg = H5.read_eeg(str(base) + ".eeg", xmlp["n_channels"])
    fs = xmlp["lfp_fs"] or 1250.0
    mats = sorted(Path(base).parent.glob("*BehavElectrData*.mat"))
    beh = H5.load_behav(mats[0]) if mats else {}
    rep = dict(session=name, xml=dict((k, v) for k, v in xmlp.items() if k != "groups"),
               n_groups=len(xmlp["groups"]), eeg_shape=list(eeg.shape), eeg_minutes=eeg.shape[0] / fs / 60,
               behav_file=str(mats[0].name) if mats else None)
    rep["format_checks"] = dict(
        xml_vs_Par_totNch=None if not beh else (beh.get("totNch") == xmlp["n_channels"]),
        Par_totNch=beh.get("totNch"), Par_fs=beh.get("fs"),
        lfp_fs_matches_Par=None if not beh or beh.get("fs") is None else abs(beh["fs"] - fs) < 1e-6)
    reg, reg_src = H5.region_map(xmlp["n_channels"])
    rep["regions"] = dict(source=reg_src, counts={r: sum(1 for v in reg.values() if v == r) for r in set(reg.values())})
    n = min(eeg.shape[0], int(max_power_s * fs))
    X = np.asarray(eeg[:n], float).T
    rep["sanitize"] = dict(nonfinite=S.nonfinite_report(X), flat=S.flat_channels(X, fs=fs, max_const_run_s=1.0),
                           clipping=S.clipping_report(X, rail=32767)["flagged"], line_noise=S.line_noise(X, fs))
    f, P = signal.welch(X, fs=fs, nperseg=int(4 * fs), axis=-1)
    th = P[:, (f >= 6) & (f <= 10)].mean(1); de = P[:, (f >= 1) & (f <= 4)].mean(1)
    rep["theta_delta_ratio_per_channel"] = (th / np.maximum(de, 1e-30)).round(3).tolist()
    rep["top_theta_channels"] = np.argsort(-th)[:10].tolist()
    non_lfp = {c for r in ("wheel_ttl", "sync") for c in (H5.DOC_129[r] if xmlp["n_channels"] == 129 else [])}
    groups = [[c for c in g if c < xmlp["n_channels"] and (c + 1) not in non_lfp] for g in xmlp["groups_clean"]]
    rep["skipped_channels"] = xmlp["skipped"]
    xb, pairs = H5.bipolar(eeg[:n], groups)                    # non-overlapping pairs (review board)
    gcols_b = [[k for k, (a, _) in enumerate(pairs) if a in g] for g in groups]
    rep["laminar_coverage"] = _laminar_coverage(X, fs, groups)
    rep["theta_plv_within_group"] = dict(common_reference=_plv_within(eeg, fs, groups),
                                         local_bipolar=_plv_within(xb, fs, gcols_b),
                                         note="first <=300 s of the session; whole-session, not split by behaviour")
    if beh:
        w, mz = beh["wheel_runs"], beh["maze_laps"]
        rep["behaviour"] = dict(
            n_wheel_runs=len(w), wheel_run_s=dict(median=float(np.median([(r['end'] - r['start']) / fs for r in w])) if w else None),
            n_maze_laps=len(mz), maze_lap_s=dict(median=float(np.median([(l['end'] - l['start']) / fs for l in mz])) if mz else None),
            speed_available=beh.get("speed_mm_s") is not None,
            n_windows_1s=dict(wheel=int(sum((r['end'] - r['start']) // int(fs) for r in w)),
                              maze=int(sum((l['end'] - l['start']) // int(fs) for l in mz))))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "qc_hc5"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for d in a.session_dir:
        rep = qc_session(d)
        (out / f"{rep['session']}.json").write_text(json.dumps(rep, indent=1, default=float))
        print(json.dumps({k: rep[k] for k in ("session", "eeg_shape", "format_checks", "regions", "theta_plv_within_group")
                          if k in rep}, default=float, indent=1))
        if "behaviour" in rep:
            print(json.dumps(rep["behaviour"], indent=1, default=float))


if __name__ == "__main__":
    main()
