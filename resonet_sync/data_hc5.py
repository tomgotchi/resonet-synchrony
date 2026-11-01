"""hc-5 (CRCNS, Pastalkova et al. 2008) loader + data-quality checks. NO hypothesis test lives here:
Pitch 1 Phase 3 is stopped (Tom, after Phase 2); this module only verifies formats and extracts epochs.

Per session directory (extracted <name>.tar.gz), using the session's OWN files, never the dataset-level
description (e.g. i01_maze08.001 reports 102 channels while the description says 129):
  parse_xml(<base>.xml)               -> nChannels, rates, anatomical groups (shank order)
  read_eeg(<base>.eeg, nChannels)     -> int16 memmap (n, nCh) @ lfpSamplingRate; size checked
  load_behav(<base>_BehavElectrData.mat)
        Par.totNch / SamplingFrequency  (cross-checked against the .xml and the file size)
        wheel runs  : contiguous runs of Laps.NLapCW > 0 or Laps.NLapCCW > 0 (per-sample run ids @1250 Hz)
        maze laps   : Laps.startLfpInd / Laps.endLfpInd (+ dirChoice, corrChoice)
        speed       : Track.speed_MMsec (mm/s @1250 Hz) if present
  bipolar(x, groups)                  -> local re-reference: each channel minus its next neighbour in the
                                         same anatomical group (a common reference inflates coherence)
Region labels (EC / right CA1 / left CA1) come from an explicit map. The documented 1-based ranges are applied
ONLY when the session really has 129 channels; otherwise regions are 'unknown' until confirmed.
"""
from __future__ import annotations

import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DOC_129 = dict(EC=range(1, 33), CA1_R=range(33, 65), CA1_L=[c for c in range(65, 129) if c not in (105, 106, 107)],
               wheel_ttl=[105, 106], sync=[107])          # 1-based, from crcns-hc5-data-description v0.6


def find_base(session_dir):
    xml = sorted(glob.glob(str(Path(session_dir) / "**" / "*.xml"), recursive=True))
    if not xml:
        raise FileNotFoundError(f"{session_dir}: no .xml")
    return Path(xml[0]).with_suffix("")


def parse_xml(path):
    root = ET.parse(path).getroot()
    acq = root.find("acquisitionSystem")
    out = dict(nbits=int(acq.findtext("nBits")), n_channels=int(acq.findtext("nChannels")),
               fs=float(acq.findtext("samplingRate")))
    lfp = root.findtext("fieldPotentials/lfpSamplingRate")
    out["lfp_fs"] = float(lfp) if lfp else None
    out["groups"] = [[int(c.text) for c in g.findall("channel")]
                     for g in root.findall("anatomicalDescription/channelGroups/group")]
    return out


def read_eeg(path, n_channels):
    size = Path(path).stat().st_size
    if size % (2 * n_channels):
        raise ValueError(f"{path}: {size} bytes is not a whole number of {n_channels}-channel int16 frames")
    return np.memmap(path, dtype="<i2", mode="r", shape=(size // (2 * n_channels), n_channels))


def _get(struct, name):
    """Field of a scipy.io.loadmat struct (1x1 object arrays) -> squeezed ndarray, or None."""
    if struct is None or name not in (struct.dtype.names or ()):
        return None
    v = struct[name]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.item()
    return np.squeeze(np.asarray(v))


def _scalar(v):
    v = np.asarray(v).ravel()
    return None if v.size == 0 else float(v[0])


def runs_from_ids(ids):
    """Per-sample run-id vector (0 = not in a run) -> list of (start, end_exclusive, id)."""
    ids = np.asarray(ids).ravel()
    on = ids > 0
    if not on.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
    out = []
    for s, e in zip(edges[::2], edges[1::2]):
        out.append((int(s), int(e), int(ids[s])))
    return out


def load_behav(mat_path):
    import scipy.io
    m = scipy.io.loadmat(mat_path)
    par = m["Par"][0, 0] if "Par" in m else None
    laps = m["Laps"][0, 0] if "Laps" in m else None
    track = m["Track"][0, 0] if "Track" in m else None
    out = dict(totNch=_scalar(_get(par, "totNch")) if par is not None else None,
               fs=_scalar(_get(par, "SamplingFrequency")) if par is not None else None)
    wheel = []
    for direction in ("CW", "CCW"):
        ids = _get(laps, f"NLap{direction}")
        if ids is not None and ids.size > 1:
            wheel += [dict(start=s, end=e, run_id=r, direction=direction) for s, e, r in runs_from_ids(ids)]
    out["wheel_runs"] = sorted(wheel, key=lambda d: d["start"])
    st, en = _get(laps, "startLfpInd"), _get(laps, "endLfpInd")
    maze = []
    if st is not None and en is not None and np.size(st) and np.size(en):
        st, en = np.atleast_1d(st).astype(int), np.atleast_1d(en).astype(int)
        dirc, corr = _get(laps, "dirChoice"), _get(laps, "corrChoice")
        for i, (s, e) in enumerate(zip(st, en)):
            maze.append(dict(start=int(s), end=int(e),
                             dir_choice=None if dirc is None or np.size(dirc) <= i else float(np.atleast_1d(dirc)[i]),
                             correct=None if corr is None or np.size(corr) <= i else float(np.atleast_1d(corr)[i])))
    out["maze_laps"] = maze
    sp = _get(track, "speed_MMsec")
    out["speed_mm_s"] = None if sp is None or sp.size <= 1 else sp.astype(float)
    return out


def region_map(n_channels):
    """0-based channel -> region. Documented ranges only for the 129-channel layout."""
    if n_channels != 129:
        return {c: "unknown" for c in range(n_channels)}, "unknown: session has %d channels, documented map is for 129" % n_channels
    reg = {}
    for name, chans in DOC_129.items():
        for c in chans:
            reg[c - 1] = name
    return {c: reg.get(c, "unlabelled") for c in range(n_channels)}, "crcns-hc5-data-description v0.6 (129-ch layout)"


def bipolar(x, groups):
    """x: (n, nCh). Each channel minus the next channel in its anatomical group (group order = probe order).
    Returns (y, pairs) with pairs[k] = (ch, ref)."""
    cols, pairs = [], []
    for g in groups:
        for a, b in zip(g[:-1], g[1:]):
            cols.append(np.asarray(x[:, a], float) - np.asarray(x[:, b], float))
            pairs.append((a, b))
    return (np.stack(cols, 1) if cols else np.zeros((len(x), 0))), pairs
