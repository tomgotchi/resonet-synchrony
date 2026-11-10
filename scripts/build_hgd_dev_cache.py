"""Build a TRAIN-ONLY development cache of the High-Gamma Dataset for scripts/v2_hgd_dev.py.

Same preprocessing as the published full-band cache (reso-net/build_hg_cache.py: 128 EEG channels, zero-phase FIR
4-125 Hz, epochs [0, 4.5) s at 500 Hz, classes feet / left_hand / rest / right_hand -> 0..3, float16), applied to
the TRAINING recording only. The test recording is not downloaded or read. The npz carries a 4-trial all-zero
placeholder under X_te / y_te only so that the published loader runs; it must never be used as test data.

  python scripts/build_hgd_dev_cache.py --raw /path/to/hgd_raw --out /path/to/hgd_dev_cache --subjects 1-6
Raw files: <raw>/train/<subject>.edf from https://gin.g-node.org/robintibor/high-gamma-dataset (data are never
committed)."""
import argparse
from pathlib import Path

import mne
import numpy as np

L_FREQ, H_FREQ = 4.0, 125.0
TMIN, N_SAMP = 0.0, 2250
CLASS_ORDER = ["feet", "left_hand", "rest", "right_hand"]


def epochs_from_edf(path):
    """Verbatim logic of reso-net/build_hg_cache.py::epochs_from_edf."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eeg = [c for c in raw.ch_names if c.startswith("EEG ")]
    assert len(eeg) == 128, f"{path}: expected 128 EEG ch, got {len(eeg)}"
    raw.pick(eeg)
    raw.filter(L_FREQ, H_FREQ, method="fir", phase="zero", verbose="ERROR")
    events, evid = mne.events_from_annotations(raw, verbose="ERROR")
    want = {evid[name]: i for i, name in enumerate(CLASS_ORDER) if name in evid}
    sf = raw.info["sfreq"]
    tmax = TMIN + (N_SAMP - 1) / sf
    ep = mne.Epochs(raw, events, event_id={n: evid[n] for n in CLASS_ORDER if n in evid},
                    tmin=TMIN, tmax=tmax, baseline=None, preload=True, verbose="ERROR")
    X = ep.get_data(copy=False).astype(np.float32)[:, :, :N_SAMP]
    y = np.array([want[c] for c in ep.events[:, 2]], dtype=np.int64)
    return X, y, float(sf)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default="1-6")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    for s in range(int(lo), int(hi) + 1):
        f = out / f"subj_{s}.npz"
        if f.exists():
            continue
        X, y, sf = epochs_from_edf(str(Path(a.raw) / "train" / f"{s}.edf"))
        assert sf == 500.0 and X.shape[1:] == (128, N_SAMP), (sf, X.shape)
        np.savez_compressed(f, X_tr=X.astype(np.float16), y_tr=y,
                            X_te=np.zeros((4, 128, N_SAMP), np.float16), y_te=np.arange(4),
                            dev_train_only=True)
        print(f"S{s}: train {X.shape} classes {np.bincount(y).tolist()} -> {f}", flush=True)


if __name__ == "__main__":
    main()
