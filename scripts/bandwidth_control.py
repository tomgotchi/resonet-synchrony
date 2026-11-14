"""The control experiment the manuscript is missing (review board 2026-11-13, decision 1).

The paper compares a "restricted" run (cache low-passed near 48 Hz, decim 4 -> 125 Hz) with a "full" run (4-125 Hz,
decim 2 -> 250 Hz) and attributes the gain to high gamma. Band and sampling rate changed together, and the
published oscillator frequencies depend on the sampling rate. This script fills the two missing cells with the
PUBLISHED model, loader, loaders, training loop and evaluation (vendor/resonance_network.py, as run_ablation.py):

                         125 Hz (decim 4)              250 Hz (decim 2)
  band to about 48 Hz    published "restricted"        NEW: --tag lp48_decim2
  band to 125 Hz         NEW: --tag full_decim4        published "full" (results_hg)

  # 1. low-passed copy of the full-band cache (CPU, about 1 min per subject)
  python scripts/bandwidth_control.py --make-lowpass --src <subj_cache_hg> --dst <subj_cache_lp48>
  # 2. the two new cells (GPU; resumable per subject and arm)
  python scripts/bandwidth_control.py --run --cache <subj_cache_lp48> --decim 2 --tag lp48_decim2
  python scripts/bandwidth_control.py --run --cache <subj_cache_hg>   --decim 4 --tag full_decim4

Writes results/bandwidth_control/<tag>/subj_XX.json and summary.json (plain test accuracy, as published).
This is the authors' own evaluation protocol on their own test sets; it involves no v2 model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARMS = ("nl_sync", "nl_power", "lin_sync", "lin_power", "eegnet", "shallow")


def make_lowpass(src, dst, cutoff=48.0, fs=500.0, chunk=64):
    from scipy import signal
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    h = signal.firwin(825, cutoff, fs=fs)                           # zero-phase by forward-backward filtering
    for f in sorted(Path(src).glob("subj_*.npz")):
        if (dst / f.name).exists():
            continue
        with np.load(f) as d:
            out = {k: d[k] for k in d.files if k not in ("X_tr", "X_te")}
            for k in ("X_tr", "X_te"):
                X = d[k]
                Y = np.empty_like(X)
                for i in range(0, len(X), chunk):
                    x = np.pad(X[i:i + chunk].astype(np.float32), ((0, 0), (0, 0), (900, 900)), mode="reflect")
                    Y[i:i + chunk] = signal.filtfilt(h, [1.0], x, axis=-1)[..., 900:-900].astype(X.dtype)
                out[k] = Y
        np.savez_compressed(dst / f.name, **out)
        print("low-passed", f.name, flush=True)


def run(cache, decim, tag, subjects, arms, epochs=80, batch=128, seed=42, n_osc=16, cap=80):
    import torch
    from resonet_sync import data_hgd
    from resonet_sync._vendor import rn
    data_hgd.set_cache_dir(cache)
    R = rn()
    dev = R.get_device()
    crit = torch.nn.CrossEntropyLoss()
    out = ROOT / "results" / "bandwidth_control" / tag; out.mkdir(parents=True, exist_ok=True)
    lo, hi = (subjects.split("-") + [subjects])[:2]
    for s in range(int(lo), int(hi) + 1):
        f = out / f"subj_{s:02d}.json"
        rec = json.loads(f.read_text()) if f.exists() else dict(subject=s, decim=decim, cache=str(cache), acc={})
        if all(a in rec["acc"] for a in arms):
            continue
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr, ytr, Xte, yte = R.load_subject(s, decim=decim, max_train_per_class=cap)
        C, T = Xtr.shape[1], Xtr.shape[2]
        tl, vl, te = R.make_loaders(Xtr, ytr, Xte, yte, batch_size=batch, seed=seed)
        build = dict(nl_sync=lambda: R.ResonanceNetwork(n_channels=C, n_osc=n_osc, readout="sync"),
                     nl_power=lambda: R.ResonanceNetwork(n_channels=C, n_osc=n_osc, readout="power"),
                     lin_sync=lambda: R.FastResonanceNetwork(n_channels=C, n_osc=n_osc, readout="sync"),
                     lin_power=lambda: R.FastResonanceNetwork(n_channels=C, n_osc=n_osc, readout="power"),
                     eegnet=lambda: R.EEGNet_Baseline(n_channels=C, n_samples=T),
                     shallow=lambda: R.ShallowConvNet(n_channels=C, n_samples=T))
        for a in arms:
            if a in rec["acc"]:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            m, _ = R.train_model(build[a](), tl, vl, dev, augment=True, max_epochs=epochs, verbose=False)
            rec["acc"][a] = round(float(R.evaluate(m, te, dev, crit)[1]), 4)
            f.write_text(json.dumps(rec, indent=1))
            print(f"S{s:02d} {a:10s} {rec['acc'][a]:.3f}", flush=True)
    per = [json.loads(p.read_text()) for p in sorted(out.glob("subj_[0-9]*.json"))]
    summ = dict(tag=tag, decim=decim, cache=str(cache), n_subjects=len(per),
                means={a: float(np.mean([p["acc"][a] for p in per if a in p["acc"]])) for a in arms
                       if any(a in p["acc"] for p in per)})
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-lowpass", action="store_true"); ap.add_argument("--src"); ap.add_argument("--dst")
    ap.add_argument("--cutoff", type=float, default=48.0)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--cache"); ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--tag", default="lp48_decim2"); ap.add_argument("--subjects", default="1-14")
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args(argv)
    if a.make_lowpass:
        make_lowpass(a.src, a.dst, a.cutoff)
    if a.run:
        run(a.cache, a.decim, a.tag, a.subjects, a.arms.split(","), epochs=a.epochs)


if __name__ == "__main__":
    main()
