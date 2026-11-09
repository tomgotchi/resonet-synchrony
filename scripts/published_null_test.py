"""The PUBLISHED linear ResoNet (vendor FastResonanceNetwork, omega ~ U(0.5, 4), h = 0.01, published training
loop) on synthetic data with known ground truth: does `sync - sync_surrogate` (the paper's capacity-matched
"phase" contrast) stay at zero when the classes carry NO phase information?

Generators and difficulty are those frozen in DEVIATION 24 (screens/calib_p1.py); seeds 3200-3209 are new.
  G0  nothing            G1  source power only (no phase information)            G2  coupling (phase lag) only

  python scripts/published_null_test.py [--workers 4]   -> results/published_null/results.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync.screens import calib_p1 as CP  # noqa: E402
from resonet_sync.stats import sign_flip_p, t_ci  # noqa: E402

OUT = ROOT / "results" / "published_null"
SEEDS = list(range(3200, 3210))
GENS = ("G0", "G1", "G2")
ARMS = {"power": "power", "sync": "sync", "surrogate": "sync_surrogate", "sync_imag": "sync_imag"}


def highpass4(X, fs=250.0, pad=900):
    """The published oscillators sit at 0.2-1.6 Hz; the real cache has no power there (4 Hz high-pass). Without this
    filter the synthetic sensor noise would drive them at resonance, which never happens on HGD. (A first run without
    it is kept as partial_UNFILTERED_discarded.json and is not used.)"""
    from scipy import signal
    h = signal.firwin(825, 4.0, pass_zero=False, fs=fs)
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    return np.ascontiguousarray(signal.filtfilt(h, [1.0], Xp, axis=-1)[..., pad:-pad])


def run_one(job):
    gen, seed, diff = job
    import torch
    torch.set_num_threads(1)
    from resonet_sync import data_hgd
    from resonet_sync._vendor import rn
    from resonet_sync.run import P1
    from resonet_sync.train import Split, evaluate, train
    cfg = dict(P1)
    Xtr, ytr, Xte, yte, _, _ = CP.make_dataset(gen, seed, diff)
    Xtr, Xte = highpass4(Xtr), highpass4(Xte)                       # like the HGD cache (4 Hz zero-phase FIR)
    mu, sd = Xtr.mean(axis=(0, 2), keepdims=True), Xtr.std(axis=(0, 2), keepdims=True)
    Xtr, Xte = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
    vmask = data_hgd.train_val_split(ytr, cfg["val_frac"], cfg["seed"])
    tr, va, te = Split(Xtr[~vmask], ytr[~vmask], 42), Split(Xtr[vmask], ytr[vmask], 43), Split(Xte, yte, 44)
    crit, dev, res = torch.nn.CrossEntropyLoss(), torch.device("cpu"), {}
    for name, ro in ARMS.items():
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        m = rn().FastResonanceNetwork(n_channels=CP.C, n_osc=cfg["n_osc"], n_classes=4, readout=ro, seed=cfg["seed"])
        m, hist = train(m, tr, va, dev, max_epochs=cfg["epochs"], batch=cfg["batch"], seed=cfg["seed"])
        res[name] = dict(bal_acc=evaluate(m, te, dev, crit, cfg["batch"])["bal_acc"], best_epoch=hist["best_epoch"])
    return dict(gen=gen, seed=seed, arms=res)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    diff = json.loads(CP.DIFF_FILE.read_text())["chosen"]
    part = OUT / "partial.json"
    rows = json.loads(part.read_text()) if part.exists() else []
    done = {(r["gen"], r["seed"]) for r in rows}
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(a.workers) as p:
        for r in p.imap_unordered(run_one, [(g, s, diff) for g in GENS for s in SEEDS if (g, s) not in done]):
            rows.append(r); part.write_text(json.dumps(rows))
            print(r["gen"], r["seed"], {k: round(v["bal_acc"], 3) for k, v in r["arms"].items()}, flush=True)
    summ = {}
    for g in GENS:
        rs = [r for r in rows if r["gen"] == g]
        S = dict(n=len(rs), means={k: float(np.mean([r["arms"][k]["bal_acc"] for r in rs])) for k in ARMS}, contrasts={})
        for name, (x, y) in {"sync - surrogate (the paper's phase contrast)": ("sync", "surrogate"),
                             "sync - power": ("sync", "power"), "surrogate - power": ("surrogate", "power"),
                             "sync_imag - 0.25 (chance)": ("sync_imag", None)}.items():
            d = np.array([r["arms"][x]["bal_acc"] - (r["arms"][y]["bal_acc"] if y else 0.25) for r in rs])
            S["contrasts"][name] = dict(mean=float(d.mean()), ci90=t_ci(d, 0.90), p_signflip=sign_flip_p(d),
                                        n_pos=int((d > 0).sum()))
        summ[g] = S
    (OUT / "results.json").write_text(json.dumps(dict(
        model="vendor FastResonanceNetwork (published linear bank), N = 16, published training loop",
        generators="DEVIATION 24 (calib_p1), frozen difficulty", seeds=SEEDS, summary=summ, rows=rows), indent=1))
    for g, S in summ.items():
        print(f"== {g} (n={S['n']})", {k: round(v, 3) for k, v in S["means"].items()})
        for c, v in S["contrasts"].items():
            print(f"   {c:46s} {v['mean']:+.3f} CI90[{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}] p {v['p_signflip']:.4f} +{v['n_pos']}/{S['n']}")


if __name__ == "__main__":
    main()
