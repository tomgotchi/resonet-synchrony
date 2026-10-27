"""CLI: python -m resonet_sync.run --phase 1 [--subjects 1-14] [--arms core|all|a,b,...] [--smoke]

Phase 1 = recalibrated replication on HGD (full-band cache, decim 2 -> 250 Hz, N = 16, cap 80/class,
<= 80 epochs, seed 42: the published high-gamma configuration) with Hz-parameterized banks and the
new arms. Per-subject results are written as they finish (resumable).
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import data_hgd
from ._vendor import rn
from .banks import BANKS, HzOscBank
from .readouts import READOUTS, Readout, RNModel
from .stats import contrast_table
from .train import Split, evaluate, train

ROOT = Path(__file__).resolve().parent.parent
P1 = dict(decim=2, cap=80, n_osc=16, epochs=80, batch=128, seed=42, band=(1.0, 40.0), val_frac=0.15,
          rk4_substeps=1)

CORE = [f"{b}:{r}" for b in ("lin", "nl_coupled") for r in ("sync", "surrogate", "power")]
ALL = (CORE
       + [f"nl_uncoupled:{r}" for r in ("sync", "surrogate", "power")]
       + [f"nl_uncoupled_wide:{r}" for r in ("sync", "surrogate", "power")]
       + [f"{b}:sync+power" for b in BANKS]
       + [f"{b}:trial_shuffle" for b in BANKS]
       + ["eegnet", "shallow"])


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def load_config():
    p = ROOT / "config.yaml"
    if not p.exists():
        return {}
    import yaml
    return yaml.safe_load(p.read_text()) or {}


def build(arm, C, T, fs, cfg):
    R = rn()
    if arm == "eegnet":
        return R.EEGNet_Baseline(n_channels=C, n_samples=T)
    if arm == "shallow":
        return R.ShallowConvNet(n_channels=C, n_samples=T)
    bank, ro = arm.split(":")
    assert bank in BANKS and ro in READOUTS, arm
    b = HzOscBank(n_channels=C, n_osc=cfg["n_osc"], fs=fs, band=tuple(cfg["band"]), kind=bank, seed=cfg["seed"],
                   rk4_substeps=cfg.get("rk4_substeps", 1))
    return RNModel(b, Readout(cfg["n_osc"], n_classes=4, kind=ro, seed=cfg["seed"]))


def run_subject(Xtr, ytr, Xte, yte, arms, cfg, fs, device, log=print, done=None):
    vmask = data_hgd.train_val_split(ytr, cfg["val_frac"], cfg["seed"])
    tr = Split(Xtr[~vmask], ytr[~vmask], cfg["seed"])
    va = Split(Xtr[vmask], ytr[vmask], cfg["seed"] + 1)
    te = Split(Xte, yte, cfg["seed"] + 2)
    C, T = Xtr.shape[1], Xtr.shape[2]
    out = dict(done or {})
    crit = torch.nn.CrossEntropyLoss()
    for arm in arms:
        if arm in out:
            continue
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        t0 = time.time()
        m = build(arm, C, T, fs, cfg)
        m, hist = train(m, tr, va, device, max_epochs=cfg["epochs"], batch=cfg["batch"], seed=cfg["seed"])
        ev = evaluate(m, te, device, crit, cfg["batch"])
        rec = dict(test_acc=ev["acc"], test_bal_acc=ev["bal_acc"], n_test=ev["n"],
                   val_loss_best=float(min(hist["val_loss"])), best_epoch=hist["best_epoch"],
                   epochs_run=hist["epochs_run"], minutes=round((time.time() - t0) / 60, 2),
                   n_params=int(sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())))
        if isinstance(m, RNModel):
            rec["f0_hz_learned"] = m.bank.f0_hz().detach().cpu().numpy().round(3).tolist()
            rec["bw_hz_learned"] = m.bank.bw_hz().detach().cpu().numpy().round(4).tolist()
        out[arm] = rec
        log(f"  {arm:28s} acc {ev['acc']:.3f} bal {ev['bal_acc']:.3f}  ep {hist['epochs_run']:3d} "
            f"(best {hist['best_epoch']})  {rec['minutes']:.1f} min")
        yield arm, out
        del m; gc.collect()


def contrasts_phase1(per_subject_acc):
    fam = {
        "H2_sync_vs_surrogate": [(f"{b}: sync-surrogate", f"{b}:sync", f"{b}:surrogate") for b in BANKS],
        "coupling": [("nl_coupled-lin (sync)", "nl_coupled:sync", "lin:sync"),
                     ("nl_coupled-nl_uncoupled (sync)", "nl_coupled:sync", "nl_uncoupled:sync"),
                     ("nl_coupled-nl_uncoupled_wide (sync)", "nl_coupled:sync", "nl_uncoupled_wide:sync"),
                     ("nl_uncoupled-lin (sync)", "nl_uncoupled:sync", "lin:sync")],
        "incremental_sync_over_power": [(f"{b}: (sync+power)-power", f"{b}:sync+power", f"{b}:power") for b in BANKS],
        "induced_vs_evoked": [(f"{b}: sync-trial_shuffle", f"{b}:sync", f"{b}:trial_shuffle") for b in BANKS],
    }
    return {k: contrast_table(per_subject_acc, v) for k, v in fam.items()}


def phase1(args):
    cfg = dict(P1)
    if args.epochs:
        cfg["epochs"] = args.epochs
    cfg["rk4_substeps"] = args.rk4_substeps
    conf = load_config()
    cache = args.cache or conf.get("paths", {}).get("hgd_cache_hg")
    out_dir = ROOT / "results" / ("phase1" if args.rk4_substeps == 1 else f"phase1_rk4x{args.rk4_substeps}")
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = CORE if args.arms == "core" else ALL if args.arms == "all" else args.arms.split(",")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        return smoke(arms, cfg, device)
    if not cache or not Path(cache).exists():
        raise SystemExit("HGD full-band cache not found: set paths.hgd_cache_hg in config.yaml or --cache")
    bc = data_hgd.band_check(cache)
    print("cache band check:", bc)
    if not bc["full_band"]:
        raise SystemExit("cache looks LOW-PASSED; Pitch 1 requires the full-band (subj_cache_hg) cache")
    lo, hi = (args.subjects.split("-") + [args.subjects])[:2]
    fs = data_hgd.FS_NATIVE / cfg["decim"]
    for s in range(int(lo), int(hi) + 1):
        path = out_dir / f"subj_{s:02d}.json"
        rec = json.loads(path.read_text()) if path.exists() else {}
        todo = [a for a in arms if a not in rec.get("arms", {})]
        if not todo:
            continue
        npz = Path(cache) / f"subj_{s}.npz"
        Xtr, ytr, Xte, yte = data_hgd.load(cache, s, cfg["decim"], cfg["cap"], cfg["seed"])
        meta = dict(subject=s, config=cfg, git=git_hash(), device=str(device), data_file=str(npz),
                    data_sha256=rec.get("data_sha256") or data_hgd.sha256(npz), band_check=bc,
                    chance_threshold=rn().chance_threshold(len(yte), 4), n_train=int(len(ytr)),
                    n_test=int(len(yte)))
        print(f"[S{s:02d}] train {Xtr.shape} test {Xte.shape} chance thr {meta['chance_threshold']:.3f}")
        for arm, res in run_subject(Xtr, ytr, Xte, yte, todo, cfg, fs, device, done=rec.get("arms")):
            path.write_text(json.dumps(dict(meta, arms=res), indent=1))
        del Xtr, Xte; gc.collect()
    summarize(out_dir)


PUB_MAP = {"nl_sync": "nl_coupled:sync", "nl_surrogate": "nl_coupled:surrogate", "nl_power": "nl_coupled:power",
           "lin_sync": "lin:sync", "lin_surrogate": "lin:surrogate", "lin_power": "lin:power"}


def compare_published(per, published_dir):
    """Hz-parameterized vs published (omega*h) arms, paired by subject, on plain test accuracy
    (the metric the published JSON stores). Published runs: GPU, same cap/decim/epochs/seed."""
    f = Path(published_dir) / "surrogate_results.json"
    if not f.exists():
        return None
    pub = {r["subject"]: r for r in json.loads(f.read_text())}
    rows_new, rows_pub = [], []
    for p in per:
        if p["subject"] in pub:
            rows_new.append({k: p["arms"][v]["test_acc"] for k, v in PUB_MAP.items() if v in p["arms"]})
            rows_pub.append({k: pub[p["subject"]].get(k, np.nan) for k in PUB_MAP})
    paired = [{**{f"new:{k}": v for k, v in a.items()}, **{f"pub:{k}": v for k, v in b.items()}}
              for a, b in zip(rows_new, rows_pub)]
    hz_vs_pub = contrast_table(paired, [(f"Hz - published: {k}", f"new:{k}", f"pub:{k}") for k in PUB_MAP])
    effect = contrast_table(paired, [(f"published sync-surrogate ({b})", f"pub:{b}_sync", f"pub:{b}_surrogate")
                                     for b in ("nl", "lin")]
                            + [(f"Hz sync-surrogate ({b})", f"new:{b}_sync", f"new:{b}_surrogate") for b in ("nl", "lin")])
    return dict(n=len(paired), hz_minus_published=hz_vs_pub, phase_effect_both=effect)


def summarize(out_dir):
    files = sorted(Path(out_dir).glob("subj_*.json"))
    per = [json.loads(f.read_text()) for f in files]
    acc = [{a: r["test_bal_acc"] for a, r in p["arms"].items()} for p in per]
    if not acc:
        return
    arms = sorted({a for u in acc for a in u})
    means = {a: float(np.nanmean([u.get(a, np.nan) for u in acc])) for a in arms}
    summ = dict(n_subjects=len(per), subjects=[p["subject"] for p in per], metric="test balanced accuracy",
                means=means, contrasts=contrasts_phase1(acc), git=per[-1].get("git"), config=per[-1]["config"])
    pub_dir = load_config().get("paths", {}).get("published_results_hg") or str(ROOT.parent / "results_hg")
    summ["vs_published"] = compare_published(per, pub_dir)
    (Path(out_dir) / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(means, indent=1))
    fams = dict(summ["contrasts"])
    if summ["vs_published"]:
        fams.update(summ["vs_published"]["hz_minus_published"] and
                    {"vs_published": summ["vs_published"]["hz_minus_published"],
                     "phase_effect_published_vs_hz": summ["vs_published"]["phase_effect_both"]})
    for fam, rows in fams.items():
        print(f"-- {fam}")
        for r in rows:
            if r["n"]:
                print(f"   {r['contrast']:40s} d={r['mean_diff']:+.3f} CI[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] "
                      f"dz={r['dz']:+.2f} p={r['p']:.4f} holm={r['p_holm']:.4f} wins {r['wins']}/{r['n']}")


def smoke(arms, cfg, device):
    """End-to-end run on tiny synthetic data (NOT a result): 8 channels, 1 s @ 250 Hz, 4 classes."""
    from .synth_smoke import tiny_dataset
    cfg = dict(cfg, n_osc=6, epochs=3, batch=32)
    Xtr, ytr, Xte, yte = tiny_dataset(seed=0)
    for arm, res in run_subject(Xtr, ytr, Xte, yte, arms, cfg, 250.0, device):
        pass
    print("smoke OK:", {a: round(r["test_acc"], 3) for a, r in res.items()})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--subjects", default="1-14")
    ap.add_argument("--arms", default="all", help="core | all | comma list like lin:sync,nl_coupled:sync")
    ap.add_argument("--cache", default=None, help="full-band HGD cache dir (subj_cache_hg)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rk4-substeps", type=int, default=1,
                    help="RK4 sub-steps per sample for nl banks (1 = published scheme)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.phase == 1:
        phase1(a)
    else:
        raise SystemExit(f"phase {a.phase} not implemented yet (stop-and-report after Phase 1)")


if __name__ == "__main__":
    main()
