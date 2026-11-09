"""ResoNet v2 design study (exploratory; docs/CRITIQUE_AND_V2.md). Synthetic only, CPU.

  python -m resonet_sync.screens.v2_study --phase pilot                 # G2i difficulty on pilot seed 999
  python -m resonet_sync.screens.v2_study --phase dev   [--workers 3]   # development seeds 3500-3502
  python -m resonet_sync.screens.v2_study --phase final [--workers 4]   # evaluation seeds 3600-3609 (run once)

Generators: G0, G1, G1r, G2 of DEVIATION 24 (screens/calib_p1.py, frozen difficulty) plus
  G2i  lagged coupling only: two source pairs (10 Hz and 20 Hz); the class is which pair is coupled and the sign of
       its +/-90 deg lag. Every source spectrum and every zero-lag covariance is the same in all classes, so the
       class lives only in the imaginary cross-spectrum (board 2026-11-09 round 3, Zhao).
Arms: v2 power / cov / cov+lag / cov+lag_null; v1 reference lin:sync+power (Hz bank; final phase); baselines FBLV
and FBCSP with shrinkage LDA; the generator-truth oracle. Protocol and criteria: DEVIATIONS.md, DEVIATION 27."""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy import signal

from ..stats import sign_flip_p, t_ci
from . import calib_p1 as CP
from .screen10 import ROOT, _strict_json

OUT = ROOT / "results" / "v2_study"
GENS = ("G0", "G1", "G1r", "G2", "G2i")
SEEDS = dict(dev=[3500, 3501, 3502], final=list(range(3600, 3610)))
V2_ARMS = ("power", "cov", "cov+lag", "cov+lag_null")
V1_ARMS = ("lin:sync+power",)                      # v1's best arm (Hz bank), on the informative generators only
MARGIN = 0.05
G2I_FILE = OUT / "difficulty_g2i.json"


def _g2i_trials(y, rng, A, rho):
    n, T = len(y), int(CP.T_S * CP.FS)
    S = CP._sources(rng, n, T)
    for pair, f in ((0, CP.SRC_F[0]), (2, CP.SRC_F[2])):
        own = np.isin(y, (pair, pair + 1))                           # classes (0, 1) -> pair A; (2, 3) -> pair B
        sign = np.where(y == pair, 1.0, -1.0)[:, None]               # +90 or -90 deg
        quad = -np.imag(signal.hilbert(S[:, pair], axis=-1)) * sign  # Re(z e^{+-i pi/2}): same spectrum, cos = 0
        noise = CP._band_noise(rng, n, f, T) / CP._scale(f)
        S[:, pair + 1] = np.where(own[:, None], rho * quad + np.sqrt(1 - rho ** 2) * noise, noise)
    bg = np.cumsum(rng.standard_normal((n, CP.C, T)), axis=-1) * 0.05
    bg = signal.sosfiltfilt(signal.butter(2, 4.0, "high", fs=CP.FS, output="sos"), bg, axis=-1)
    X = np.einsum("ck,nkt->nct", A, S) + bg + rng.standard_normal((n, CP.C, T))
    return X.astype(np.float32), S


def make_dataset(gen, seed, diff):
    if gen != "G2i":
        return CP.make_dataset(gen, seed, diff)
    rng = np.random.default_rng([seed, 99])
    A = rng.standard_normal((CP.C, len(CP.SRC_F)))
    ytr, yte = np.repeat(np.arange(4), CP.N_TR), np.repeat(np.arange(4), CP.N_TE)
    Xtr, Str = _g2i_trials(ytr, rng, A, diff["G2i_rho"])
    Xte, Ste = _g2i_trials(yte, rng, A, diff["G2i_rho"])
    mu, sd = Xtr.mean(axis=(0, 2), keepdims=True), Xtr.std(axis=(0, 2), keepdims=True)
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte, Str, Ste


def oracle_features(gen, S):
    if gen != "G2i":
        return CP.oracle_features(gen, S)
    z = signal.hilbert(S, axis=-1)
    out = []
    for a, b in ((0, 1), (2, 3)):
        c = (z[:, a] * z[:, b].conj()).mean(-1) / np.sqrt((np.abs(z[:, a]) ** 2).mean(-1) * (np.abs(z[:, b]) ** 2).mean(-1))
        out += [c.real, c.imag]
    return np.stack(out, 1)


def load_diff():
    d = json.loads(CP.DIFF_FILE.read_text())["chosen"]
    d["G2i_rho"] = json.loads(G2I_FILE.read_text())["chosen"] if G2I_FILE.exists() else 0.5
    return d


def pilot():
    from .. import data_hgd
    grid, accs = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5], []
    for rho in grid:
        Xtr, ytr, Xte, yte, Str, Ste = make_dataset("G2i", CP.PILOT_SEED, dict(G2i_rho=rho))
        vmask = data_hgd.train_val_split(ytr, 0.15, 42)
        accs.append(CP._logreg_val(oracle_features("G2i", Str), ytr, oracle_features("G2i", Ste), yte, vmask))
    chosen = grid[int(np.argmin(np.abs(np.array(accs) - 0.75)))]
    OUT.mkdir(parents=True, exist_ok=True)
    G2I_FILE.write_text(json.dumps(dict(pilot_seed=CP.PILOT_SEED, oracle_target=0.75,
                                        grid_oracle_acc=dict(zip(map(str, grid), accs)), chosen=chosen), indent=1))
    print(dict(zip(grid, np.round(accs, 3))), "->", chosen)


def run_one(job):
    gen, seed, diff, v2cfg, with_v1 = job
    import torch
    torch.set_num_threads(1)
    from .. import baselines as BL, data_hgd
    from ..run import P1, build
    from ..train import Split, evaluate, train
    from ..v2 import build_v2
    cfg = dict(P1)
    Xtr, ytr, Xte, yte, Str, Ste = make_dataset(gen, seed, diff)
    vmask = data_hgd.train_val_split(ytr, cfg["val_frac"], cfg["seed"])
    tr, va, te = Split(Xtr[~vmask], ytr[~vmask], 42), Split(Xtr[vmask], ytr[vmask], 43), Split(Xte, yte, 44)
    crit, dev, res = torch.nn.CrossEntropyLoss(), torch.device("cpu"), {}

    def fit(name, m):
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        m, hist = train(m, tr, va, dev, max_epochs=cfg["epochs"], batch=cfg["batch"], seed=cfg["seed"])
        torch.manual_seed(cfg["seed"] + 7)                           # the null readout draws shifts at evaluation
        res[name] = dict(bal_acc=evaluate(m, te, dev, crit, cfg["batch"])["bal_acc"], best_epoch=hist["best_epoch"],
                         epochs_run=hist["epochs_run"])
    for ro in V2_ARMS:
        fit(f"v2:{ro}", build_v2(ro, CP.C, CP.FS, v2cfg, seed=cfg["seed"]))
    if with_v1 and gen != "G0":
        for arm in V1_ARMS:
            torch.manual_seed(cfg["seed"])
            fit(arm, build(arm, CP.C, Xtr.shape[2], CP.FS, cfg))
    W = BL.fbcsp_fit(Xtr[~vmask], ytr[~vmask], CP.FS)
    post = dict(oracle=CP._logreg_val(oracle_features(gen, Str), ytr, oracle_features(gen, Ste), yte, vmask),
                FBLV_LDA=BL.lda(BL.fblv(Xtr, CP.FS), ytr, BL.fblv(Xte, CP.FS), yte, vmask),
                FBCSP_LDA=BL.lda(BL.fbcsp(Xtr, W, CP.FS), ytr, BL.fbcsp(Xte, W, CP.FS), yte, vmask))
    return dict(gen=gen, seed=seed, arms=res, post=post)


CONTRASTS = {
    "L  (cov+lag) - (cov+lag_null): lagged coupling beyond band covariance": ("v2:cov+lag", "v2:cov+lag_null"),
    "R  cov - power: zero-lag covariance beyond band power": ("v2:cov", "v2:power"),
    "D  (cov+lag_null) - cov: cost of the added null dimensions": ("v2:cov+lag_null", "v2:cov"),
    "Q  v2 cov+lag - v1 sync+power (decoder quality)": ("v2:cov+lag", "lin:sync+power"),
}


def summarise(rows):
    out = {}
    for g in GENS:
        rs = [r for r in rows if r["gen"] == g]
        if not rs:
            continue
        arms = list(rs[0]["arms"])
        S = dict(n=len(rs), means={k: float(np.mean([r["arms"][k]["bal_acc"] for r in rs])) for k in arms},
                 post_means={k: float(np.mean([r["post"][k] for r in rs])) for k in rs[0]["post"]}, contrasts={})
        for name, (a, b) in CONTRASTS.items():
            if a not in arms or b not in arms:
                continue
            d = np.array([r["arms"][a]["bal_acc"] - r["arms"][b]["bal_acc"] for r in rs])
            ci = t_ci(d, 0.90) if len(d) > 1 else [float("nan")] * 2
            S["contrasts"][name] = dict(mean=float(d.mean()), ci90=ci, p_signflip=sign_flip_p(d), n_pos=int((d > 0).sum()),
                                        deltas=d.round(4).tolist(), valid=bool(-MARGIN < ci[0] and ci[1] < MARGIN),
                                        sensitive=bool(sign_flip_p(d) < 0.05 and d.mean() >= MARGIN))
        out[g] = S
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "dev", "final"], required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gens", default=",".join(GENS))
    ap.add_argument("--n-bands", type=int, default=None); ap.add_argument("--n-spatial", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "pilot":
        pilot(); return
    diff = load_diff()
    v2cfg = {k: v for k, v in dict(n_bands=a.n_bands, n_spatial=a.n_spatial).items() if v is not None}
    name = a.phase + (f"_{a.tag}" if a.tag else "")
    part = OUT / f"partial_{name}.json"
    rows = json.loads(part.read_text()) if part.exists() else []
    done = {(r["gen"], r["seed"]) for r in rows}
    jobs = [(g, s, diff, v2cfg, a.phase == "final") for g in a.gens.split(",") for s in SEEDS[a.phase] if (g, s) not in done]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(a.workers) as p:
        for r in p.imap_unordered(run_one, jobs):
            rows.append(r); part.write_text(json.dumps(rows))
            print(r["gen"], r["seed"], {k: round(v["bal_acc"], 3) for k, v in r["arms"].items()},
                  {k: round(v, 3) for k, v in r["post"].items()}, flush=True)
    summ = summarise(rows)
    from ..run import git_hash
    (OUT / f"results_{name}.json").write_text(_strict_json(dict(
        study="ResoNet v2 design study (exploratory)", phase=a.phase, git=git_hash(), difficulty=diff, v2_config=v2cfg,
        seeds=SEEDS[a.phase], summary=summ, rows=rows), indent=1))
    for g, S in summ.items():
        print(f"== {g} (n={S['n']})", {k: round(v, 3) for k, v in S["means"].items()}, {k: round(v, 3) for k, v in S["post_means"].items()})
        for c, v in S["contrasts"].items():
            print(f"   {c:70s} {v['mean']:+.3f} CI90[{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}] +{v['n_pos']}/{S['n']}")


if __name__ == "__main__":
    main()
