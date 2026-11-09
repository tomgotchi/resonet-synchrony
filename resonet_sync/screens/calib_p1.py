"""DEVIATION 24: calibration screen of the Phase 1 contrasts (review board 2026-11-09). Synthetic only, CPU, lin bank.

  python -m resonet_sync.screens.calib_p1 --phase pilot            # difficulty, pilot seed 999 only
  python -m resonet_sync.screens.calib_p1 --phase run --workers 4  # evaluation seeds 3000-3009 (resumable)
  python -m resonet_sync.screens.calib_p1 --phase report           # pre-committed criteria -> Phase 1 wording

HGD-like trials: 16 channels, 4.5 s at 250 Hz, 4 classes, 80 train trials per class (the seed-42 15% validation
carve is inside them, as in Phase 1) and 40 test trials per class. Four narrowband Gaussian sources (10, 12, 20,
24 Hz, 3 Hz wide) are mixed into the channels by a fixed random matrix, plus white sensor noise and a 4 Hz
high-passed brownian background (the HGD cache is 4-125 Hz).

Generators (the class is carried by ...):
  G0   nothing (labels independent of the data)
  G1   source power only: class k = amplitude drop (ERD) of source k; independent sources, random phases
  G1r  one source's power only (rank-1 SNR; the dilution null): class = amplitude level of the 10 Hz source
  G2   coupling only: sources 1 and 2 share a 10 Hz component with fixed coherence rho; class = their phase lag
       {0, 45, 90, 135} deg. Each source's spectrum is identical in every class (oracle source power at chance).

Arms (Phase 1 training loop, lin bank, published config except n_channels): power, sync, surrogate,
surrogate_random, sync+power, power+surrogate, power+surrogate_random. Post hoc on the trained sync+power bank
(frozen): band-power baselines FBLV (filter-bank log-variance) and FBCSP, their hybrids with sync and with the
fresh-roll surrogate, and the D2 probe (logistic regression on frozen sync / fixed-roll / fresh-roll features)."""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy import signal

from ..stats import sign_flip_p, t_ci
from .screen10 import ROOT, _strict_json

OUT = ROOT / "results" / "calib_p1"
FS, T_S, C, NCLS = 250.0, 4.5, 16, 4
N_TR, N_TE = 80, 40
SRC_F = (10.0, 12.0, 20.0, 24.0)
SRC_BW = 3.0
PILOT_SEED = 999
SEEDS = list(range(3000, 3010))
GENS = ("G0", "G1", "G1r", "G2")
ARMS = ("power", "sync", "surrogate", "surrogate_random", "sync+power", "power+surrogate", "power+surrogate_random")
FB_BANDS = [(lo, lo + 4.0) for lo in range(4, 40, 4)]          # 9 bands, 4-40 Hz
MARGIN = 0.05
# difficulty (set by --phase pilot on seed 999 only; frozen in DEVIATION 24 before any evaluation seed)
DIFF = dict(G1_erd=None, G1r_step=None, G2_rho=None)
DIFF_FILE = OUT / "difficulty.json"


# ============================================================================ generators
def _band_noise(rng, n, f, T, burn=500):
    sos = signal.butter(2, [f - SRC_BW / 2, f + SRC_BW / 2], btype="band", fs=FS, output="sos")
    return signal.sosfilt(sos, rng.standard_normal((n, T + burn)), axis=-1)[:, burn:]


_SCALE = {}


def _scale(f):
    """Fixed normalising constant (std of the band-limited source); a constant, so per-trial power varies."""
    if f not in _SCALE:
        _SCALE[f] = float(_band_noise(np.random.default_rng(12345), 64, f, int(T_S * FS)).std())
    return _SCALE[f]


def _sources(rng, n, T):
    return np.stack([_band_noise(rng, n, f, T) / _scale(f) for f in SRC_F], 1)      # (n, K, T)


def _trials(gen, y, rng, A, diff):
    n, T = len(y), int(T_S * FS)
    S = _sources(rng, n, T)
    if gen == "G1":
        S[np.arange(n), y] *= (1.0 - diff["G1_erd"])
    elif gen == "G1r":
        S[:, 0] *= (1.0 - diff["G1r_step"] * y)[:, None]
    elif gen == "G2":
        rho = diff["G2_rho"]
        z1 = signal.hilbert(S[:, 0], axis=-1)
        lag = np.deg2rad(np.array([0.0, 45.0, 90.0, 135.0]))[y]
        shared = np.real(z1 * np.exp(1j * lag)[:, None])         # same spectrum as source 0, rotated by the lag
        S[:, 1] = rho * shared + np.sqrt(1 - rho ** 2) * _band_noise(rng, n, SRC_F[0], T) / _scale(SRC_F[0])
    bg = np.cumsum(rng.standard_normal((n, C, T)), axis=-1) * 0.05
    bg = signal.sosfiltfilt(signal.butter(2, 4.0, "high", fs=FS, output="sos"), bg, axis=-1)
    X = np.einsum("ck,nkt->nct", A, S) + bg + rng.standard_normal((n, C, T))
    return X.astype(np.float32), S


def make_dataset(gen, seed, diff):
    rng = np.random.default_rng([seed, GENS.index(gen)])
    A = rng.standard_normal((C, len(SRC_F)))
    ytr, yte = np.repeat(np.arange(NCLS), N_TR), np.repeat(np.arange(NCLS), N_TE)
    if gen == "G0":
        Xtr, Str = _trials("G0", ytr, rng, A, diff)
        Xte, Ste = _trials("G0", yte, rng, A, diff)
        ytr, yte = rng.permutation(ytr), rng.permutation(yte)      # labels carry nothing
    else:
        Xtr, Str = _trials(gen, ytr, rng, A, diff)
        Xte, Ste = _trials(gen, yte, rng, A, diff)
    mu, sd = Xtr.mean(axis=(0, 2), keepdims=True), Xtr.std(axis=(0, 2), keepdims=True)
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte, Str, Ste


# ============================================================================ baselines and probes
def _logreg_val(Ftr, ytr, Fte, yte, vmask, Cs=np.logspace(-3, 2, 11)):
    """L2 logistic regression; C chosen on the validation carve, refit on train minus the carve."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    a, b = ~vmask, vmask
    sc = StandardScaler().fit(Ftr[a])
    A, B, Te = sc.transform(Ftr[a]), sc.transform(Ftr[b]), sc.transform(Fte)
    best = max(Cs, key=lambda c: (LogisticRegression(C=c, max_iter=3000).fit(A, ytr[a]).predict(B) == ytr[b]).mean())
    pred = LogisticRegression(C=best, max_iter=3000).fit(A, ytr[a]).predict(Te)
    return float(np.mean([(pred[yte == k] == k).mean() for k in np.unique(yte)]))


def _fb(X):
    return [signal.sosfilt(signal.butter(4, b, btype="band", fs=FS, output="sos"), X, axis=-1)[..., 125:]
            for b in FB_BANDS]


def fblv(X):
    return np.concatenate([np.log(x.var(-1)) for x in _fb(X)], 1)


def fbcsp_fit(Xtr, ytr, n_pairs=2):
    """One-vs-rest CSP per band (shrinkage covariances), n_pairs filter pairs per class and band; train only."""
    W = []
    for x in _fb(Xtr):
        cov = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
        cov = cov / np.trace(cov, axis1=1, axis2=2)[:, None, None]
        Wb = []
        for k in np.unique(ytr):
            Ca, Cb = cov[ytr == k].mean(0), cov[ytr != k].mean(0)
            Ca, Cb = [0.9 * M + 0.1 * np.trace(M) / C * np.eye(C) for M in (Ca, Cb)]
            from scipy.linalg import eigh
            _, V = eigh(Ca, Ca + Cb)
            Wb.append(np.c_[V[:, :n_pairs], V[:, -n_pairs:]])
        W.append(np.concatenate(Wb, 1))
    return W


def fbcsp(X, W):
    return np.concatenate([np.log(np.einsum("cf,nct->nft", w, x).var(-1)) for w, x in zip(W, _fb(X))], 1)


def oracle_features(gen, S):
    """What the generator actually varies: source log-power (G1, G1r) or the source-1/2 cross-spectrum (G2)."""
    if gen == "G2":
        z = signal.hilbert(S[:, :2], axis=-1)
        c = (z[:, 0] * z[:, 1].conj()).mean(-1) / np.sqrt((np.abs(z[:, 0]) ** 2).mean(-1) * (np.abs(z[:, 1]) ** 2).mean(-1))
        return np.c_[c.real, c.imag]
    return np.log(S.var(-1))


# ============================================================================ one seed
def run_one(job):
    gen, seed, diff = job
    import torch
    torch.set_num_threads(1)
    from .. import data_hgd
    from ..readouts import Readout
    from ..run import P1, build
    from ..train import Split, evaluate, train
    cfg = dict(P1)
    Xtr, ytr, Xte, yte, Str, Ste = make_dataset(gen, seed, diff)
    vmask = data_hgd.train_val_split(ytr, cfg["val_frac"], cfg["seed"])
    tr, va, te = Split(Xtr[~vmask], ytr[~vmask], cfg["seed"]), Split(Xtr[vmask], ytr[vmask], cfg["seed"] + 1), \
        Split(Xte, yte, cfg["seed"] + 2)
    crit = torch.nn.CrossEntropyLoss()
    dev = torch.device("cpu")
    res, keep = {}, None
    for ro in ARMS:
        torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        m = build(f"lin:{ro}", C, Xtr.shape[2], FS, cfg)
        m, hist = train(m, tr, va, dev, max_epochs=cfg["epochs"], batch=cfg["batch"], seed=cfg["seed"])
        ev = evaluate(m, te, dev, crit, cfg["batch"])
        res[ro] = dict(bal_acc=ev["bal_acc"], best_epoch=hist["best_epoch"], epochs_run=hist["epochs_run"],
                       train_loss_drop=float(hist["train_loss"][0] - min(hist["train_loss"])))
        if ro == "sync+power":
            keep = m
    # ---- post hoc on the frozen sync+power bank
    keep.eval()
    with torch.no_grad():
        ztr, zte = keep.bank(torch.as_tensor(Xtr)), keep.bank(torch.as_tensor(Xte))
    feats = {}
    for kind in ("sync", "surrogate", "surrogate_random", "power"):
        r = Readout(cfg["n_osc"], kind=kind, seed=cfg["seed"])
        torch.manual_seed(cfg["seed"] + 7)
        with torch.no_grad():
            feats[kind] = (r.features(ztr).numpy(), r.features(zte).numpy())
    W = fbcsp_fit(Xtr[~vmask], ytr[~vmask])
    feats["FBLV"] = (fblv(Xtr), fblv(Xte))
    feats["FBCSP"] = (fbcsp(Xtr, W), fbcsp(Xte, W))
    feats["oracle"] = (oracle_features(gen, Str), oracle_features(gen, Ste))
    cat = lambda *ks: (np.concatenate([feats[k][0] for k in ks], 1), np.concatenate([feats[k][1] for k in ks], 1))
    post = {}
    for name, ks in [("oracle", ("oracle",)), ("FBLV", ("FBLV",)), ("FBCSP", ("FBCSP",)),
                     ("FBLV+sync", ("FBLV", "sync")), ("FBLV+surrogate_random", ("FBLV", "surrogate_random")),
                     ("FBCSP+sync", ("FBCSP", "sync")), ("FBCSP+surrogate_random", ("FBCSP", "surrogate_random")),
                     ("probe:sync", ("sync",)), ("probe:surrogate", ("surrogate",)),
                     ("probe:surrogate_random", ("surrogate_random",)), ("probe:power", ("power",))]:
        a, b = cat(*ks)
        post[name] = _logreg_val(a, ytr, b, yte, vmask)
    return dict(gen=gen, seed=seed, arms=res, post=post)


# ============================================================================ contrasts and criteria
CONTRASTS = {   # name: (a, b, source)   source = "arms" (trained readouts) or "post" (frozen bank)
    "F1 sync-surrogate": ("sync", "surrogate", "arms"),
    "F1r sync-surrogate_random": ("sync", "surrogate_random", "arms"),
    "F2 (sync+power)-(power+surrogate)": ("sync+power", "power+surrogate", "arms"),
    "F2r (sync+power)-(power+surrogate_random)": ("sync+power", "power+surrogate_random", "arms"),
    "F3 (sync+power)-power": ("sync+power", "power", "arms"),
    "D3 (power+surrogate)-power": ("power+surrogate", "power", "arms"),
    "F5a (FBCSP+sync)-(FBCSP+surrogate_random)": ("FBCSP+sync", "FBCSP+surrogate_random", "post"),
    "F5b (FBLV+sync)-(FBLV+surrogate_random)": ("FBLV+sync", "FBLV+surrogate_random", "post"),
}


def score(r, key, src):
    return r["arms"][key]["bal_acc"] if src == "arms" else r["post"][key]


def summarise(rows):
    out = {}
    for g in GENS:
        rs = [r for r in rows if r["gen"] == g]
        if not rs:
            continue
        S = dict(n=len(rs), means={k: float(np.mean([r["arms"][k]["bal_acc"] for r in rs])) for k in ARMS},
                 post_means={k: float(np.mean([r["post"][k] for r in rs])) for k in rs[0]["post"]}, contrasts={})
        for name, (a, b, src) in CONTRASTS.items():
            d = np.array([score(r, a, src) - score(r, b, src) for r in rs])
            ci90 = t_ci(d, 0.90)
            S["contrasts"][name] = dict(mean=float(d.mean()), ci90=ci90, p_signflip=sign_flip_p(d),
                                        n_pos=int((d > 0).sum()), deltas=d.round(4).tolist(),
                                        valid=bool(-MARGIN < ci90[0] and ci90[1] < MARGIN),
                                        sensitive=bool(sign_flip_p(d) < 0.05 and d.mean() >= MARGIN))
        out[g] = S
    return out


def decisions(summ):
    """Pre-committed mapping (DEVIATION 24) from calibration outcomes to what Phase 1 may claim."""
    ok = lambda c: all(summ[g]["contrasts"][c]["valid"] for g in ("G0", "G1", "G1r")) and summ["G2"]["contrasts"][c]["sensitive"]
    F2, F3 = "F2 (sync+power)-(power+surrogate)", "F3 (sync+power)-power"
    phase_word = ok(F2) and ok(F3)
    f5 = next((nm for nm, c in (("FBCSP", "F5a (FBCSP+sync)-(FBCSP+surrogate_random)"),
                                ("FBLV", "F5b (FBLV+sync)-(FBLV+surrogate_random)")) if ok(c)), None)
    return dict(
        F2_F3_may_be_called_phase=phase_word,
        F2_F3_wording=("phase synchrony beyond the bank's log-powers" if phase_word else
                       "zero-lag cross-oscillator structure beyond the bank's 16 log-powers"),
        F5_baseline=f5,
        F5_status=("confirmatory secondary; the only Phase 1 contrast that may use the word 'phase'" if f5 else
                   "descriptive; no Phase 1 result is called 'phase'"),
        per_contrast={c: dict(valid_G0=summ["G0"]["contrasts"][c]["valid"], valid_G1=summ["G1"]["contrasts"][c]["valid"],
                              valid_G1r=summ["G1r"]["contrasts"][c]["valid"],
                              sensitive_G2=summ["G2"]["contrasts"][c]["sensitive"]) for c in CONTRASTS})


# ============================================================================ pilot (difficulty), seed 999 only
def pilot():
    """Pick the ERD / amplitude step / coherence so the ORACLE (generator-truth features, logistic regression)
    scores about 0.75 balanced accuracy on pilot seed 999. No ResoNet arm is trained here."""
    grids = dict(G1_erd=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4], G1r_step=[0.04, 0.06, 0.08, 0.1, 0.12, 0.15],
                 G2_rho=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    gen_of = dict(G1_erd="G1", G1r_step="G1r", G2_rho="G2")
    chosen, log = {}, {}
    from .. import data_hgd
    for key, grid in grids.items():
        accs = []
        for v in grid:
            diff = dict(DIFF, **{key: v})
            Xtr, ytr, Xte, yte, Str, Ste = make_dataset(gen_of[key], PILOT_SEED, diff)
            vmask = data_hgd.train_val_split(ytr, 0.15, 42)
            accs.append(_logreg_val(oracle_features(gen_of[key], Str), ytr, oracle_features(gen_of[key], Ste), yte, vmask))
        log[key] = dict(zip(map(str, grid), accs))
        chosen[key] = grid[int(np.argmin(np.abs(np.array(accs) - 0.75)))]
        print(key, {g: round(a, 3) for g, a in zip(grid, accs)}, "->", chosen[key], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    DIFF_FILE.write_text(json.dumps(dict(pilot_seed=PILOT_SEED, oracle_target=0.75, grid_oracle_acc=log,
                                         chosen=chosen), indent=1))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "run", "report"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "pilot":
        pilot(); return
    diff = json.loads(DIFF_FILE.read_text())["chosen"]
    part = OUT / "partial.json"
    rows = json.loads(part.read_text()) if part.exists() else []
    if a.phase == "run":
        done = {(r["gen"], r["seed"]) for r in rows}
        jobs = [(g, s, diff) for g in GENS for s in SEEDS if (g, s) not in done]
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with Pool(a.workers) as p:
            for r in p.imap_unordered(run_one, jobs):
                rows.append(r)
                part.write_text(json.dumps(rows))
                print(r["gen"], r["seed"], {k: round(v["bal_acc"], 3) for k, v in r["arms"].items()},
                      {k: round(v, 3) for k, v in r["post"].items()}, flush=True)
    summ = summarise(rows)
    from ..run import git_hash
    res = dict(protocol="DEVIATIONS.md DEVIATION 24", git=git_hash(), difficulty=diff, seeds=SEEDS, summary=summ,
               rows=rows)
    if all(g in summ and summ[g]["n"] == len(SEEDS) for g in GENS):
        res["decisions"] = decisions(summ)
    (OUT / "results.json").write_text(_strict_json(res, indent=1))
    for g, S in summ.items():
        print(f"== {g} (n={S['n']}) means", {k: round(v, 3) for k, v in S["means"].items()})
        print("   post", {k: round(v, 3) for k, v in S["post_means"].items()})
        for c, v in S["contrasts"].items():
            print(f"   {c:45s} {v['mean']:+.3f} CI90[{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}] p {v['p_signflip']:.4f} "
                  f"+{v['n_pos']}/{S['n']} valid={v['valid']} sensitive={v['sensitive']}")
    if "decisions" in res:
        print(json.dumps(res["decisions"], indent=1))


if __name__ == "__main__":
    main()
