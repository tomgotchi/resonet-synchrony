"""How much cross-oscillator structure does the published surrogate keep? (DEVIATION 31; descriptive)

The surrogate rolls each oscillator's unit-phasor series by `surr_frac[i] * T` samples (vendor/resonance_network.py)
and recomputes C_ij = <u_i conj(u_j)>_t. For each published checkpoint (RNsync_S*.pt) and its subject's TEST trials
(the authors' own evaluation data; no v2 model, no training) this reports the rms of |C_ij| over the 120 pairs with
and without the shift, and the ratio by the pair's relative shift (under 0.5 s, 0.5-1 s, over 1 s).

  python scripts/surrogate_retention.py --ckpt-dir <results_hg> --cache <subj_cache_hg> [--subjects 1-14]
  python scripts/surrogate_retention.py --ckpt-dir <results_hg> --synthetic      # prototype input, checkpoint 1
Writes results/surrogate_retention/results.json (or --out). Magnitude is not class information: see DEVIATION 31."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FS = 250.0
BINS = (("under_0.5s", 0.0, 0.5), ("0.5_to_1s", 0.5, 1.0), ("over_1s", 1.0, 1e9))


def synthetic_input(seed=0, B=6, T=1125, C=128):
    """The board's prototype input: 6 shared 1/f sources + sensor noise, 4 Hz high-pass, z-scored."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    rng.normal(size=(B, 6, T + 400))                 # unused draw, kept so the stream matches the board's prototype
    f = np.fft.rfftfreq(T + 400, 1 / FS)
    spec = (rng.normal(size=(B, 6, len(f))) + 1j * rng.normal(size=(B, 6, len(f)))) / np.maximum(f, 1)[None, None] ** 0.9
    src = np.fft.irfft(spec, n=T + 400, axis=-1)[..., 200:-200]
    X = np.einsum("cs,bst->bct", rng.normal(size=(C, 6)), src) + 0.3 * rng.normal(size=(B, C, T))
    X = signal.filtfilt(signal.firwin(201, 4.0, pass_zero=False, fs=FS), [1.0], X, axis=-1)
    return ((X - X.mean(axis=(0, 2), keepdims=True)) / X.std(axis=(0, 2), keepdims=True)).astype(np.float32)


def retention_from_phasors(u, shifts, fs=FS):
    """u (B, T, N) complex unit phasors; shifts (N,) samples. -> dict of rms |C| and shifted/unshifted ratios."""
    T, N = u.shape[1], u.shape[2]
    us = torch.stack([torch.roll(u[:, :, i], int(shifts[i]), dims=1) for i in range(N)], dim=2)
    iu, ju = torch.triu_indices(N, N, 1)
    C = lambda U: (torch.einsum("bti,btj->bij", U, U.conj()) / T)[:, iu, ju].abs() ** 2          # (B, P)
    c0, c1 = C(u).mean(0).cpu().numpy(), C(us).mean(0).cpu().numpy()                             # mean square per pair
    d = np.array([min(abs(int(shifts[i]) - int(shifts[j])), T - abs(int(shifts[i]) - int(shifts[j]))) / fs
                  for i, j in zip(iu.tolist(), ju.tolist())])
    out = dict(rms_unshifted=float(np.sqrt(c0.mean())), rms_shifted=float(np.sqrt(c1.mean())),
               ratio=float(np.sqrt(c1.mean() / c0.mean())), mean_abs_phasor=float(u.mean(1).abs().mean()), by_relative_shift={})
    for name, lo, hi in BINS:
        k = (d >= lo) & (d < hi)
        out["by_relative_shift"][name] = dict(n_pairs=int(k.sum()),
                                              ratio=float(np.sqrt(c1[k].mean() / c0[k].mean())) if k.any() else None)
    return out


def retention(model, X, device="cpu", batch=32):
    """Run the published nonlinear network on trials X (n, C, T) and measure what its surrogate keeps."""
    model = model.to(device).eval()
    us = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            _, z = model(torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device), return_traj=True)
            us.append((z / (z.abs() + 1e-6)).cpu())
    u = torch.cat(us)
    return retention_from_phasors(u, (model.surr_frac.cpu() * u.shape[1]).long())


def load_model(ckpt):
    from resonet_sync._vendor import rn
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = rn().ResonanceNetwork(n_channels=sd["W_enc"].shape[1], n_osc=len(sd["omega"]), n_classes=4, readout="sync")
    m.load_state_dict(sd)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--cache", default=None)
    ap.add_argument("--subjects", default="1-14"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "surrogate_retention" / "results.json"))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.synthetic:
        r = retention(load_model(Path(a.ckpt_dir) / "RNsync_S1.pt"), synthetic_input(), dev)
        print(json.dumps(r, indent=1)); return r
    from resonet_sync import data_hgd
    from resonet_sync._vendor import rn
    data_hgd.set_cache_dir(a.cache)
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else dict(note="DEVIATION 31; published checkpoints, test trials", per_subject={})
    for s in range(int(lo), int(hi) + 1):
        ck = Path(a.ckpt_dir) / f"RNsync_S{s}.pt"
        if str(s) in res["per_subject"] or not ck.exists():
            continue
        _, _, Xte, _ = rn().load_subject(s, decim=2, max_train_per_class=80)       # as published (decim 2)
        res["per_subject"][str(s)] = retention(load_model(ck), Xte, dev)
        out.write_text(json.dumps(res, indent=1))
        print(f"S{s:02d} ratio {res['per_subject'][str(s)]['ratio']:.3f}", flush=True)
    ratios = [v["ratio"] for v in res["per_subject"].values()]
    if ratios:
        med = float(np.median(ratios))
        res["median_ratio"] = med
        res["reading"] = ("the surrogate keeps most of the cross-term magnitude" if med >= 0.8 else
                          "it removes most" if med <= 0.5 else "partial")
        out.write_text(json.dumps(res, indent=1))
        print(f"median ratio over {len(ratios)} subjects: {med:.3f} -> {res['reading']}")
    return res


if __name__ == "__main__":
    main()
