"""What do the PUBLISHED trained ResoNet models compute? Read-only checks on the published checkpoints
(reso-net/results_hg/RNsync_S*.pt, fs = 250 Hz, h = 0.01) and on the published code (vendor/).

  python scripts/published_forensics.py [--ckpt-dir ../results_hg]   -> results/published_forensics.json

1. Learned oscillator frequencies and bandwidths in Hz, per subject.
2. Frequency response of the trained oscillators over the data band (4-124 Hz): gain, similarity between
   oscillators, selectivity for +f over -f (a resonating complex oscillator passes +f0 and rejects -f0).
3. Do the 16 distinct oscillator frequencies matter? Sync features of the published linear model vs the same model
   with ONE shared frequency and damping, on synthetic HGD-like data high-passed at 4 Hz like the real cache.
4. Is the oscillator "phase" an oscillation phase? Instantaneous frequency of the oscillator states."""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FS, H = 250.0, 0.01


def checkpoints(ckpt_dir):
    rows = []
    for f in sorted(glob.glob(str(Path(ckpt_dir) / "RNsync_S*.pt")), key=lambda p: int(p.split("_S")[-1][:-3])):
        sd = torch.load(f, map_location="cpu", weights_only=True)
        om, mu, W = sd["omega"].numpy(), sd["mu"].numpy(), sd["W"].numpy()
        f0, bw = om * H * FS / (2 * np.pi), np.abs(mu) * H * FS / (2 * np.pi)
        ev = np.linalg.eigvals(np.diag(mu + 1j * om) + W)
        rows.append(dict(subject=int(f.split("_S")[-1][:-3]), f0_hz_min=float(f0.min()), f0_hz_max=float(f0.max()),
                         bw_hz_min=float(bw.min()), bw_hz_max=float(bw.max()), max_re_eig=float(ev.real.max()),
                         omega=om.tolist(), mu=mu.tolist()))
    return rows


def response(omega, mu):
    lam = (np.asarray(mu) + 1j * np.asarray(omega)) * H
    f = np.linspace(4, 124, 481)
    Om = 2 * np.pi * f / FS
    Hp = 1 / (1 - np.exp(lam)[:, None] * np.exp(-1j * Om)[None])
    Hn = 1 / (1 - np.exp(lam)[:, None] * np.exp(+1j * Om)[None])
    g = 20 * np.log10(np.abs(Hp))
    sel = 20 * np.log10(np.abs(Hp) / np.abs(Hn))
    P = (np.abs(Hp) ** 2 * (1 / f)[None]).mean(0)
    cum = np.cumsum(P) / P.sum()
    at = lambda hz: int(np.searchsorted(f, hz))
    return dict(gain_db={str(hz): float(g[:, at(hz)].mean()) for hz in (4, 10, 30, 100)},
                gain_spread_db={str(hz): float(np.ptp(g[:, at(hz)])) for hz in (4, 10, 30)},
                min_correlation_between_gain_curves=float(np.corrcoef(g).min()),
                plus_minus_selectivity_db={str(hz): float(sel[:, at(hz)].mean()) for hz in (4, 10, 30)},
                selectivity_of_a_10hz_resonator_db=float(20 * np.log10(20 / 0.3)),
                output_power_share_for_1_over_f_input=dict(below_8hz=float(cum[at(8)]), below_13hz=float(cum[at(13)]),
                                                           above_30hz=float(1 - cum[at(30)])))


def behaviour():
    from scipy import signal
    from resonet_sync._vendor import rn
    from resonet_sync.screens import calib_p1 as CP
    torch.set_num_threads(1)
    diff = json.loads(CP.DIFF_FILE.read_text())["chosen"]
    X = CP.make_dataset("G1", CP.PILOT_SEED, diff)[0][:96]
    h = signal.firwin(825, 4.0, pass_zero=False, fs=FS)                  # 4 Hz zero-phase FIR, like the HGD cache
    X = signal.filtfilt(h, [1.0], np.pad(X, ((0, 0), (0, 0), (900, 900)), mode="reflect"), axis=-1)[..., 900:-900]
    X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
    R = rn()

    def feats(m):
        with torch.no_grad():
            _, z = m(X, return_traj=True)
            u = z / (z.abs() + 1e-6)
            Cu = (torch.einsum("bti,btj->bij", u, u.conj()) / z.shape[1])[:, m.iu, m.ju]
        return torch.cat([Cu.real, Cu.imag], 1).numpy(), z
    m = R.FastResonanceNetwork(n_channels=16, n_osc=16, readout="sync", seed=42).eval()
    m1 = R.FastResonanceNetwork(n_channels=16, n_osc=16, readout="sync", seed=42).eval()
    with torch.no_grad():
        m1.omega.fill_(float(m.omega.mean())); m1.mu.fill_(float(m.mu.mean()))
    a, z = feats(m)
    b, _ = feats(m1)
    per = np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])])
    dphi = np.angle((z[:, 1:] * z[:, :-1].conj()).numpy())
    return dict(shared_frequency_model=dict(overall_feature_correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                                            per_feature_correlation_median=float(np.median(per)),
                                            per_feature_correlation_p10=float(np.quantile(per, 0.1))),
                instantaneous_frequency_hz=dict(median=float(np.median(dphi) * FS / (2 * np.pi)),
                                                q25=float(np.quantile(dphi, 0.25) * FS / (2 * np.pi)),
                                                q75=float(np.quantile(dphi, 0.75) * FS / (2 * np.pi)),
                                                fraction_rotating_backwards=float((dphi < 0).mean())))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT.parent / "results_hg"))
    a = ap.parse_args(argv)
    ck = checkpoints(a.ckpt_dir)
    out = dict(note="published checkpoints; fs 250 Hz (decim 2), h = 0.01; the data are band-passed 4-125 Hz",
               n_checkpoints=len(ck),
               trained_f0_hz=[min(r["f0_hz_min"] for r in ck), max(r["f0_hz_max"] for r in ck)] if ck else None,
               trained_bw_hz=[min(r["bw_hz_min"] for r in ck), max(r["bw_hz_max"] for r in ck)] if ck else None,
               init_f0_hz=[0.5 * H * FS / (2 * np.pi), 4.0 * H * FS / (2 * np.pi)],
               per_subject=[{k: v for k, v in r.items() if k not in ("omega", "mu")} for r in ck],
               response_subject_1=response(ck[0]["omega"], ck[0]["mu"]) if ck else None, behaviour=behaviour())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "published_forensics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_subject"}, indent=1))


if __name__ == "__main__":
    main()
