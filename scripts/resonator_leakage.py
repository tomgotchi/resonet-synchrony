"""Where does each v2 resonator's output power come from on real EEG? (review board 2026-11-13, Asante)

For the HGD configuration (12 bands, 4-100 Hz, constant Q) and the sensor-average power spectrum of the training
recordings, the share of a resonator's output power that comes from inside its own -3 dB band, for resonator
orders 1 (the original one-pole), 2 and 4 (cascaded poles, `CrossSpectralBank(order=...)`) and, as a reference, a
4th-order Butterworth band-pass on the same band. Training recordings only; no model is fitted.

  python scripts/resonator_leakage.py --cache DIR [--subjects 1-6]
Writes results/v2_probe/resonator_leakage.json."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync import data_hgd  # noqa: E402
from resonet_sync.run import P1, V2_HGD, load_config  # noqa: E402

ORDERS = (1, 2, 4)


def response2(fp, f0, bw, fs, order):
    """|H|^2 of the bank's resonator at +f and at -f (the image that a real input also excites)."""
    dec = 2 * np.pi * bw / fs / (np.sqrt(2 ** (1 / order) - 1) if order > 1 else 1.0)
    om, w0 = 2 * np.pi * fp / fs, 2 * np.pi * f0 / fs
    g = 1 - np.exp(-dec)
    hp = np.abs(g / (1 - np.exp(-dec + 1j * (w0 - om)))) ** (2 * order)
    hn = np.abs(g / (1 - np.exp(-dec + 1j * (w0 + om)))) ** (2 * order)
    return hp, hn


def shares(fp, P, fs, n_bands, band):
    f0 = np.geomspace(band[0], min(band[1], 0.45 * fs), n_bands)
    bw = f0 * ((f0[-1] / f0[0]) ** (1 / (n_bands - 1)) - 1) / 2
    rows = []
    for f, b in zip(f0, bw):
        inb = (fp >= f - b) & (fp <= f + b)
        row = dict(f0=float(f), half_bw=float(b))
        for o in ORDERS:
            hp, hn = response2(fp, f, b, fs, o)
            G = P * (hp + hn)
            row[f"order{o}"] = dict(in_band=float(G[inb].sum() / G.sum()), below_40Hz=float(G[fp < 40].sum() / G.sum()),
                                    image=float((P * hn).sum() / G.sum()))
        sos = signal.butter(4, (max(f - b, 0.5), min(f + b, 0.49 * fs)), "band", fs=fs, output="sos")
        Gb = P * np.abs(signal.sosfreqz(sos, worN=fp, fs=fs)[1]) ** 2
        row["butter4"] = dict(in_band=float(Gb[inb].sum() / Gb.sum()))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None); ap.add_argument("--subjects", default="1-6")
    a = ap.parse_args(argv)
    cache = a.cache or load_config().get("paths", {}).get("hgd_cache_hg")
    fs = data_hgd.FS_NATIVE / P1["decim"]
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    subs, psd = list(range(int(lo), int(hi) + 1)), []
    for s in subs:
        X, _, _, _ = data_hgd.load(cache, s, P1["decim"], P1["cap"], P1["seed"])        # test split never used
        fp, P = signal.welch(X, fs=fs, nperseg=int(2 * fs), axis=-1)
        psd.append(P.mean((0, 1)))
    P = np.mean(psd, 0)
    rows = shares(fp, P, fs, V2_HGD["n_bands"], V2_HGD["band"])
    out = ROOT / "results" / "v2_probe" / "resonator_leakage.json"
    out.write_text(json.dumps(dict(note="training recordings only; sensor- and subject-average Welch spectrum",
                                   subjects=subs, fs=fs, bands=rows), indent=1))
    print(" f0 Hz   +-bw | in-band share: order 1, 2, 4 | Butterworth-4 | from <40 Hz: order 1, 4")
    for r in rows:
        print(f"{r['f0']:6.1f} {r['half_bw']:5.1f} |  {r['order1']['in_band']:.2f}  {r['order2']['in_band']:.2f}  {r['order4']['in_band']:.2f}  |"
              f"  {r['butter4']['in_band']:.2f}  |  {r['order1']['below_40Hz']:.2f}  {r['order4']['below_40Hz']:.2f}")


if __name__ == "__main__":
    main()
