"""Is there a fixed timing offset between channels (which would leak zero-lag covariance into the imaginary
cross-spectrum)? A pure delay d gives a cross-spectral phase -2 pi f d: linear in f through zero. Fit that slope for
every channel against the average of all channels over 30-120 Hz, where a delay would show most. Training trials
only. The estimate is an upper bound on hardware offsets, because genuine physiological lags are included.

  python scripts/channel_delay_check.py --cache /path/to/hgd_dev_cache [--subjects 1,3]"""
import argparse

import numpy as np
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="development cache (scripts/build_hgd_dev_cache.py)")
ap.add_argument("--subjects", default="1,3")
a = ap.parse_args()
for s in [int(v) for v in a.subjects.split(",")]:
    d = np.load(f'{a.cache}/subj_{s}.npz')
    X = d['X_tr'][:200].astype(np.float32)                    # (trials, 128, 2250) at 500 Hz
    ref = X.mean(1)
    f, _ = signal.welch(ref[0], fs=500, nperseg=500)
    band = (f >= 30) & (f <= 120) & ~((f > 47) & (f < 53)) & ~((f > 97) & (f < 103))
    delays, coh_m = [], []
    for c in range(128):
        _, Pxy = signal.csd(X[:, c], ref, fs=500, nperseg=500, axis=-1)
        _, Pxx = signal.welch(X[:, c], fs=500, nperseg=500, axis=-1)
        _, Pyy = signal.welch(ref, fs=500, nperseg=500, axis=-1)
        S, coh = Pxy.mean(0), np.abs(Pxy.mean(0)) ** 2 / (Pxx.mean(0) * Pyy.mean(0))
        ph = np.angle(S[band] * np.sign(S[band].real))          # fold polarity; phase near 0 for zero-lag mixtures
        w = coh[band]
        slope = np.sum(w * ph * f[band]) / np.sum(w * f[band] ** 2)      # phase = slope * f (through the origin)
        delays.append(-slope / (2 * np.pi)); coh_m.append(w.mean())
    dl = np.array(delays) * 1e6
    k = np.arange(128)
    print(f'S{s}: fitted delay per channel (microseconds): median {np.median(dl):+.1f}, IQR [{np.quantile(dl,.25):+.1f}, {np.quantile(dl,.75):+.1f}], '
          f'5-95% [{np.quantile(dl,.05):+.1f}, {np.quantile(dl,.95):+.1f}], max |d| {np.abs(dl).max():.0f}; corr with channel index {np.corrcoef(k, dl)[0,1]:+.2f}; mean coherence {np.mean(coh_m):.2f}')
    print('    a 100-microsecond offset would give %.1f deg at 13 Hz and %.1f deg at 80 Hz' % (360 * 13 * 1e-4, 360 * 80 * 1e-4))
