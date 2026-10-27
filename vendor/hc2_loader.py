"""hc2_loader.py — loader for CRCNS hc-2 (Mizuseki, Sirota, Pastalkova & Buzsáki 2009): rat
hippocampal CA1 spikes + 2-D position. Produces (obs=binned population spike-counts, vel, pos)
windows for spatial_decode's grid-cell path-integration decoding test on REAL neural data.

Neuroscope / hc-2 format (per the CRCNS hc-2 documentation):
  fs = 20000 Hz. Position (.whl) is sampled at fs/512 = 39.0625 Hz (one row per 512 samples).
  <base>.res.<k> : spike sample-times (int @20 kHz), one per line, for shank k.
  <base>.clu.<k> : first line = #clusters; then one cluster-id per spike (aligned to .res.<k>).
                   clusters 0 (mechanical artifact) and 1 (noise) are discarded; units are ids >= 2.
  <base>.whl     : columns  x1 y1 x2 y2  (two tracking LEDs, cm); -1 = LED undetected.
                   position = mean of the valid LEDs; missing frames are linearly interpolated.
Bin = one .whl frame (512 samples); obs[t,u] = spike count of unit u in frame t. Position is
min-max normalized to [0,1]^2; velocity = per-frame position difference (the self-motion the grid
path-integrates). Point `base` at a session prefix, e.g. .../ec013.527/ec013.527 .
"""
import glob, os, numpy as np

FS = 20000
WHL_DS = 512


def _load_shank(base, k):
    res = np.loadtxt(f"{base}.res.{k}", dtype=np.int64)
    clu = np.loadtxt(f"{base}.clu.{k}", dtype=np.int64)
    res = np.atleast_1d(res); clu = np.atleast_1d(clu)
    clu = clu[1:]                                    # drop the leading cluster-count line
    assert len(clu) == len(res), f"shank {k}: clu/res length mismatch"
    keep = clu >= 2                                  # discard artifact(0) + noise(1)
    return res[keep], clu[keep]


def load_session(base):
    shanks = sorted({int(f.split(".res.")[1]) for f in glob.glob(base + ".res.*")})
    if not shanks:
        raise FileNotFoundError(f"no {base}.res.* files")
    times, units = [], []
    for k in shanks:
        res, clu = _load_shank(base, k)
        times.append(res)
        units.append(clu + k * 1000)                 # globally-unique unit ids across shanks
    times = np.concatenate(times); units = np.concatenate(units)
    uids = {u: i for i, u in enumerate(sorted(set(units.tolist())))}
    U = len(uids)

    whl = np.loadtxt(base + ".whl")
    whl = np.atleast_2d(whl); n = len(whl)
    pos = np.full((n, 2), np.nan)
    for i, row in enumerate(whl):
        leds = [(row[0], row[1]), (row[2], row[3])]
        valid = [(x, y) for x, y in leds if x >= 0 and y >= 0]
        if valid:
            pos[i] = np.mean(valid, 0)
    for c in range(2):                               # interpolate missing frames
        col = pos[:, c]; good = ~np.isnan(col)
        if good.any():
            col[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), col[good])
        pos[:, c] = col

    frame = (times // WHL_DS).astype(int)
    obs = np.zeros((n, U), dtype=np.float32)
    for t, u in zip(frame, units):
        if 0 <= t < n:
            obs[t, uids[u]] += 1.0

    rng = np.ptp(pos, 0); rng[rng == 0] = 1.0
    pos = ((pos - pos.min(0)) / rng).astype(np.float32)
    vel = np.zeros_like(pos); vel[1:] = pos[1:] - pos[:-1]
    return obs, vel.astype(np.float32), pos, U


def make_windows(obs, vel, pos, S, stride=None):
    stride = stride or S
    idx = range(0, len(obs) - S, stride)
    O = np.stack([obs[i:i + S] for i in idx])
    V = np.stack([vel[i:i + S] for i in idx])
    P = np.stack([pos[i:i + S] for i in idx])
    return O, V, P


def split_and_window(obs, vel, pos, S, occ, train_frac=0.7, stride=None, run_quantile=0.6):
    """Leak-free temporal holdout: the RAW timeline is split into train/test BEFORE windowing, then
    each half is windowed separately -- no window in either half can share a single raw frame with a
    window in the other half, regardless of the overlapping stride used within each half.

    (A prior version windowed the whole session first with an overlapping stride, S//3, then cut the
    resulting WINDOW array 70/30. Since adjacent windows overlap by construction, the last training
    window and first test window could -- and on real sessions, did -- share raw frames, occasionally
    reaching into the scored occluded region. Found independently by three critics reviewing this
    project's first full hc-2 sweep, one of whom demonstrated the byte-identical overlap directly on
    real session data.)

    Returns (Otr, Vtr, Ptr, Ote, Vte, Pte), filtered to "running" windows (peak position displacement
    during the occlusion window above the `run_quantile` percentile). The threshold is computed from
    the TRAIN split only and applied to both, so no test-derived statistic influences either split's
    inclusion."""
    stride = stride or S // 3
    raw_cut = int(train_frac * len(obs))
    Otr, Vtr, Ptr = make_windows(obs[:raw_cut], vel[:raw_cut], pos[:raw_cut], S, stride=stride)
    Ote, Vte, Pte = make_windows(obs[raw_cut:], vel[raw_cut:], pos[raw_cut:], S, stride=stride)
    d_tr = np.sqrt(((Ptr[:, occ:] - Ptr[:, occ - 1:occ]) ** 2).sum(-1)).max(1)
    thresh = np.quantile(d_tr, run_quantile)
    d_te = np.sqrt(((Pte[:, occ:] - Pte[:, occ - 1:occ]) ** 2).sum(-1)).max(1)
    keep_tr, keep_te = d_tr > thresh, d_te > thresh
    return Otr[keep_tr], Vtr[keep_tr], Ptr[keep_tr], Ote[keep_te], Vte[keep_te], Pte[keep_te]


def _selftest():
    import tempfile
    d = tempfile.mkdtemp(); base = os.path.join(d, "sess")
    # shank 1: 3 clusters (0,1 discarded; unit 2 kept), spikes at samples 100,700,1300
    open(base + ".res.1", "w").write("100\n700\n1300\n")
    open(base + ".clu.1", "w").write("3\n2\n1\n2\n")     # spike0->u2, spike1->noise, spike2->u2
    # 4 position frames (512 samples each): frame0 spike@100, frame1 spike@700, frame2 spike@1300
    open(base + ".whl", "w").write("0 0 0.1 0.1\n1 1 1.1 1.1\n-1 -1 2 2\n3 3 -1 -1\n")
    obs, vel, pos, U = load_session(base)
    O, V, P = make_windows(obs, vel, pos, S=2)
    assert obs.shape == (4, 1) and U == 1, (obs.shape, U)          # one real unit (noise/artifact dropped)
    assert obs[0, 0] == 1 and obs[1, 0] == 0 and obs[2, 0] == 1    # spike@700 was noise -> frame1 empty
    assert not np.isnan(pos).any() and pos.min() >= 0 and pos.max() <= 1
    assert O.shape == (1, 2, 1)
    print(f"hc2_loader self-test PASSED: obs{obs.shape} units={U} pos[0]={pos[0]} vel range ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        obs, vel, pos, U = load_session(sys.argv[1])
        print(f"loaded {sys.argv[1]}: {obs.shape[0]} frames, {U} units, pos range "
              f"{pos.min(0)}..{pos.max(0)}")
    else:
        _selftest()
