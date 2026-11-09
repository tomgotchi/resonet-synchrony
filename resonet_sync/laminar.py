"""Laminar CA1 forward model, CSD, layer localisation and generator-mix statistics (review board: the H3 reframing).

The board ruled that "coupling between LFP sites" is not a well-defined construct (LFP = lead field x shared
generators). The replacement hypothesis is about the **generator mix**: how strongly the CA3-driven (stratum radiatum)
and EC3-driven (stratum lacunosum-moleculare) theta current sinks contribute, wheel vs maze.

Forward model (1-D, homogeneous conductor; standard disc-free point approximation with lateral radius R):
    phi(z, t) = sum_k I_k(t) / (4 pi sigma) / sqrt((z - z_k)^2 + R^2)
Each generator is a sink/source dipole pair along the somato-dendritic axis, driven by its own narrowband theta
current. Optional nuisances: sawtooth asymmetry (2f harmonic), slow frequency drift (speed), 1/f background noise.

CSD estimate (standard second spatial difference, Vaknin-padded ends):
    CSD_i = -sigma (phi_{i-1} - 2 phi_i + phi_{i+1}) / h^2
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SIGMA = 0.3          # S/m
R_LAT = 50e-6        # m, lateral source radius (smooths the point-source singularity)


# ---------------------------------------------------------------- forward model
def theta_source(n, fs, f0=8.0, bw=1.0, phase=0.0, asym=0.0, drift_hz=0.0, rng=None):
    """Unit-variance narrowband theta current. asym > 0 adds a sawtooth-like 2f harmonic in fixed phase relation;
    drift_hz adds a slow random-walk frequency modulation (speed-locked frequency changes)."""
    rng = np.random.default_rng(rng)
    t = np.arange(n) / fs
    inst_f = f0 + drift_hz * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    ph = 2 * np.pi * np.cumsum(inst_f) / fs + phase
    pole = np.exp(-2 * np.pi * bw / fs)                                   # slow amplitude envelope (bw Hz)
    env = signal.lfilter([1 - pole], [1, -pole], rng.standard_normal(n))
    a = np.clip(1.0 + 0.3 * env / (env.std() + 1e-12), 0.1, None)
    s = a * (np.cos(ph) + asym * np.cos(2 * ph - np.pi / 2))
    return (s - s.mean()) / s.std()


def forward(z_sites, generators, sigma=SIGMA, R=R_LAT):
    """generators: list of (z_sink, z_source, current (T,)) -> phi (n_sites, T) in volts (arbitrary scale)."""
    phi = np.zeros((len(z_sites), len(generators[0][2])))
    for z_sink, z_src, I in generators:
        for zk, sgn in ((z_sink, -1.0), (z_src, +1.0)):
            phi += sgn * np.outer(1.0 / np.sqrt((z_sites - zk) ** 2 + R ** 2), I) / (4 * np.pi * sigma)
    return phi


def pink_noise(shape, fs, exponent=1.0, rng=None):
    rng = np.random.default_rng(rng)
    N = np.fft.rfft(rng.standard_normal(shape), axis=-1)
    f = np.fft.rfftfreq(shape[-1], 1 / fs); f[0] = f[1]
    x = np.fft.irfft(N * f ** (-exponent / 2), n=shape[-1], axis=-1)
    return x / x.std(axis=-1, keepdims=True)


# Default CA1 geometry (depth in m below the alveus; one shank)
DEPTHS = dict(pyramidal=150e-6, radiatum=350e-6, lm=550e-6)


def simulate_laminar(T_s=60.0, fs=1250.0, spacing=50e-6, n_sites=16, w_ca3=1.0, w_ec=1.0, ec_phase_deg=180.0,
                     f0=8.0, asym=0.0, drift_hz=0.0, noise_rel=0.05, noise_exp=1.0, ripple_amp=0.3, seed=0,
                     z0=0.0):
    """Two generators (CA3 -> radiatum sink with perisomatic return source; EC3 -> l-m sink with return source in
    radiatum/pyramidal direction) with their own theta currents. Returns (phi (n_sites, T), z_sites)."""
    rng = np.random.default_rng(seed)
    n = int(T_s * fs)
    z = z0 + np.arange(n_sites) * spacing
    I_ca3 = w_ca3 * theta_source(n, fs, f0, asym=asym, drift_hz=drift_hz, rng=rng.integers(1 << 31))
    I_ec = w_ec * theta_source(n, fs, f0, phase=np.deg2rad(ec_phase_deg), asym=asym, drift_hz=drift_hz,
                               rng=rng.integers(1 << 31))
    gens = [(DEPTHS["radiatum"], DEPTHS["pyramidal"], I_ca3), (DEPTHS["lm"], DEPTHS["radiatum"] + 50e-6, I_ec)]
    if ripple_amp:                       # sharp-wave ripples: 200 Hz bursts from a tight perisomatic dipole
        burst = np.zeros(n)
        for c in rng.integers(0, n - int(0.08 * fs), max(1, int(T_s * 0.5))):
            k = np.arange(int(0.08 * fs))
            burst[c + k] += np.hanning(len(k)) * np.sin(2 * np.pi * 200.0 * k / fs)
        gens.append((DEPTHS["pyramidal"] - 15e-6, DEPTHS["pyramidal"] + 15e-6, ripple_amp * 20 * burst))
    phi = forward(z, gens)
    phi = phi / phi.std()
    phi = phi + noise_rel * pink_noise(phi.shape, fs, noise_exp, rng.integers(1 << 31))
    return phi, z


# ---------------------------------------------------------------- analysis
def csd(phi, spacing, sigma=SIGMA, vaknin=True):
    """Standard CSD (second spatial difference). phi: (n_sites, T)."""
    p = np.vstack([phi[:1], phi, phi[-1:]]) if vaknin else phi
    c = -sigma * (p[:-2] - 2 * p[1:-1] + p[2:]) / spacing ** 2
    return c


def band(x, fs, lo, hi, order=4):
    return signal.sosfiltfilt(signal.butter(order, (lo, hi), "band", fs=fs, output="sos"), x, axis=-1)


def localise_layers(phi, fs, theta=(6.0, 10.0), ripple=(150.0, 250.0)):
    """Train-data layer localisation on one shank: pyramidal = max ripple-band power; theta reversal = the first site
    below the pyramidal layer whose theta phase differs by > 90 deg from the pyramidal site. Returns indices (or None)."""
    rip = band(phi, fs, *ripple).var(-1) if fs / 2 > ripple[1] else None
    th = signal.hilbert(band(phi, fs, *theta), axis=-1)
    i_pyr = int(np.argmax(rip)) if rip is not None else int(np.argmax(np.abs(th).mean(-1)))
    dphi = np.angle(np.mean(th * th[i_pyr].conj(), axis=-1))
    below = [i for i in range(i_pyr + 1, len(phi)) if abs(dphi[i]) > np.pi / 2]
    return dict(pyramidal=i_pyr, theta_reversal=below[0] if below else None,
                theta_phase_deg=np.rad2deg(dphi).tolist(), theta_amp=np.abs(th).mean(-1).tolist())


def coverage_ok(layers, spacing, min_span=300e-6):
    """Feasibility gate (pre-registered): the shank must contain the pyramidal layer AND a theta phase reversal, and
    span >= min_span below the pyramidal layer, or the radiatum and l-m sinks cannot be separated by CSD."""
    return layers["theta_reversal"] is not None and (len(layers["theta_phase_deg"]) - 1 - layers["pyramidal"]) * spacing >= min_span


def generator_mix(phi, fs, spacing, i_rad, i_lm, theta=(6.0, 10.0)):
    """Generator-mix statistics from theta-band CSD at the radiatum and l-m sink sites (chosen on train data):
    log power ratio (l-m / radiatum) and their mean phase difference (deg)."""
    c = signal.hilbert(band(csd(phi, spacing), fs, *theta), axis=-1)
    r, l = c[i_rad], c[i_lm]
    return dict(log_power_ratio_lm_over_rad=float(np.log(np.mean(np.abs(l) ** 2) / np.mean(np.abs(r) ** 2))),
                phase_lm_minus_rad_deg=float(np.rad2deg(np.angle(np.mean(l * r.conj())))))


def sink_sites(z_sites):
    """Indices of the sites nearest the radiatum and l-m sink depths (known geometry; on real data these come from
    the train-split CSD profile)."""
    return int(np.argmin(np.abs(z_sites - DEPTHS["radiatum"]))), int(np.argmin(np.abs(z_sites - DEPTHS["lm"])))


# =============================================================================================== #
# Gate 1 v2 (review board 2026-11-09). v1 above is kept unchanged so the committed dry runs reproduce.
#  - disc sources (delta-iCSD form, radius 0.5 mm) instead of 50 um point sources;
#  - ripple = perisomatic tripole (LFP peaks AT the pyramidal layer; v1's dipole had a null there);
#  - theta generators share one septal phase path (common speed-locked drift); CA3-EC coherence is a parameter
#    (stationary OU phase noise on the EC drive); optional perisomatic and DG theta generators;
#  - EC-only asymmetry with the fundamental held fixed; noise in absolute units (no renormalisation);
#  - per-session probe offset, layer-thickness factor and per-site gain;
#  - generator identification by fitting two sink/source templates to the neutral theta-CSD cross-spectrum,
#    then generator power by projection (generator_mix_fit), instead of two picked sites.
# =============================================================================================== #
R_DISC = 0.5e-3
# depths below the pyramidal layer (m), before the thickness factor
GEN2 = dict(ca3=(200e-6, 0.0), ec=(400e-6, 250e-6), peri=(60e-6, 0.0), dg=(550e-6, 750e-6))   # (sink, source)
PYR_DEPTH = 150e-6          # pyramidal layer below the alveus


def disc_forward(z_sites, sources, sigma=SIGMA, R=R_DISC):
    """sources: list of (z, current (T,)) of CSD discs (current per area; sinks negative) -> phi (n_sites, T).
    phi(z) = 1/(2 sigma) sum_j I_j (sqrt((z - z_j)^2 + R^2) - |z - z_j|)."""
    phi = 0.0
    for zj, I in sources:
        d = z_sites - zj
        phi = phi + np.outer((np.sqrt(d ** 2 + R ** 2) - np.abs(d)) / (2 * sigma), I)
    return np.asarray(phi)


def _ou(n, fs, tau, var, rng):
    a = np.exp(-1.0 / (fs * tau))
    return np.sqrt(var) * signal.lfilter([np.sqrt(1 - a * a)], [1, -a], rng.standard_normal(n))


_NOISE_REF = {}


def simulate_laminar_v2(T_s=120.0, fs=1250.0, spacing=50e-6, n_sites=16, top=-150e-6, thick=1.0, gain_sd=0.03,
                        w_ca3=1.0, w_ec=1.0, ec_phase_deg=180.0, plv=0.95, f0=8.0, drift_hz=0.0, asym_ec=0.0,
                        w_peri=0.0, w_dg=0.0, rad_shift=0.0, noise_rel=0.05, noise_exp=1.0, ripple_amp=0.3,
                        seed=0, return_truth=False, gains=None):
    """top: depth of the first site relative to the pyramidal layer (negative = above it). Returns (phi, z_sites)
    with z_sites relative to the alveus; with return_truth also the true sink depths."""
    rng = np.random.default_rng(seed)
    n = int(T_s * fs)
    z = PYR_DEPTH + top + np.arange(n_sites) * spacing
    inst_f = f0 + drift_hz * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)          # ONE septal frequency path
    ph = 2 * np.pi * np.cumsum(inst_f) / fs

    def env():
        pole = np.exp(-2 * np.pi * 1.0 / fs)
        e = signal.lfilter([1 - pole], [1, -pole], rng.standard_normal(n))
        return np.clip(1.0 + 0.3 * e / (e.std() + 1e-12), 0.1, None)

    psi = _ou(n, fs, 0.5, -2 * np.log(plv), rng) if plv < 1 else 0.0
    lag = np.deg2rad(ec_phase_deg)
    cur = dict(ca3=w_ca3 * env() * np.cos(ph),
               ec=w_ec * env() * (np.cos(ph + lag + psi) + asym_ec * np.cos(2 * (ph + lag + psi) - np.pi / 2)))
    if w_peri:
        cur["peri"] = w_peri * env() * np.cos(ph + np.pi / 2)
    if w_dg:
        cur["dg"] = w_dg * env() * np.cos(ph + np.deg2rad(200.0))
    srcs, truth = [], {}
    for g, I in cur.items():
        zs, zr = GEN2[g]
        zs = PYR_DEPTH + thick * (zs - (rad_shift if g == "ca3" else 0.0))
        zr = PYR_DEPTH + thick * zr
        srcs += [(zs, -I), (zr, +I)]
        truth[g] = zs
    if ripple_amp:
        burst = np.zeros(n)
        k = np.arange(int(0.08 * fs))
        for c in rng.integers(0, n - len(k), max(1, int(T_s * 0.5))):
            burst[c + k] += np.hanning(len(k)) * np.sin(2 * np.pi * 200.0 * k / fs)
        r = ripple_amp * 3 * burst
        srcs += [(PYR_DEPTH, -2 * r), (PYR_DEPTH - 60e-6, r), (PYR_DEPTH + 60e-6, r)]
    phi = disc_forward(z, srcs)
    key = (round(spacing, 9), n_sites)
    if key not in _NOISE_REF:            # absolute noise scale: theta LFP std of the default session, fixed
        ref = disc_forward(z, [(PYR_DEPTH + GEN2["ca3"][0], -np.cos(ph)), (PYR_DEPTH, np.cos(ph))])
        _NOISE_REF[key] = float(ref.std())
    g = (1 + gain_sd * rng.standard_normal(n_sites)) if gains is None else np.asarray(gains)   # same probe: pass
    phi = phi * g[:, None]                                                                    # one vector per session
    phi = phi + noise_rel * _NOISE_REF[key] * pink_noise(phi.shape, fs, noise_exp, rng.integers(1 << 31))
    return (phi, z, truth) if return_truth else (phi, z)


def dipole_templates(z_sites, spacing, depth_step=10e-6, offsets=(-300e-6, -200e-6, -150e-6, -100e-6, 150e-6,
                                                                   250e-6, 350e-6)):
    """CSD-estimate templates (the same csd() as the data) of unit sink/source disc pairs. Sinks on a fine depth
    grid over the shank; the return source at sink + offset (negative = above). Returns (T (sites, m), sinks)."""
    sinks = np.arange(z_sites[0], z_sites[-1] + 1e-12, depth_step)
    cols, meta = [], []
    for zs in sinks:
        for off in offsets:
            t = csd(disc_forward(z_sites, [(zs, -np.ones(1)), (zs + off, np.ones(1))]), spacing)[:, 0]
            cols.append(t / np.linalg.norm(t)); meta.append((zs, zs + off))
    return np.array(cols).T, np.array(meta)


def theta_csd_cov(phi, fs, spacing, theta=(6.0, 10.0)):
    c = signal.hilbert(band(csd(phi, spacing), fs, *theta), axis=-1)
    return c @ c.conj().T / c.shape[1]


def fit_generators(S, Tm, meta, min_sep=150e-6):
    """Best pair of templates (sinks at least min_sep apart) spanning the 2-D subspace that captures most of the
    theta-CSD cross-spectrum S: maximise ||Q^H S Q||_F^2, Q an orthonormal basis of span(t_i, t_j).
    Returns indices (shallow sink first) and their (sink, source) depths."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _fit_generators(S, Tm, meta, min_sep)


def _fit_generators(S, Tm, meta, min_sep):
    G = (Tm.T @ Tm).real
    H = Tm.T @ S @ Tm
    gi, gj, gij = np.diag(G)[:, None], np.diag(G)[None, :], G
    det = gi * gj - gij ** 2
    hi, hj, hij = np.diag(H).real[:, None], np.diag(H).real[None, :], H
    # tr(G2^-1 H2 G2^-1 H2) for every pair, closed form with G2^-1 = adj(G2) / det
    a, b, d = gj / det, -gij / det, gi / det                            # G2^-1 = [[a, b], [b, d]]
    m11 = a * hi + b * hij.conj(); m12 = a * hij + b * hj
    m21 = b * hi + d * hij.conj(); m22 = b * hij + d * hj
    score = (m11 * m11 + m12 * m21 + m21 * m12 + m22 * m22).real
    sep = np.abs(meta[:, 0][:, None] - meta[:, 0][None, :]) >= min_sep
    score = np.where(sep & (det > 1e-6), score, -np.inf)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    if meta[i, 0] > meta[j, 0]:
        i, j = j, i
    return (i, j), meta[[i, j]]


def _template(z_sites, spacing, zs, zr):
    t = csd(disc_forward(z_sites, [(zs, -np.ones(1)), (zr, np.ones(1))]), spacing)[:, 0]
    return t / np.linalg.norm(t)


def fine_bank(z_sites, spacing, step=10e-6, src_above=(50e-6, 450e-6)):
    """Fine template bank: sinks every `step` over the shank, return source `src_above` above the sink (both CA3
    and EC3 return currents flow proximally), also every `step`."""
    cols, meta = [], []
    for zs in np.arange(z_sites[0], z_sites[-1] + 1e-12, step):
        for off in np.arange(src_above[0], src_above[1] + 1e-12, step):
            cols.append(_template(z_sites, spacing, zs, zs - off)); meta.append((zs, zs - off))
    return np.array(cols).T, np.array(meta)


def _pair_score(S, f, Tm):
    """||Q^H S Q||^2 for span(f, t) for every column t of Tm (f, t unit norm)."""
    g = Tm.T @ f.conj()                                     # <t, f> (real templates)
    det = 1 - np.abs(g) ** 2
    hff = np.real(f.conj() @ S @ f)
    hft = Tm.T @ (S.conj().T @ f.conj()).conj()           # t^H S f  (real t)
    htt = np.real(np.einsum("ij,ik,kj->j", Tm, S, Tm))
    hft = np.einsum("ij,i->j", Tm, S @ f)                  # t^T S f
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b, d = 1 / det, -g / det, 1 / det
        m11 = a * hff + b * hft.conj(); m12 = a * np.conj(hft) + b * htt
        m21 = b * hff + d * hft.conj(); m22 = b * np.conj(hft) + d * htt
        sc = (m11 * m11 + m12 * m21 + m21 * m12 + m22 * m22).real
    return np.where(det > 1e-6, sc, -np.inf)


def fit_generators_refined(S, z_sites, spacing, min_sep=150e-6, n_iter=4):
    """Coarse pair search (dipole_templates) followed by coordinate refinement over the fine bank."""
    Tc, mc = dipole_templates(z_sites, spacing)
    (i, j), _ = fit_generators(S, Tc, mc, min_sep)
    Tf, mf = fine_bank(z_sites, spacing)
    cur = [Tc[:, i], Tc[:, j]]
    dep = [mc[i], mc[j]]
    for _ in range(n_iter):
        for k in (0, 1):
            other = cur[1 - k]
            sc = _pair_score(S, other, Tf)
            sc = np.where(np.abs(mf[:, 0] - dep[1 - k][0]) >= min_sep, sc, -np.inf)
            m = int(np.argmax(sc))
            cur[k], dep[k] = Tf[:, m], mf[m]
    order = np.argsort([dep[0][0], dep[1][0]])
    return np.array([cur[o] for o in order]).T, np.array([dep[o] for o in order])


def generator_powers(S, Tm, idx):
    P = Tm[:, list(idx)]
    Pp = np.linalg.pinv(P)
    M = Pp @ S @ Pp.conj().T
    return np.real(np.diag(M))


def generator_mix_per_regime(fit_halves, eval_halves, fs, spacing):
    """Per-regime variant: each regime's generators are fitted on its own first half and its G computed on its own
    second half (robust to a generator depth that differs between regimes)."""
    z = np.arange(fit_halves[0].shape[0]) * spacing
    G, dep = [], []
    for f, e in zip(fit_halves, eval_halves):
        P, d = fit_generators_refined(theta_csd_cov(f, fs, spacing), z, spacing)
        pw = generator_powers(theta_csd_cov(e, fs, spacing), P, (0, 1))
        G.append(float(np.log(max(pw[1], 1e-30) / max(pw[0], 1e-30)))); dep.append(d[:, 0].tolist())
    return dict(G=G, sinks_rel_first_site=dep)


def generator_mix_fit(phi_neutral, regimes, fs, spacing):
    """Fit the two generator templates on neutral data, then log power ratio (deep / shallow sink = EC / CA3) of
    the projected generator powers in each regime. Returns dict(fit=depths, G=[per regime])."""
    z = np.arange(phi_neutral.shape[0]) * spacing
    P, depths = fit_generators_refined(theta_csd_cov(phi_neutral, fs, spacing), z, spacing)
    G = []
    for p in regimes:
        pw = generator_powers(theta_csd_cov(p, fs, spacing), P, (0, 1))
        G.append(float(np.log(max(pw[1], 1e-30) / max(pw[0], 1e-30))))
    return dict(sinks_rel_first_site=depths[:, 0].tolist(), sources_rel_first_site=depths[:, 1].tolist(), G=G)
