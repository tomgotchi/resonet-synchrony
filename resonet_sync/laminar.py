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
