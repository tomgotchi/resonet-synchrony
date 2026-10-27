"""
Resonance Network (RN) for High-Gamma EEG Classification
=========================================================
Complex-valued, supercritical-Hopf oscillatory neural network with a
vectorized RK4 ODE solver, trained against an EEGNet ablation baseline on the
High-Gamma Dataset (HGD).

This is a faithful software implementation of the "Resonance Network
Implementation Guide", with the following diagnosed corrections to the guide
(the guide contained errors that would prevent it from running on the real data):

  * Data source: the guide says ".npz windows"; the raw store is actually .edf.
    We load the already-preprocessed per-subject cache in `subj_cache/subj_N.npz`
    (keys: X_tr, y_tr, X_te, y_te).
  * Channels: the guide says 44 sensors; the real data has 128 channels. The
    encoder (128 -> 64) and EEGNet spatial conv are sized to 128.
  * Normalization: data is mean-centred but not unit-variance. We apply
    leakage-safe per-channel z-scoring using TRAIN statistics only.
  * Intrinsic frequencies: the guide's parameter table says omega ~ N(1.0, 0.2),
    which collapses every oscillator to ~1 rad/step and destroys the diversity
    that the resonance premise depends on. We instead spread omega across a band
    of intrinsic frequencies (documented in ResonanceNetwork.__init__).
  * RK4 forcing: the guide's rk4_step signature passes a single I_t but also
    requires midpoint interpolation. We pass (I0, Imid, I1) so the time-varying
    driving force is interpolated correctly at t, t+h/2, t+h.
  * Time decimation: full 2250-step BPTT is intractable on an 8.6 GB laptop GPU.
    A configurable anti-aliased decimation factor keeps high-gamma content while
    making training feasible (calibrated by probe_timing()).

Dependencies: torch, numpy, scipy (+ mne only for the optional EDF path).
"""

from __future__ import annotations
import os
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import decimate, welch

# --------------------------------------------------------------------------- #
# Phase 1: Environment configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(r"C:\Users\asusd\Documents\RESEARCH\eeg_data\mne_data_backup")
# CACHE_DIR is overridable via env (RN_CACHE_DIR) so the high-gamma cache
# (subj_cache_hg) can be used without disturbing the default low-passed run.
CACHE_DIR = Path(os.environ.get("RN_CACHE_DIR",
                                r"C:\Users\asusd\Documents\RESEARCH\subj_cache"))
SFREQ = 500  # native HGD sampling rate (Hz)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("CUDA is NOT available to PyTorch. Falling back to CPU.")
        # Diagnostic: check if CUDA is even detected at a low level
        try:
            import subprocess
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if res.returncode == 0:
                print("Note: nvidia-smi detected a GPU, but PyTorch cannot use it.")
                print("This usually means a version mismatch between PyTorch CUDA and your drivers.")
            else:
                print("Note: nvidia-smi failed or not found. No NVIDIA GPU detected by the OS.")
        except Exception:
            pass
    return dev


# --------------------------------------------------------------------------- #
# Phase 2: Data loading, normalization, stratified split, augmentation
# --------------------------------------------------------------------------- #
def _balanced_subsample(y, max_per_class, seed=42):
    """Return indices for a class-balanced subset (<= max_per_class each)."""
    if max_per_class is None:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        idx.extend(ci[:max_per_class].tolist())
    idx = np.array(sorted(idx))
    return idx


def _chunked_decimate(X, q, chunk=100):
    """Anti-aliased FIR decimation along the time axis, processed in trial
    chunks so peak host RAM stays bounded (the full 880x128x2250 array + FIR
    copies would otherwise OOM a 16 GB machine)."""
    if q <= 1:
        return X.astype(np.float32)
    out = []
    for i in range(0, X.shape[0], chunk):
        out.append(decimate(X[i:i + chunk], q, axis=2, ftype="fir",
                            zero_phase=True).astype(np.float32))
    return np.concatenate(out, axis=0)


def load_subject(subject: int, decim: int = 4, max_train_per_class=120, seed=42):
    """Load one subject's cached arrays, optionally balance-subsample the train
    set, decimate the time axis, and per-channel z-score (train stats only).

    max_train_per_class caps training trials for tractability on this hardware
    (most subjects have ~880 trials = 1 GB). The cap is applied identically to
    whatever model trains on the returned data, so the RN-vs-EEGNet comparison
    stays fair; the full 160-trial test set is always kept intact.
    """
    with np.load(CACHE_DIR / f"subj_{subject}.npz") as d:
        X_tr, y_tr = d["X_tr"], d["y_tr"].astype(np.int64)
        X_te, y_te = d["X_te"].astype(np.float32), d["y_te"].astype(np.int64)
        sub = _balanced_subsample(y_tr, max_train_per_class, seed)
        X_tr = X_tr[sub].astype(np.float32)   # subsample BEFORE copy/decimate
        y_tr = y_tr[sub]

    X_tr = _chunked_decimate(X_tr, decim)
    X_te = _chunked_decimate(X_te, decim)

    # leakage-safe per-channel z-score: fit on the TRAIN-MINUS-VAL trials only
    # (the val fold is held out of the scaler too, matching the paper's stated
    # "computed from the training data" and removing the last trace of val
    # leakage into normalization). The val mask is deterministic (same seed) so
    # make_loaders carves off exactly these trials.
    vmask = stratified_trial_split(np.arange(len(y_tr)), y_tr, 0.15, seed)
    fit = X_tr[~vmask]
    mu = fit.mean(axis=(0, 2), keepdims=True)
    sd = fit.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    return X_tr, y_tr, X_te, y_te


def stratified_trial_split(trial_id_tr, y_tr, val_frac=0.15, seed=42):
    """Assign exactly `val_frac` of trials per class to a boolean val mask.

    Each cached row is an independent trial, so trial_id == row index here, but
    we keep the trial-grouped logic so no fragment of a val trial can leak.
    """
    rng = np.random.RandomState(seed)
    trial_id_tr = np.asarray(trial_id_tr)
    y_tr = np.asarray(y_tr)
    # map each unique trial -> its class
    trial_to_class = {int(t): int(y_tr[trial_id_tr == t][0]) for t in np.unique(trial_id_tr)}
    val_trials = set()
    for c in np.unique(y_tr):
        trials_c = np.array([t for t, cls in trial_to_class.items() if cls == c])
        rng.shuffle(trials_c)
        n_val = int(round(val_frac * len(trials_c)))
        val_trials.update(trials_c[:n_val].tolist())
    val_mask = np.array([trial_id_tr[i] in val_trials for i in range(len(trial_id_tr))])
    return val_mask


def augment_batch(bx):
    """Online augmentation: temporal roll +-10, additive noise, amplitude scale.

    bx: (B, C, T) float tensor on device. Returns augmented copy.
    """
    shift = int(torch.randint(-10, 11, (1,)).item())
    bx = torch.roll(bx, shifts=shift, dims=-1)
    bx = bx + 0.05 * torch.randn_like(bx)
    scale = torch.empty(bx.size(0), 1, 1, device=bx.device).uniform_(0.8, 1.2)
    return bx * scale


# --------------------------------------------------------------------------- #
# Phase 3: Vectorized RK4 solver
# --------------------------------------------------------------------------- #
def rk4_step(f, z, h, I0, Imid, I1):
    """One explicit RK4 step for dz/dt = f(z, I) with time-varying forcing.

    I0, Imid, I1 are the driving force at t, t+h/2, t+h respectively.
    """
    k1 = h * f(z, I0)
    k2 = h * f(z + 0.5 * k1, Imid)
    k3 = h * f(z + 0.5 * k2, Imid)
    k4 = h * f(z + k3, I1)
    return z + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# --------------------------------------------------------------------------- #
# Phase 4: Complex Resonance Network
# --------------------------------------------------------------------------- #
def complex_tanh(z):
    """Split tanh: apply tanh to real and imaginary parts separately.

    The guide specifies tanh() on the complex encoder output, but complex tanh
    has poles at +-i.pi/2 and blows up (observed: |I_enc| -> 2200), corrupting
    the pooled feature and producing NaN. Split-tanh is bounded and pole-free,
    the standard activation for complex-valued networks.
    """
    return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))


class ResonanceNetwork(nn.Module):
    """Complex supercritical-Hopf oscillator bank read out three ways:

      readout='power' : log mean |z|^2 per oscillator  (band-power proxy; the
                        guide's degraded readout -- the ablation control).
      readout='sync'  : the inter-oscillator PHASE-SYNCHRONIZATION signature,
                        i.e. the time-averaged pairwise complex coherence
                        C_ij = <u_i conj(u_j)>_t with u = z/|z|. This is the
                        "resonance BETWEEN oscillators" -- the novel definition,
                        and the one quantity a static band-power baseline cannot
                        compute. Gauge-invariant (uses relative phase only).
      readout='both'  : concatenation of the two.
    """
    def __init__(self, n_channels=128, n_osc=64, n_classes=4, h=0.01,
                 readout='sync', dropout=None, seed=42):
        super().__init__()
        self.N = n_osc
        self.C = n_channels
        self.h = h
        self.readout_mode = readout
        g = torch.Generator().manual_seed(seed)

        # --- oscillator parameters (real-valued, combined into complex) ---
        # omega: intrinsic frequency. Guide says N(1.0,0.2) but that collapses
        # all oscillators to one rhythm; we spread them across a band so the
        # network can tune distinct mu/beta/gamma-like resonances.
        omega0 = torch.empty(n_osc).uniform_(0.5, 4.0, generator=g)
        self.omega = nn.Parameter(omega0)
        # mu: Hopf bifurcation / growth rate. Moderately damped (mu<0) so the
        # linear operator diag(mu+i.omega)+W is contractive at init (resonator
        # mode). This damped band-pass regime empirically beat the lightly-damped
        # high-Q regime on this data (the abstract omega timescale does not align
        # with literal EEG-band Hz, so sharper tuning just adds variance).
        self.mu = nn.Parameter(torch.empty(n_osc).uniform_(-1.0, -0.4, generator=g))
        # beta: amplitude saturation (positive -> bounded limit cycle).
        self.beta = nn.Parameter(torch.ones(n_osc))
        # gamma: phase-amplitude coupling. Start at 0.
        self.gamma = nn.Parameter(torch.zeros(n_osc))

        # --- complex coupling matrix W (power coupling) ---
        # Guide says randn*0.1, but that gives spectral radius ~0.8 which
        # overwhelms the damping and diverges; we use 0.05 (radius ~0.4).
        Wr = torch.randn(n_osc, n_osc, generator=g) * 0.05
        Wi = torch.randn(n_osc, n_osc, generator=g) * 0.05
        self.W = nn.Parameter(torch.complex(Wr, Wi))
        self.amp_max = 5.0  # forward safety clamp on |z| to prevent RK4 overflow
        # learnable input gain: the per-step injection h*tanh(.) is tiny (~0.005),
        # so without this the seed/noise washes out class signal. Clamped in fwd.
        self.input_gain = nn.Parameter(torch.tensor(3.0))

        # --- complex real-to-complex encoder (C -> N) ---
        s = 0.5 / math.sqrt(n_channels)
        Er = torch.randn(n_osc, n_channels, generator=g) * s
        Ei = torch.randn(n_osc, n_channels, generator=g) * s
        self.W_enc = nn.Parameter(torch.complex(Er, Ei))
        self.b_enc = nn.Parameter(torch.zeros(n_osc, dtype=torch.complex64))

        # --- z(0) symmetry-breaking seed ---
        self.register_buffer("z0", torch.full((n_osc,), 0.01 + 0.01j, dtype=torch.complex64))

        # --- readout head ---
        # upper-triangle indices for the N x N coherence matrix (i < j pairs)
        iu, ju = torch.triu_indices(n_osc, n_osc, offset=1)
        self.register_buffer("iu", iu)
        self.register_buffer("ju", ju)
        n_pairs = iu.numel()                        # N(N-1)/2
        feat_dim = {'power': n_osc,
                    'sync': 2 * n_pairs,            # Re + Im of each coherence
                    'sync_imag': n_pairs,           # Im only = volume-conduction-robust
                    'sync_surrogate': 2 * n_pairs,  # capacity-matched phase-null control
                    'both': n_osc + 2 * n_pairs}[readout]
        # surrogate control: fixed per-oscillator circular phase offsets (fraction
        # of T). Destroys inter-oscillator phase alignment while preserving each
        # oscillator's own spectrum AND the feature dimensionality, so a 'sync' >
        # 'sync_surrogate' gap is genuine phase-locking, not readout capacity.
        self.register_buffer("surr_frac",
                             torch.rand(n_osc, generator=torch.Generator().manual_seed(seed + 1)))
        # sync features are high-dim (2016 pairs for N=64) -> regularise harder
        drop = dropout if dropout is not None else (0.5 if readout != 'power' else 0.2)
        self.feat_norm = nn.BatchNorm1d(feat_dim)
        self.feat_drop = nn.Dropout(drop)
        self.readout = nn.Linear(feat_dim, n_classes)

    def _f(self, z, I):
        lam = self.mu + 1j * self.omega            # (N,) complex
        sat = self.beta + 1j * self.gamma          # (N,) complex
        amp2 = z.real ** 2 + z.imag ** 2           # (B,N) real
        coupling = z @ self.W.t()                  # sum_k W_jk z_k
        return lam * z - sat * amp2 * z + coupling + I

    def forward(self, x, return_traj=False):
        # x: (B, C, T) real
        B, C, T = x.shape
        gain = self.input_gain.clamp(0.1, 10.0)
        xt = x.transpose(1, 2).to(torch.complex64)                     # (B,T,C)
        I_enc = gain * complex_tanh(xt @ self.W_enc.t() + self.b_enc)  # (B,T,N) complex
        z = self.z0.unsqueeze(0).expand(B, self.N).contiguous()

        want_coh = self.readout_mode in ('sync', 'sync_imag', 'both')
        want_surr = self.readout_mode == 'sync_surrogate'
        want_pow = self.readout_mode in ('power', 'both')

        traj = [] if return_traj else None
        pow_sum = z.new_zeros(B, self.N, dtype=torch.float32)
        coh = z.new_zeros(B, self.N, self.N) if want_coh else None   # complex
        u_traj = [] if want_surr else None
        for t in range(T - 1):
            I0 = I_enc[:, t, :]
            I1 = I_enc[:, t + 1, :]
            Imid = 0.5 * (I0 + I1)
            z = rk4_step(self._f, z, self.h, I0, Imid, I1)
            # safety projection: cap |z| so an explicit-RK4 overshoot on the
            # cubic term can never reach inf/nan (no-op while |z| < amp_max).
            amp = z.abs()
            scale = torch.clamp(amp, max=self.amp_max) / (amp + 1e-8)
            z = z * scale
            pow_sum = pow_sum + amp * amp
            if want_coh:
                # unit phasors -> pairwise complex coherence (running mean)
                u = z / (amp + 1e-6)
                coh = coh + u.unsqueeze(2) * u.unsqueeze(1).conj()   # (B,N,N)
            elif want_surr:
                # keep the unit-phasor trajectory; the per-oscillator circular
                # time-shift (below) needs the whole series, so we can't use the
                # running-mean accumulator here.
                u_traj.append(z / (amp + 1e-6))                      # (B,N) complex
            if return_traj:
                traj.append(z)

        feats = []
        if want_pow:
            # log mean resonance power per oscillator (band-power proxy)
            feats.append(torch.log(pow_sum / (T - 1) + 1e-6))
        if want_coh:
            # time-averaged complex coherence; keep upper-triangle pairs only.
            # |C_ij| is the phase-locking value; arg(C_ij) the mean phase lag.
            C = coh / (T - 1)
            Cu = C[:, self.iu, self.ju]                              # (B, P) complex
            feats.append(Cu.imag if self.readout_mode == 'sync_imag'
                         else torch.cat([Cu.real, Cu.imag], dim=1))  # (B, P) or (B, 2P)
        if want_surr:
            # phase-randomized surrogate: roll each oscillator's phasor series by a
            # fixed per-oscillator offset, destroying cross-oscillator alignment but
            # preserving each spectrum and the 2P feature dim (capacity control).
            U = torch.stack(u_traj, dim=1)                           # (B, T-1, N) complex
            Tm = U.shape[1]
            sh = (self.surr_frac * Tm).long()
            U = torch.stack([torch.roll(U[:, :, i], int(sh[i]), dims=1)
                             for i in range(self.N)], dim=2)         # (B, T-1, N)
            C = torch.einsum('bti,btj->bij', U, U.conj()) / Tm       # (B,N,N) coherence
            Cu = C[:, self.iu, self.ju]                              # (B, P) complex
            feats.append(torch.cat([Cu.real, Cu.imag], dim=1))      # (B, 2P)
        feat = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        logits = self.readout(self.feat_drop(self.feat_norm(feat)))
        if return_traj:
            return logits, torch.stack(traj, dim=1)            # (B, T-1, N) complex
        return logits


# --------------------------------------------------------------------------- #
# Phase 4b: Fast LINEAR resonator bank (parallel, FFT) -- the efficient arm
# --------------------------------------------------------------------------- #
class FastResonanceNetwork(nn.Module):
    """Diagonal *linear* complex oscillators solved in parallel.

    Drops the nonlinear saturation `(beta+i.gamma)|z|^2 z` and the recurrent
    coupling `Wz`, leaving a bank of independent driven damped oscillators
        dz_k/dt = (mu_k + i.omega_k) z_k + g.I_k(t),   mu_k < 0.
    Each oscillator's impulse response is a damped complex sinusoid, so the whole
    time evolution is a causal convolution -> computed with one FFT instead of a
    T-step Python loop (~50-100x faster, no BPTT pathologies, unconditionally
    stable for mu<0). This is the LinOSS/coRNN-style efficient + "linear /
    stimulus-induced synchronization" arm of the dynamics ablation. The readout
    (sync / power / both) is identical to ResonanceNetwork, so swapping this in
    isolates "does the nonlinear coupling earn its cost?".
    """
    def __init__(self, n_channels=128, n_osc=64, n_classes=4, h=0.01,
                 readout='sync', dropout=None, seed=42):
        super().__init__()
        self.N = n_osc; self.h = h; self.readout_mode = readout
        g = torch.Generator().manual_seed(seed)
        self.omega = nn.Parameter(torch.empty(n_osc).uniform_(0.5, 4.0, generator=g))
        self.mu = nn.Parameter(torch.empty(n_osc).uniform_(-1.0, -0.4, generator=g))
        self.input_gain = nn.Parameter(torch.tensor(3.0))
        s = 0.5 / math.sqrt(n_channels)
        self.W_enc = nn.Parameter(torch.complex(
            torch.randn(n_osc, n_channels, generator=g) * s,
            torch.randn(n_osc, n_channels, generator=g) * s))
        self.b_enc = nn.Parameter(torch.zeros(n_osc, dtype=torch.complex64))
        iu, ju = torch.triu_indices(n_osc, n_osc, offset=1)
        self.register_buffer("iu", iu); self.register_buffer("ju", ju)
        n_pairs = iu.numel()
        feat_dim = {'power': n_osc, 'sync': 2 * n_pairs, 'sync_imag': n_pairs,
                    'sync_surrogate': 2 * n_pairs, 'both': n_osc + 2 * n_pairs}[readout]
        # surrogate control: fixed per-oscillator phase offsets (fraction of T)
        # to destroy inter-oscillator phase alignment while preserving each
        # oscillator's spectrum AND the feature dimensionality. If 'sync' beats
        # 'sync_surrogate', the advantage is genuinely phase-locking, not capacity.
        self.register_buffer("surr_frac",
                             torch.rand(n_osc, generator=torch.Generator().manual_seed(seed + 1)))
        drop = dropout if dropout is not None else (0.2 if readout == 'power' else 0.5)
        self.feat_norm = nn.BatchNorm1d(feat_dim)
        self.feat_drop = nn.Dropout(drop)
        self.readout = nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_traj=False):
        B, C, T = x.shape
        gain = self.input_gain.clamp(0.1, 10.0)
        xt = x.transpose(1, 2).to(torch.complex64)                     # (B,T,C)
        I_enc = gain * complex_tanh(xt @ self.W_enc.t() + self.b_enc)   # (B,T,N)

        mu = self.mu.clamp(max=-0.05)                  # enforce stable decay |a|<1
        lam = (mu + 1j * self.omega).to(torch.complex64)               # (N,)
        n = torch.arange(T, device=x.device, dtype=torch.float32)
        kernel = torch.exp(lam.unsqueeze(0) * self.h * n.unsqueeze(1))  # (T,N) complex
        L = 1
        while L < 2 * T:
            L *= 2
        # causal linear convolution via FFT (parallel over the whole sequence)
        Z = torch.fft.ifft(torch.fft.fft(I_enc, n=L, dim=1) *
                           torch.fft.fft(kernel, n=L, dim=0).unsqueeze(0), dim=1)
        z = Z[:, :T, :]                                                # (B,T,N) complex

        amp = z.abs()
        feats = []
        if self.readout_mode in ('power', 'both'):
            feats.append(torch.log((amp * amp).mean(dim=1) + 1e-6))    # (B,N)
        if self.readout_mode in ('sync', 'sync_imag', 'sync_surrogate', 'both'):
            u = z / (amp + 1e-6)                                       # unit phasors
            if self.readout_mode == 'sync_surrogate':
                # phase-randomized surrogate: independent circular time-shift per
                # oscillator -> destroys inter-oscillator phase alignment, keeps
                # each oscillator's own spectrum and the feature dimensionality.
                sh = (self.surr_frac * T).long()
                u = torch.stack([torch.roll(u[:, :, i], int(sh[i]), dims=1)
                                 for i in range(self.N)], dim=2)
            Cmat = torch.einsum('bti,btj->bij', u, u.conj()) / T       # (B,N,N) coherence
            Cu = Cmat[:, self.iu, self.ju]                            # (B,P) complex
            feats.append(Cu.imag if self.readout_mode == 'sync_imag'
                         else torch.cat([Cu.real, Cu.imag], dim=1))
        feat = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        logits = self.readout(self.feat_drop(self.feat_norm(feat)))
        if return_traj:
            return logits, z
        return logits


# --------------------------------------------------------------------------- #
# Phase 5: Ablated EEGNet baseline
# --------------------------------------------------------------------------- #
class EEGNet_Baseline(nn.Module):
    def __init__(self, n_classes=4, n_channels=128, n_samples=450,
                 F1=8, D=2, F2=16, kern1=None, dropout=0.5):
        super().__init__()
        # temporal kernel ~ quarter second at the (decimated) rate
        if kern1 is None:
            kern1 = max(15, (n_samples // 4) | 1)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern1), padding=(0, kern1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.classify = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)            # (B,1,C,T)
        x = self.block1(x)
        x = self.block2(x)
        return self.classify(x.flatten(1))


class ShallowConvNet(nn.Module):
    """Schirrmeister et al. (2017) ShallowConvNet -- the canonical HGD baseline.

    Temporal conv -> spatial conv -> square -> mean-pool -> log: a *learned
    log-band-power* model (mimics FBCSP log-variance). This is the STRONG
    band-power comparator the RN 'power' readout is really competing against;
    if RN-sync can't beat this, that is the honest, expected result.
    """
    def __init__(self, n_classes=4, n_channels=128, n_samples=450, F=40,
                 kern=None, pool=None, stride=None, dropout=0.5):
        super().__init__()
        kern = kern or max(13, (n_samples // 18) | 1)   # ~25 @ 500Hz, scaled to decim
        pool = pool or max(8, n_samples // 8)
        stride = stride or max(2, pool // 5)
        self.temporal = nn.Conv2d(1, F, (1, kern), bias=False)
        self.spatial = nn.Conv2d(F, F, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            d = self.bn(self.spatial(self.temporal(torch.zeros(1, 1, n_channels, n_samples))))
            d = self.pool(d * d).flatten(1)
        self.classify = nn.Linear(d.shape[1], n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn(self.spatial(self.temporal(x)))
        x = self.pool(x * x)                       # square -> mean-pool
        x = torch.log(x.clamp(min=1e-6))           # log band-power
        return self.classify(self.drop(x.flatten(1)))


# --------------------------------------------------------------------------- #
# Phase 6: Unified training engine
# --------------------------------------------------------------------------- #
def make_loaders(X_tr, y_tr, X_te, y_te, batch_size=128, val_frac=0.15, seed=42):
    val_mask = stratified_trial_split(np.arange(len(y_tr)), y_tr, val_frac, seed)
    Xtr = torch.tensor(X_tr[~val_mask]); ytr = torch.tensor(y_tr[~val_mask])
    Xva = torch.tensor(X_tr[val_mask]);  yva = torch.tensor(y_tr[val_mask])
    Xte = torch.tensor(X_te);            yte = torch.tensor(y_te)
    mk = lambda X, y, sh: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=sh)
    return mk(Xtr, ytr, True), mk(Xva, yva, False), mk(Xte, yte, False)


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        loss_sum += crit(logits, by).item() * by.size(0)
        correct += (logits.argmax(1) == by).sum().item()
        tot += by.size(0)
    return loss_sum / tot, correct / tot


def train_model(model, train_loader, val_loader, device, *, augment=False,
                max_epochs=100, lr=1e-3, weight_decay=1e-4, patience=15,
                use_amp=True, label_smoothing=0.05, grad_clip=1.0,
                verbose=True, log_every=5):
    model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    # AMP/GradScaler is unimplemented for complex tensors (the guide's "AMP is
    # non-negotiable" is incorrect for a complex-valued net). The RN needs only
    # ~1GB anyway, so we disable AMP whenever the model has complex parameters.
    has_complex = any(p.is_complex() for p in model.parameters())
    amp_on = use_amp and device.type == 'cuda' and not has_complex
    scaler = torch.amp.GradScaler('cuda', enabled=amp_on)

    best_val = float('inf'); best_state = None; bad = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            if augment:
                bx = augment_batch(bx)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp_on):
                logits = model(bx)
                loss = crit(logits, by)
            scaler.scale(loss).backward()
            if grad_clip:
                if amp_on:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        sched.step()
        vloss, vacc = evaluate(model, val_loader, device, crit)
        history['val_loss'].append(vloss); history['val_acc'].append(vacc)
        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:3d}  val_loss {vloss:.4f}  val_acc {vacc:.3f}")
        if vloss < best_val - 1e-5:
            best_val = vloss; best_state = {k: v.detach().cpu().clone()
                                            for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop @ epoch {epoch} (best val_loss {best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------- #
# Phase 7: Hopf probe (interpretability)
# --------------------------------------------------------------------------- #
def hopf_probe_signal(x, sfreq=500, fmin=4, fmax=45):
    """Extract (peak_freq, spectral_peakedness, autocorr_half_life) from a 1D
    real oscillator trajectory."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if np.allclose(x, 0):
        return dict(peak_hz=np.nan, peakedness=0.0, half_life=0.0)
    nperseg = min(256, len(x))
    f, Pxx = welch(x, fs=sfreq, nperseg=nperseg)
    band = (f >= fmin) & (f <= fmax)
    if band.sum() < 2:
        band = np.ones_like(f, dtype=bool)
    fb, Pb = f[band], Pxx[band]
    peak_hz = float(fb[np.argmax(Pb)])
    p = Pb / (Pb.sum() + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    Hmax = math.log(len(p))
    peakedness = float(1.0 - H / (Hmax + 1e-12))
    # autocorrelation half-life
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    below = np.where(ac < 0.5)[0]
    half_life = float(below[0]) if len(below) else float(len(ac))
    return dict(peak_hz=peak_hz, peakedness=peakedness, half_life=half_life)


def run_hopf_probe(model, X_sample, device, sfreq_eff, fmin=4, fmax=45, top=12):
    """Extract the 64-oscillator trajectory for one sample and tabulate each
    oscillator's learned peak frequency / peakedness / autocorrelation half-life.

    Returns a pandas DataFrame (concrete evidence of the RN's interpretability).
    sfreq_eff is the decimated sampling rate (500 / decim).
    """
    import pandas as pd
    model.eval()
    x = torch.as_tensor(X_sample, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, traj = model(x, return_traj=True)          # (1, T-1, N) complex
    traj = traj[0].cpu().numpy()                       # (T-1, N)
    rows = []
    for j in range(traj.shape[1]):
        m = hopf_probe_signal(traj[:, j].real, sfreq=sfreq_eff, fmin=fmin, fmax=fmax)
        rows.append(dict(oscillator=j, **m))
    df = pd.DataFrame(rows).sort_values("peakedness", ascending=False).reset_index(drop=True)
    return df.head(top) if top else df


def chance_threshold(n_samples, n_classes=4, alpha=0.05):
    """Upper significance threshold on accuracy under the null (Combrisson &
    Jerbi 2015): the smallest accuracy that significantly exceeds chance for
    `n_samples` test trials. Use THIS, not the theoretical 1/n_classes, for
    small test sets. Returns the threshold accuracy in [0, 1]."""
    from scipy.stats import binom
    k = binom.ppf(1 - alpha, n_samples, 1.0 / n_classes)
    return float((k + 1) / n_samples)


def class_coherence(model, X, y, device, n_classes=4):
    """Interpretability: per-class mean pairwise coherence matrices and the most
    class-DISCRIMINATIVE oscillator pairs. Returns (C_by_class [K,N,N] complex,
    discriminability [N,N] = variance of |C| across classes). The Im part is the
    volume-conduction-robust component (imaginary coherence)."""
    model.eval()
    N = model.N
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = np.asarray(y)
    C_by = np.zeros((n_classes, N, N), dtype=np.complex64)
    with torch.no_grad():
        for c in range(n_classes):
            xb = X[y == c]
            if len(xb) == 0:
                continue
            _, z = model(xb, return_traj=True)         # (B, T, N) complex
            u = z / (z.abs() + 1e-6)
            C = torch.einsum('bti,btj->bij', u, u.conj()) / z.shape[1]
            C_by[c] = C.mean(0).cpu().numpy()
    discrim = np.abs(C_by).std(axis=0)                 # (N,N) across-class spread of PLV
    return C_by, discrim


# --------------------------------------------------------------------------- #
# Probe helper (calibrate decimation / batch on this GPU)
# --------------------------------------------------------------------------- #
def probe_timing(T_list=(450, 750, 1125), batch=128, n_channels=128, device=None):
    device = device or get_device()
    out = {}
    for T in T_list:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = ResonanceNetwork(n_channels=n_channels).to(device)
        x = torch.randn(batch, n_channels, T, device=device)
        y = torch.randint(0, 4, (batch,), device=device)
        crit = nn.CrossEntropyLoss()
        t0 = time.time()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        out[T] = dict(sec_per_batch=round(dt, 3), peak_gb=round(mem, 2))
        del model, x, logits, loss
    return out


if __name__ == "__main__":
    dev = get_device()
    print("device:", dev)
    print("timing probe (fwd+bwd, batch=128):")
    for T, m in probe_timing().items():
        print(f"  T={T:5d}: {m['sec_per_batch']}s/batch  peak {m['peak_gb']} GB")
