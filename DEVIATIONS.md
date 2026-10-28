# DEVIATIONS

Every departure from TASK_PITCH1.md, or a judgement call the spec leaves open, with the reason.
PREREG.md does not exist yet. Per Tom, it is written together before Phase 3; entries logged before then
are design decisions, not post-hoc changes.

| # | date | where | decision | reason |
|---|---|---|---|---|
| 1 | 2026-10-27 | banks.py, bandwidth init | `bw` initialised from the published `mu ~ U(-1,-0.4)` converted to Hz at the model's fs (`bw = -mu*h*fs/2pi`, about 0.16-0.40 Hz at 250 Hz) | S2 says "keep everything else identical". This keeps per-sample damping at init exactly as published, so only the frequency axis changes. |
| 2 | 2026-10-27 | banks.py, nl integrator | Default is the published one-RK4-step-per-sample scheme. An opt-in `--rk4-substeps k` is added. | With Hz-correct f0, one RK4 step per sample over-damps fast oscillators: bandwidth is 1.24x at 30 Hz and 2.3x at 40 Hz (fs = 250). Above about 60 Hz the oscillator decays to zero. The linear bank uses the exact exponential, so for the top two HGD oscillators lin and nl differ in effective bandwidth, not only in nonlinearity/coupling. k = 2 fixes 1-40 Hz (at most 1.04x). Recommendation: run HGD Phase 1 with k = 1 (primary, faithful) and k = 2 as a sensitivity check. **The hippocampal 2-150 Hz band needs k >= 4 at 250 Hz** (Phase 3 decision, to go into PREREG). |
| 3 | 2026-10-27 | banks.py, f0 range | f0 clamped to [0.5 Hz, 0.45 fs] | The spec asks for 2-150 Hz in hippocampus with 250 Hz decimation. 150 Hz is above Nyquist (125 Hz). Options for PREREG: cap at 112 Hz, or decimate to 500 Hz instead. |
| 4 | 2026-10-27 | trial_shuffle becomes the evoked-vs-induced decomposition (`decomposition.py`) | **Decided by Tom.** It is not a decoder arm, because using the label at test time would be leakage. Shuffle partners come from the same split only (train with train, test with test; inside the train-only CV, from the same fold). It is reported as genuine vs trial-shuffled (evoked) vs genuine-minus-shuffled (induced) coherence: class-consistent magnitudes, plus train-only 5-fold CV discriminability. It stays out of the main decoder-accuracy table. | Magnitudes use class-averaged complex coherence, because per-trial |C_ij| is high for any narrowband pair. The decomposition is validated on known evoked-only and induced-only data (`tests/test_decomposition.py`). |
| 5 | 2026-10-27 | banks.py, nl_uncoupled_wide | The extra parameters form a second complex encoder layer `I = ctanh(M ctanh(W_enc x + b) + b2)`, with M (N x N complex) initialised to identity + 0.05 noise. Extra real params are 2N^2 + 2N, against 2N^2 for W. | A merely reparameterised linear layer would match the count but add no capacity. This adds real capacity in the encoder, which is what the control has to rule out. |
| 6 | 2026-10-27 | banks.py, RNG order | All four banks draw bw, W, W_enc and M from one generator in the same order, so they share an identical encoder init for a given seed. | The published lin and nl banks drew different streams, so their encoders started differently. Sharing the stream makes the bank contrasts cleaner. |
| 7 | 2026-10-27 | metric | Primary score is test **balanced** accuracy, with plain accuracy also stored | S4 asks for balanced accuracy. HGD test sets are balanced (40/class), so the two coincide there. |
| 8 | 2026-10-27 | Phase 2 generators | "Power-matched" means each channel of each trial is z-scored. The `power` readout can still decode, because oscillator power is the power of a spatial mixture W_enc x, which depends on inter-channel phase relationships (as in CSP). | Can't be removed without changing the published encoder. It applies equally to the published HGD `power` arm, and is reported, not hidden. |
| 9 | 2026-10-27 | Phase 2 difficulty | Driven input gain kappa = 6.5 and emergent coupling K = 40 were set on **pilot seed 999 only**, never an evaluation seed. The aim was an untrained-linear-bank train-CV of about 0.7 for both, so that neither contrast is floor- or ceiling-limited. | The initial values gave 1.00 (driven) and 0.63 (emergent). Pilot values: driven kappa 5 -> 0.51, 8 -> 0.83, 12 -> 0.96; emergent K 25 -> 0.63, 40 -> 0.69, 60 -> 0.68. |
| 10 | 2026-10-27 | Phase 2 bank band | 2-40 Hz, N = 16, fs = 250 Hz, 1 s windows | The synthetic units sit at 7-9 Hz. HGD-like settings otherwise (S4: same N and window length wherever possible). |
| 11 | 2026-10-28 | **Exploratory step after the Phase 2 stop (Tom's decision): mechanism-statistic screen, synthetic only.** Phase 3 is NOT started. | Phase 2 showed the nl-vs-lin decoder contrast cannot tell emergent from driven class information. This step screens statistics that might, before any redesign of H3. Everything below was written and committed **before any screening code was run**. |

### DEVIATION 11: success criterion and protocol (frozen 2026-10-28, before any screening run)

**Generators.** Three generators, each with the Phase 2 data format (8 channels, 1 s at 250 Hz, z-scored per channel
and trial, 4 classes, 100 train trials per class):
- EMERGENT: unchanged from Phase 2 (K = 40).
- DRIVEN: unchanged from Phase 2 (kappa = 6.5). The shared input is a narrowband oscillatory drive.
- **DRIVEN_COMMON (new):** the *same* uncoupled noise-driven Stuart-Landau units as EMERGENT (same lam, natural
  frequencies and total noise variance), with coupling replaced by an **unobserved shared white-noise source**.
  It enters unit j with class-specific complex loading rho * exp(i psi_{k,j}). Private noise is scaled so the total
  noise variance per unit equals EMERGENT's sigma^2. There is no coupling and no oscillatory drive; it is the
  classic common-noise confound for coupling inference.
  - *Calibration (pilot seed 999 only):* rho is chosen from the grid {0.3, 0.5, 0.7, 0.85} so that DRIVEN_COMMON's
    mean pairwise PLV (6-10 Hz) is closest to EMERGENT's on seed 999. This makes plain coherence non-diagnostic by
    construction, so the screen is hard.

**Phase extraction (offline analysis, acausal is fine):** 4th-order Butterworth 6-10 Hz with sosfiltfilt, then Hilbert,
per channel per trial. The first and last 0.1 s are discarded. All statistics pool the 400 train trials of a dataset
(one dataset = one generator x one seed). Labels are not used.

**Candidate statistics (all parameters fixed here):**
1. **Coupling-function strength** (`S_couple`): for each unit i, a multivariate Kuramoto-type regression on samples
   pooled over trials, dphi_i/dt = w_i + sum_{j != i} [a_ij sin(phi_j - phi_i) + b_ij cos(phi_j - phi_i)] + e. The
   derivative is the wrapped first difference x fs, and the fit is ordinary least squares. S_couple = mean over
   ordered pairs of sqrt(a_ij^2 + b_ij^2), in rad/s.
2. **Directed phase transfer entropy** (Lobier et al. 2014 style): phases are binned into 8 equal-width bins, with
   delay delta = 8 samples (32 ms, about a quarter cycle at 8 Hz). TE_{j->i} = I(phi_i(t+delta); phi_j(t) | phi_i(t))
   uses plug-in entropies over within-trial samples pooled across trials. Two statistics come from it:
   2a `S_TE` = mean over ordered pairs of TE_{j->i} (bits).
   2b `S_asym` = mean over unordered pairs of |TE_ij - TE_ji| / (TE_ij + TE_ji) (dPTE-style asymmetry).
3. **Residual coherence after removing a common-input estimate** (`S_resid`): per trial, the common-input estimate
   is the first complex principal component of the 8 analytic signals (unsupervised, computed per trial). Each
   channel's analytic signal is regressed on it (complex least squares), and S_resid = mean over trials and pairs of
   the residual-phase PLV |<u_i conj u_j>_t|.
- Reference only, not a candidate: plain mean pairwise PLV (`S_plv`). It is expected to fail by construction.

**Success criterion (Tom).** A statistic *separates emergent from both driven generators* if **all** of these hold:
- (a) Emergent vs DRIVEN: the paired difference S(E, seed) - S(D, seed) has the same sign in **>= 9/10 seeds**, and
  its 95% bootstrap CI (10,000 resamples of seeds) **excludes zero**.
- (b) Emergent vs DRIVEN_COMMON: the same requirement, separately.
- (c) The sign is the same in (a) and (b), so that a single threshold on S separates emergent from both driven
  generators.
- Evaluation seeds are 0-9 (the seed is the unit). No multiplicity correction is applied: this is an exploratory
  screen of 4 candidates, and every candidate is reported. Any statistic that passes must be re-validated on fresh
  seeds, and at other difficulty and N settings, before it is used in a redesigned H3.
- This screen makes no claim about real data.
