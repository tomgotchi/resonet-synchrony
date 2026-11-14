# DEVIATIONS

Every departure from TASK_PITCH1.md, or a judgement call the spec leaves open, with the reason.

> **Layout note (2026-11-03, repository cleanup).** Frozen entries below are left verbatim. Since then:
> - `resonet_sync/screen.py`, `screen2..10.py`, `diag10*.py` live in `resonet_sync/screens/`; `screen.py` is now
>   `screen1.py`, with output in `results/screen1/`.
> - `boot_ci` moved to `resonet_sync/stats.py`.
> - `TASK_PITCH1.md`, `RUN_PHASE1_LOCAL.md`, `PREREG_H3_REDESIGN_DRAFT.md` and `REVIEW_BOARD.md` live in `docs/`.
>
> Only import paths changed; no protocol logic changed. Run a screen as
> `python -m resonet_sync.screens.screenN`.
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
- *Calibration outcome (2026-10-28, seed 999 only):* EMERGENT PLV 0.685. DRIVEN_COMMON PLV for rho 0.3 / 0.5 / 0.7 / 0.85
  was 0.485 / 0.504 / 0.566 / 0.672. **rho = 0.85** was selected by the frozen rule (closest to EMERGENT). The statistic
  implementations were validated on toy systems with known answers (`tests/test_mechanism.py`) before any
  evaluation seed: S_couple is monotone in K but attenuated by the 6-10 Hz filter; TE recovers the direction of a
  unidirectional coupling; S_resid falls to the 1-s PLV noise floor (~0.45) for a common-input system.
| 12 | 2026-10-28 | **TE re-validation (Tom: "go").** Exploratory and synthetic only; Phase 3 still not started. | DEVIATIONS 11 requires a passing statistic to be re-validated on fresh seeds and at other settings before any use in H3. Everything below was frozen **before** this run. |

### DEVIATION 12: protocol (frozen 2026-10-28, before any re-validation run)
**Statistic.** S_TE exactly as in DEVIATIONS 11 (8 bins, delay 8 samples, 6-10 Hz, plug-in, mean over ordered
pairs).
*Secondary, descriptive:* **S_TE_excess** = S_TE minus the same statistic computed with each source unit taken from a
*different trial* (a fixed random derangement of trials). This measures how much of TE is genuinely between units
rather than estimator bias or own-phase predictability.

**Seeds.** **10-19** (fresh; the screen used 0-9). The seed is the unit. Calibration uses seed 999 only.

**Conditions.** The base condition plus 10 one-at-a-time variants. Every generator in a condition uses that condition's
setting.
| id | change from base (M = 8 units, K = 40, rho = 0.85, 1 s windows, no observation noise) | rho for DRIVEN_COMMON |
|---|---|---|
| base | none | 0.85 (DEVIATIONS 11) |
| K25 / K60 | emergent coupling K = 25 / 60 | recalibrated* |
| rho70 / rho95 | DRIVEN_COMMON rho = 0.70 / 0.95 | fixed by definition |
| M4 / M16 | 4 / 16 units (graph density 0.35, kept connected) | recalibrated* |
| win05 / win2 | 0.5 s / 2 s windows | recalibrated* |
| snr10 / snr0 | white observation noise at 10 / 0 dB (vs. the channel's signal variance), added before z-scoring | recalibrated* |

\*Recalibration uses the frozen DEVIATIONS 11 rule: on seed 999, pick rho from {0.3, 0.5, 0.7, 0.85, 0.95} with mean
pairwise PLV closest to EMERGENT's in the same condition.

**Criteria.**
- *Per condition:* the DEVIATIONS 11 criterion (>= 9/10 seeds with the same sign and a 95% bootstrap CI excluding 0,
  for both E-D and E-DC, with the same sign in both).
- **Primary:** TE must pass the *base* condition on fresh seeds 10-19.
- **Robustness:** TE is called robust if it also passes in **>= 8 of the 10 variants**. Every failing variant is
  listed with its numbers. (This threshold is Claude's proposal, fixed here before running.)
- **Use in H3:** TE becomes a candidate for an H3 redesign only if both the primary and the robustness criteria hold.
  Even then, it says nothing about real data until tested there.
| 13 | 2026-11-01 | **Exploratory screen 3: lagged vs instantaneous dependence.** Synthetic only; Phase 3 still not started. | DEVIATIONS 12 showed TE cannot separate coupling from strong unobserved common noise (rho = 0.95). Coupling predicts *lagged* influence. Common white noise mainly predicts *instantaneous* innovation correlation. This screen tests a statistic built on that difference. Frozen before any run. |

### DEVIATION 13: protocol (frozen 2026-11-01, before any run)
**Model.** For each dataset, a VAR(p) with p = 25 samples (100 ms, about one theta cycle; fixed a priori, not
selected) is fit by OLS on the 8 observed, z-scored channels. The fit pools within-trial samples across the
400 train trials, and lags never cross trial boundaries.
- **S_GC** (candidate) = mean over ordered pairs (j -> i) of the conditional Granger index
  log(var(e_i | model without j's lags) / var(e_i | full model)).
- **S_inst** (descriptive) = mean |correlation| between full-model residuals of channel pairs (instantaneous
  dependence).
- **S_TE** (reference) = the DEVIATIONS 11 statistic, recomputed for comparison.

**Data.** Generators as in DEVIATIONS 12. **Fresh seeds 20-29** (0-9 and 10-19 are used). Conditions: **base**
(rho 0.85), **rho70** and **rho95** (rho95 is the case TE failed).

**Criterion.** The DEVIATIONS 11 rule (>= 9/10 seeds with the same sign, 95% bootstrap CI excluding 0, for both E-D
and E-DC, same sign). S_GC **succeeds** only if it passes in **all three** conditions, including rho95.

**Known caveat (stated before running).** The generators decimate 1000 -> 250 Hz with a *zero-phase* FIR. Its
symmetric smearing can turn instantaneous correlation into short two-sided lagged dependence, and GC is sensitive
to that. The caveat applies equally to all generators but is reported with the result.
| 14 | 2026-11-01 | **Confirmatory screen 4: the post-hoc conjunction rule from DEVIATIONS 13.** Synthetic only. | Screen 3 suggested, post hoc, that "high TE AND low instantaneous residual correlation" separates emergent coupling from both driven generators. A post-hoc rule needs an independent confirmatory test. Frozen before any seed 30-39 dataset was generated. |

### DEVIATION 14: protocol (frozen 2026-11-01, before any run)
**Rule.** A dataset is classified **emergent** iff S_TE > t_TE **and** S_inst < t_inst. Statistics are exactly as in
DEVIATIONS 11 and 13.

**Thresholds** are fixed now from the screen-3 *base* condition (seeds 20-29), as midpoints of generator means, and
are never re-tuned:
- t_TE = (mean S_TE of E + mean S_TE of D) / 2 = **0.015295**
- t_inst = (mean S_inst of E + mean S_inst of DC) / 2 = **0.111262**

**Data.** **Fresh seeds 30-39.**
- Conditions: base, rho70, rho95, K25, K60, M16, win05, snr0.
- rho for the recalibrated variants is taken from the DEVIATIONS 12 calibration (K25 0.70, K60 0.95, M16 0.70,
  win05 0.85, snr0 0.85).

**Success criterion (per condition).**
- >= 9/10 EMERGENT datasets are classified emergent.
- >= 9/10 DRIVEN datasets and >= 9/10 DRIVEN_COMMON datasets are classified not emergent.

The rule is **confirmed** if it succeeds in base, rho70 and rho95 (primary: the common-input confound) **and** in at
least 4 of the 5 other variants. Thresholds are in absolute units and are not adapted per condition, so the variants
also test whether fixed thresholds transfer. That transfer is a precondition for any use on real data.
| 15 | 2026-11-01 | **Boundary test 5: strong common input x low SNR** for the DEVIATIONS 14 rule. Synthetic only. | Screen 4 caveat 1: at 0 dB the instantaneous leg collapsed, and common input was rejected by TE alone, which fails at rho = 0.95. The combination was untested. Frozen before any run. |

### DEVIATION 15: protocol (frozen 2026-11-01, before any run)
- **Rule and thresholds unchanged** from DEVIATIONS 14 (t_TE = 0.015295, t_inst = 0.111262).
- **Fresh seeds 40-49.** rho is fixed at **0.95** in every condition (no recalibration; strong common input is the
  point). White observation noise is set at **20, 10, 5 and 0 dB** (four conditions, all three generators).
- **Per condition:** the success criterion from DEVIATIONS 14 (>= 9/10 correct for each generator).
- **Reported:** the lowest SNR at which the rule still succeeds. This defines its *operating range*. Real-data use
  would require the recording's SNR in the theta band to lie inside that range, estimated on train data only.
- **Expectation stated in advance:** failure at 0 dB is likely. A failure is the informative result here, not a
  problem to tune away.
| 16 | 2026-11-02 | **Screen 6: the SNR-robust rule with LDS-denoised instantaneous dependence.** Synthetic only. | Screen 5 found the DEVIATIONS 14 rule fails under strong common input below 20 dB, because sensor noise dilutes S_inst. Frozen before calibration and before any seed 50-59 dataset. |

### DEVIATION 16: protocol (frozen 2026-11-02)
**How the design was reached, on pilot seed 999 only:**
1. Normalising the raw residual covariance by theta-band power failed. At 0 dB emergent rose to the common-input
   level.
2. Raw LDS block norms are not identifiable, because they depend on the latent basis.
3. Adopted: denoise through a fitted latent LDS, then measure. On the pilot, S_inst_dn stayed separated at
   every SNR (common input 0.19-0.24; emergent 0.03-0.04). The denoised GC again ranked common input above
   coupling, so it is excluded.

**Statistic S_inst_dn (`resonet_sync/lds.py`):**
- Fit: EM for the latent LDS (one 2-D oscillator per channel, full A and Q, diagonal R), z-scored channels,
  150 trials subsampled with seed 0, at most 200 iterations, tol 1e-6.
- Then simulate the model-implied **noise-free** process (no R): 200 trials x 250 samples, burn-in 100, seed 0.
- S_inst_dn = mean |residual correlation| of a VAR(25) on that process (the DEVIATIONS 13 definition).

**Rule.** A dataset is **emergent** iff S_TE > 0.015295 (unchanged from DEVIATIONS 14) **and** S_inst_dn < t_dn.

**Calibration (fixed now, run next).** t_dn = midpoint of the mean S_inst_dn of EMERGENT and DRIVEN_COMMON in the
screen-3 *base* condition, seeds 20-29 (rho 0.85, no observation noise). It is committed before seeds 50-59 are
generated and never re-tuned.

**Confirmation.** **Fresh seeds 50-59.**
- **Primary:** rho 0.95 x observation SNR {20, 10, 5, 0} dB, the exact failure grid of screen 5.
- **Generalisation:** base, rho70, K25, M16, win05.

**Criterion.** Per condition, >= 9/10 correct for each generator. **Confirmed** iff all 4 primary conditions succeed
and >= 4 of the 5 generalisation conditions succeed.
- *Calibration outcome (2026-11-02, seeds 20-29 base):* mean S_inst_dn was EMERGENT 0.0352 and DRIVEN_COMMON 0.1967,
  giving **t_dn = 0.115964**. Committed before any seed 50-59 dataset was generated.
| 17 | 2026-11-02 | **Screen 7: within-session contrast rule (no absolute thresholds).** Synthetic only; Phase 3 still not started. | H3 is a within-session contrast (wheel vs maze, same animal, electrodes and noise). Absolute thresholds do not transfer to real data, but *paired differences* between two regimes of one session might. Frozen before any run on seeds 60-69. |

### DEVIATION 17: protocol (frozen 2026-11-02)
**Generator.** The new `mixed` kind has coupling (K) **and** unobserved common input (rho) in one network. A
"session" is a seed. Its two regimes A and B share the units, natural frequencies, graphs and loadings, and use
independent trial draws (`stream` 0 and 1). Pilot on seed 999 only:
- S_inst_dn tracks rho regardless of K (about 0.10 at rho 0.5 and about 0.29 at rho 0.9).
- S_TE rises with K, but **also with rho**: at K ~ 0 and rho 0.9, S_TE equals coupling's.

**Statistics.** S_TE and S_inst_dn exactly as in DEVIATIONS 14 and 16. Paired differences Delta = S(A) - S(B) per
session.

**Decision rule.** A is "more coupling-dominated than B" iff, across the 10 sessions:
- (i) Delta S_TE > 0 in >= 9/10 sessions with a 95% bootstrap CI excluding 0, **and**
- (ii) Delta S_inst_dn shows no increase: the upper 95% CI bound is < **+0.03**. That margin is about 15% of the
  pilot's rho 0.5 -> 0.9 effect, fixed now.

**Scenarios** (A vs B, as (K, rho)). Each runs at observation SNR 20 dB and 0 dB.
| id | A | B | truth | the rule must |
|---|---|---|---|---|
| S1 coupling-up | (40, 0.5) | (25, 0.5) | more coupling in A | **fire** |
| S2 common-up (trap) | (25, 0.9) | (25, 0.5) | only common input differs | **not fire** |
| S3 null | (25, 0.5) | (25, 0.5) | identical mechanism | **not fire** |
| S5 coupling vs common | (40, ~0) | (~0, 0.9) | A coupled, B common-input only | **fire** |

**Data.** **Fresh seeds 60-69** (sessions). For each regime: 100 trials per class x 4 classes, 1 s, 8 channels.

**Success.** All 8 scenario x SNR cells behave as the table requires. Any wrong firing in S2 or S3 (a false
positive) is reported as a failure of the rule, whatever happens elsewhere.
| 18 | 2026-11-02 | **Screen 8: scale-free guard for the within-session rule.** Synthetic only. | The DEVIATIONS 17 guard margin (+0.03 absolute) has no meaning on a real recording's scale. A ratio of the same statistic does. Frozen before any run on sessions 70-79. |

### DEVIATION 18: protocol (frozen 2026-11-02)
**Rule v2.** A is "more coupling-dominated than B" iff:
- (i) Delta S_TE > 0 in >= 9/10 sessions with 95% CI > 0 (unchanged), **and**
- (ii) the per-session relative change r_s = (S_inst_dn(A) - S_inst_dn(B)) / S_inst_dn(B) has an upper 95%
  bootstrap CI **< 0.25**.

*Rationale, from screen 7 (sessions 60-69), stated before this run:* r was about +0.04 when coupling rose and about
+1.4 in the common-input trap. The 0.25 margin is fixed now.

**Data.** **Fresh sessions 70-79**, observation SNR **0 dB** throughout (the harder case). Scale-changing variants:
| variant | change |
|---|---|
| M16 | 16 units |
| win05 | 0.5 s windows |

Scenarios per variant are S1 coupling-up, S2 trap, S3 null and S5 coupling-vs-common (as in DEVIATIONS 17). In
addition there is **S2b, a mild trap** in the base configuration: A = (K 25, rho 0.7) vs B = (K 25, rho 0.5),
which must **not fire**.

**Success.** All 9 cells behave as required. A false positive in any trap or null cell is a failure of rule v2 and
is reported as such.
| 19 | 2026-11-02 | **Screen 9: nuisance-confound traps for rule v2** (theta-frequency and SNR differences between regimes). Synthetic only. | The H3 redesign draft names theta-frequency and signal-strength differences between wheel and maze as possible confounds of Delta S_TE. They are untested. Frozen before any run on sessions 80-89. |
| 20 | 2026-11-03 | **Screen 10: rule v3 = rule v2 + cycle-normalised TE delay + LDS noise-fraction matching.** Synthetic only. | Screen 9 failed: +1 Hz theta (F1) and +10 dB SNR (N1) each triggered rule v2 without any coupling change. Fixes and both stages frozen before any run. |
| 21 | 2026-11-04 | **Screen 11: terminal calibration of rule v3r** (review board rounds 1-2; Tom approved). Synthetic only. | Stage C found no counter-example within its family. The board requires graded traps, including the rival hypothesis (rhythmic common input), with false-positive rates estimated by resampling rather than pass/fail cells. Retirement is pre-committed. Frozen before any calibration or tier run. |
| 22 | 2026-11-05 | **New arm `power+surrogate`** (log power + surrogate-phase coherence, N + 2P dims) and family `incremental_capacity_matched` = (sync+power) - (power+surrogate), per bank. Added to `ALL` (not `CORE`). | Review board (statistician): `(sync+power) - power` is capacity-confounded (N + 2P vs N dims). The capacity-matched control is the analogue of the published sync-vs-surrogate logic. Frozen before any all-arms Phase 1 result exists; the running Colab job computes core arms only. |

### DEVIATION 19: protocol (frozen 2026-11-02)
**Rule v2** is unchanged from DEVIATIONS 18. Generator `mixed` with K = 25 and rho = 0.5 in **both** regimes
(identical mechanism), except where noted. New parameter `f_shift_hz` adds a constant to all units' natural
frequencies, applied after the seeded draw (so templates are otherwise identical).
| cell | A | B | must |
|---|---|---|---|
| F1 freq up | f_shift +1 Hz, 10 dB | 0 Hz, 10 dB | not fire |
| F2 freq down | f_shift -1 Hz, 10 dB | 0 Hz, 10 dB | not fire |
| N1 SNR up | 10 dB | 0 dB | not fire |
| N2 SNR down | 0 dB | 10 dB | not fire |
| C1 control: coupling up | K 40, 10 dB | K 25, 10 dB | **fire** |

**Data.** Fresh sessions 80-89, independent trial streams for A and B.

**Success.** All 5 cells as required. Any firing in F1, F2, N1 or N2 is a **false positive caused by a nuisance
variable**. It would invalidate H3' as drafted unless the corresponding nuisance is matched or adjusted in the hc-5
design.

### DEVIATION 20: protocol (frozen 2026-11-03, before any run)
**Rule v3.** Per session, per regime, label-free (code: `resonet_sync/nuisance.py`, `resonet_sync/screen10.py`):
1. **Frequency.** f_c = Welch peak in 6-10 Hz, pooled over trials and channels, with parabolic interpolation.
   The TE delay = 0.256 cycles of f_c (= 8 samples at 8 Hz, 250 Hz), applied as a fractional delay on the
   unwrapped phase. Bins, band and edges are unchanged.
2. **Noise matching.** Fit the latent LDS (EM, 200 iterations, tol 1e-6, at most 150 trials) to each regime.
   R_c = noise fraction of channel c (z-scored data). Per channel, the target is max(R_A,c, R_B,c). The regime
   below target gets white noise of variance v = (target - R) / (1 - target) added to its z-scored channel
   (seed = 1000 * session + regime index). Refit the LDS on both (possibly degraded) regimes.
3. **Statistics on the matched data:** S_TE_cyc and S_inst_dn (as DEVIATIONS 16).
4. **Decision rule:** unchanged from v2. Delta S_TE_cyc > 0 in >= 9/10 sessions with bootstrap CI > 0, **and**
   the relative Delta S_inst_dn has an upper CI < 0.25.
- **Also reported:** Delta f_c, mean Delta R before matching, and the unmatched v2 statistics.

**Stage D (development): sessions 80-89, the 5 screen 9 cells.** These sessions are already used, so this is a bug
and feasibility check, not evidence. If any cell is wrong, one documented revision is allowed before stage C. A
second failure ends the line.

**Stage C (confirmation): fresh sessions 90-99, 9 cells.** Mechanism K 25, rho 0.5 unless stated; 10 dB unless
stated.

| cell | A | B | must |
|---|---|---|---|
| F1 freq up | +1 Hz | 0 | not fire |
| F2 freq down | -1 Hz | 0 | not fire |
| N1 SNR up | 10 dB | 0 dB | not fire |
| N2 SNR down | 0 dB | 10 dB | not fire |
| X1 both nuisances up | +1 Hz, 10 dB | 0 Hz, 0 dB | not fire |
| T1 mild common-input trap | rho 0.7 | rho 0.5 | not fire |
| C1 coupling up | K 40 | K 25 | **fire** |
| C2 coupling up, A noisier | K 40, 0 dB | K 25, 10 dB | **fire** |
| C3 coupling up, A slower | K 40, -1 Hz | K 25, 0 Hz | **fire** |

**Success.** All 9 cells correct. A miss in C2 or C3 (matching that removes real coupling effects) counts as a
failure, as does any false positive.


#### DEVIATION 20, revision 1 (the one allowed; frozen 2026-11-03, before any stage C run)
Stage D failed (F2 false positive under the cycle-normalised delay). Two development diagnostics on the used
sessions 80-89 (`results/screen10/diag10_variants.log`, `diag10b_alignment.log`) showed the following:
- A uniform frequency shift with white-noise input is a rotating-frame change, so the physics is identical in real
  time. The delay must therefore stay fixed in **time**; delay-in-cycles was the wrong fix.
- The remaining bias comes from plug-in TE **bin alignment** (mean phase advance per delay = 2.05 bins at 8 Hz,
  fractional at 7/9 Hz) together with the fixed band.
- Only the combination **band centred on f_c (f_c +/- 2 Hz) + future phase rotated by -2 pi f_c d** removed it in both
  directions: F1 -0.0001 (5/10), F2 +0.0001 (6/10), C1 +0.0061 (10/10). Bin dithering added nothing.

**Rule v3r** = rule v3 with step 1 replaced by:
1'. f_c as before. Analysis band = f_c +/- 2 Hz (same Butterworth/Hilbert/edges). Delay = 8 samples (32 ms).
    The future phase is rotated by -2 pi f_c * 32 ms before binning (`nuisance.s_te_aligned`).
- Steps 2-4 are unchanged: LDS noise matching, S_inst_dn on the matched data, and the decision rule with margin
  0.25.

**Stage C proceeds as frozen** (fresh sessions 90-99, 9 cells, all must be correct), using S_TE_al. A failure ends
this line: no further revision.
- **Caveat stated in advance:** the noise-matching cells were developed with S_TE_cyc. The S_TE_al + matching
  combination is first seen in stage C.

#### DEVIATION 20, pre-commitment before stage C finishes (2026-11-03; review board round 2)
Frozen while stage C is still running. The judge read stage C partial results mid-run once, and checkpoint commits
of partial.json exist, so this is stated before the final verdict:
1. **A stage C pass is NOT confirmation of rule v3r.** The F1, F2, X1 and C3 cells run on the Euler-Maruyama
   generator. Its damping depends on frequency (-5.32 / -5.03 / -4.69 s^-1 at 7 / 8 / 9 Hz vs -6.28 nominal), so
   these cells mix the estimator confound with a small change in the physics, and they cannot be credited as
   evidence of frequency robustness.
   - A pass is reported only as "no counter-example found".
   - A fail ends the v3r line, as frozen.
2. **Verdict code.** The running process loaded `decide()` before the NaN fix (commit c43eaf7). The final verdict
   is recomputed with the fixed `decide()` on the final data, and both verdicts are reported, including
   `valid`.
3. **The sign threshold `ceil(0.9 n)` (c43eaf7) applies to the synthetic n = 10 only (= 9/10).** For hc-5 (n = 7)
   the threshold is an open pre-registration decision: 6/7 gives p = 0.0625, and 7/7 has low power. It is not
   settled by that bug fix.
4. **Rule development is closed.** After stage C, the only further synthetic work on this rule is a *terminal
   calibration* (screen 11): graded traps (including rhythmic common input), with the false-positive rate
   estimated by resampling, frozen before running, with no revisions allowed.
   - Pre-commitment: if any rhythmic-common-input or rho cell has an estimated P(fire) > 0.05, the TE/LDS rule is
     retired from H3' and reported as a negative methods result.

### DEVIATION 21: protocol (frozen 2026-11-04, before any run)
**Rule under test:** v3r, unchanged (DEVIATIONS 20 + revision 1). This is a calibration, not a development step:
**no revisions are allowed after any result is seen.** Code: `resonet_sync/screens/screen11.py`.

**Generator (review board corrections).**
- **Every regime** uses `integrator="exact"`. With it, a uniform frequency shift is an exact symmetry of the
  simulator.
- **Rhythmic common input:**
  - an AR(1) complex narrowband drive at 8 Hz, 2 Hz bandwidth, its frequency shifting with `f_shift_hz`;
  - its **own loadings**: a smooth spatial phase gradient of 15 deg per unit, not the white-noise loadings;
  - its own RNG stream.
- **Volume conduction:** `vc_spread` (W_ij = exp(-|i-j|/s), rows normalised).
- **Spatially correlated 1/f noise:** `obs_noise_exp` 1.5, `obs_noise_corr` 0.3 (Sigma_ij = 0.3^|i-j|).

**Base regime:** K 25, rho 0.5, 10 dB, exact integrator. Regime A uses trial stream 0, regime B stream 1.

**Calibration** (pilot sessions 990-994, used for nothing else). Rhythmic levels L1, L2, L3 are the kappa_nb values
whose mean pairwise PLV increase over the base equals 0.5x, 1x and 2x the PLV increase of K 25 -> 40. They are
interpolated on the grid {2, 4, 8, 12, 16, 24, 32}, using 50 trials per class. This is a matched-synchrony design:
does the rule confuse a synchrony increase caused by shared rhythm with one caused by coupling?

**Cells** (A vs B; B = base unless stated):

| must NOT fire | | must fire | |
|---|---|---|---|
| N0 | null (identical parameters) | K30 / K35 / K40 | coupling K vs 25 |
| F1 / F2 | +1 / -1 Hz | K35R | K 35 with rhythm L2 vs K 25 with rhythm L2 |
| R1 / R2 / R3 | rhythmic kappa L1 / L2 / L3 | | |
| W055 / W060 / W065 | white common rho 0.55 / 0.60 / 0.65 | | |
| D075 / D150 | self-damping only (extra_self_damp_K 7.5 / 15) | | |
| VC | volume conduction s = 1.0 | | |
| Q1 | correlated 1/f noise in both; A 10 dB vs B 0 dB | | |

**Tier T (TE leg, all cells).**
- **Sessions:** 100-129 (30, fresh), 100 trials per class.
- **Statistic:** S_TE_al on the *unmatched* data, plus PLV.
- **P_TE(fire):** the probability that the TE leg fires (>= 9/10 positive and boot_ci lower bound > 0) on a random
  10 of the 30 sessions. 10,000 subsets drawn without replacement.

**Tier F (full v3r).**
- **Sessions:** 100-109.
- **Cells, selected by a pre-specified rule:**
  - always Q1, K35 and K35R;
  - plus every must-not-fire cell with P_TE(fire) > 0.05.
- **P(fire):** a smoothed bootstrap of the per-session (Delta S_TE_al, rel Delta S_inst_dn) pairs, drawing 10
  sessions with replacement. The kernel bandwidth is 1.06 x SD x n^(-1/5) per coordinate, with 2,000 resamples.
  The full rule is applied to each resample.
- **Per cell:** P(fire) = the tier-F estimate if the cell ran in tier F, otherwise P_TE(fire), which is an upper
  bound, since the rule can only fire if its TE leg fires.

**Pre-committed decisions.**
1. **Retirement.**
   - If any rhythmic (R) or white-common-input (W) cell has P(fire) > 0.05, the TE/LDS rule is **retired**.
   - It is then not used on any real data, and is reported as a negative methods result.
   - A non-finite tier-F value counts as a failure.
2. **Calibration pass.** Every must-not-fire cell has P(fire) <= 0.05, **and** power at K35 (tier F P(fire)) >= 0.8.
   - The effective alpha = the maximum P(fire) over the must-not-fire cells.
3. **Reported in every outcome:**
   - P(fire) per cell;
   - mean Delta PLV and Delta TE;
   - the effective alpha.

**Fixed details.**
- **Compute:** OMP_NUM_THREADS = 1, 4 workers.
- **Resumability:** the run resumes from `results/screen11/tier*_partial.json` (untracked).
- **Provenance:** results are written with strict JSON.

**DEVIATION 21 outcome (2026-11-04, appended after the run; protocol above unchanged).** Calibration failed:
effective alpha 1.00, power at K35 0.99. The pre-committed retirement was triggered by R1/R2/R3 (rhythmic common
input: P(fire) 1.00 / 1.00 / 0.94). **The TE/LDS rule is retired from real-data use.** Results:
`results/screen11/report.json` and README_results "Screen 11".

### DEVIATION 22: protocol (frozen 2026-11-05, before any all-arms Phase 1 result)
- **Readout.** `power+surrogate` = concat[ log mean |z|^2 per oscillator (N), surrogate coherence (2P; the same
  fixed per-oscillator circular roll as `surrogate`) ].
  - Same BatchNorm, dropout (0.5), training loop, seeds and splits as `sync+power`.
  - Exactness test: `tests/test_banks.py::test_power_surrogate_is_exact_concatenation`.
- **Arms.** `{bank}:power+surrogate` for all four banks. They are part of `--arms all` and can be run alone with
  `--arms lin:power+surrogate,nl_coupled:power+surrogate`.
- **Family `incremental_capacity_matched`.** Per bank: (sync+power) - (power+surrogate) in test balanced
  accuracy, paired over the 14 subjects. Holm within the family, the same machinery as the other Phase 1
  families.
  - **This is the confirmatory test of "phase synchrony adds information beyond band power".**
  - `incremental_sync_over_power` stays, but is reported as capacity-confounded (descriptive).


| # | date | where | decision | reason |
|---|---|---|---|---|
| 23 | 2026-11-09 | **Phase 1 analysis rules** (review board 2026-11-09, rounds 1-2): declared families over lin and nl_coupled, exact sign-flip test, TOST margin 0.05, intersection-union rule, wording limits, a secondary frozen-bank family, analysis sets, diagnostics, no interim inference. Code: provenance per arm, curves and weights stored. | The board showed (verified by the judge) that the Phase 1 "phase" contrasts are not phase-specific: they fire on source-power-only data (sync 0.98 vs power 0.79), they are inflated by dilution, and the fixed-roll surrogate leaks phase. Frozen before any capacity-matched arm has run, before subjects 8-14 and the rest of subject 7, and before the calibration screen (DEVIATION 24). |

### DEVIATION 23: protocol (frozen 2026-11-09, before any capacity-matched arm, before subjects 7-14 finish, before DEVIATION 24 runs)
- **Disclosure: what had been seen.** On 2026-11-05 the judge read the core-arm records (lin and nl_coupled x
  sync / surrogate / power: test accuracy, balanced accuracy, best epoch, best validation loss, epochs run) of
  subjects 1-6 (Drive `resonet_synchrony_runs/phase1/subj_01..06.json`, last written 2026-11-04 by 08:44 UTC,
  git 3c10616) and wrote the interim section of `results/README_results.md`. The half-finished `subj_07.json`
  (08:45 UTC) was not read. No capacity-matched arm exists for any subject.
- **Families (confirmatory; all 14 subjects; test balanced accuracy).** Declared over `CONFIRMATORY_BANKS =
  (lin, nl_coupled)` only (`run.py`), so Holm has m = 2; coupling has m = 1. Rows for other banks, if run, form
  one exploratory family. This replaces "Holm over all four banks", under which banks that never ran entered as
  p = 1.
  - F1 `H2_sync_vs_surrogate` (as published).
  - F2 `incremental_capacity_matched` = (sync+power) - (power+surrogate) (DEVIATION 22).
  - F3 `incremental_sync_over_power` = (sync+power) - power. Promoted from descriptive: it is the second leg
    of the rule below.
  - F4 `coupling` = nl_coupled - lin (sync).
- **Test and intervals.** Primary: exact two-sided sign-flip permutation test of the mean paired difference
  (all 2^14 sign patterns). The exact Wilcoxon signed-rank p is reported alongside (scipy's default falls back
  to a normal approximation with ties). Intervals: t-based 95% CI; t-based 90% CI for equivalence.
- **Outcomes per contrast.** *Superior*: Holm p < 0.05 and mean > 0. *Inferior*: Holm p < 0.05 and mean < 0.
  *Equivalent*: TOST with margin Delta = 0.05 balanced accuracy (8 of 160 test trials; about one third of the
  published phase effect; 80% TOST power at a true 0 when the paired SD is 0.06), Holm across the family on the
  TOST p. Otherwise *inconclusive*. A non-significant superiority test is never reported as "no effect".
- **Rule for "adds beyond the bank's log-powers"** (per bank): F2 superior AND F3 superior
  (intersection-union; no extra correction). F2 alone can be positive under the null through dilution by
  noise-like surrogate dimensions (toy: +0.13, 40/40). Validity check, descriptive: (power+surrogate) - power;
  if its mean is below -Delta, F2 is reported as inflated by dilution.
- **Wording limits.**
  - F1 compares sync with a surrogate that uses one fixed roll per oscillator for every trial. It is neither a
    phase null (it leaks cross-oscillator phase: 0.42-1.00 on phase-only synthetic data, also at 4.5 s) nor
    capacity-free. Report it as "sync vs the published fixed-roll surrogate".
  - The F2/F3 rule may be called "phase" or "synchrony" only if DEVIATION 24 shows that F2 and F3 are calibrated
    on its null and source-power-only generators and sensitive on its phase generator. Otherwise the result is
    worded "zero-lag cross-oscillator structure beyond the bank's 16 log-powers". (The bank's "power" is itself
    the power of learned spatial mixtures, so it contains cross-channel phase; DEVIATION 8.)
- **F5, secondary family: phase beyond sensor band power** (frozen now; post hoc on the saved trained banks of
  the `{bank}:sync+power` arms, all 14 subjects; Holm m = 2).
  - Contrast: (B + sync) - (B + surrogate_random), where sync and surrogate_random (a fresh uniform circular
    roll per trial and per oscillator) come from the frozen bank, and B is a train-fitted band-power baseline
    computed from the same trials.
  - Head: L2 logistic regression; C chosen on the seed-42 validation carve; fitted on train minus the carve;
    scored on test (balanced accuracy).
  - B is the first of [FBCSP, filter-bank log-variance] (definitions in DEVIATION 24) that passes DEVIATION 24's
    calibration and sensitivity criteria. If neither passes, F5 is descriptive and no Phase 1 result is called
    "phase".
- **Analysis sets.** F1-F5 use all 14 subjects. Questions generated by the interim (does the surrogate readout
  train at all; is power >= sync under Hz initialisation) are answered on subjects 7-14 only, descriptively
  (n = 8; minimal detectable dz 1.34).
- **Pre-declared diagnostics (descriptive).**
  - D1 optimiser failure: training loss at the best epoch less than 0.05 nats below epoch 0 (needs the stored
    curves; arms run after this entry).
  - D2 fixed-feature probe: on each saved trained bank, train-CV logistic regression on sync, fixed-roll
    surrogate and random-roll surrogate features. A surrogate that the probe decodes while the trained readout is
    near chance is a training failure, not an information floor.
  - D3 (power+surrogate) - power (the dilution handicap).
  - D4 drift of learned f0 and bw from initialisation.
- **Baselines (descriptive):** filter-bank log-variance + shrinkage LDA, FBCSP + shrinkage LDA, filter-bank
  tangent space + logistic regression; same cap, validation carve and windows (definitions in DEVIATION 24).
- **No interim inference.** `summarize()` and the notebook print and store means with n per arm, but p-values,
  CIs, dz and Holm only for contrasts with n = 14 (`stats.contrast_table(min_n=...)`). If the run stops before
  14 subjects, Phase 1 is descriptive.
- **Code changes (no change to any result).**
  - Provenance per arm: git, code-dirty flag, time, device, GPU, torch/CUDA/numpy/scipy/python versions.
  - A resume keeps the original file-level metadata, appends to `resumes`, and refuses a different config or a
    changed data file.
  - Atomic JSON writes. Learning curves per arm. Trained weights per arm in `results/phase1/weights/` (on Drive;
    git-ignored).
  - Proof of no change in results: a CPU smoke run of every arm at ac481c4 and with the patch gives identical
    accuracy, validation loss, best epoch and learned f0/bw; `tests/test_run_provenance.py`.
- **Judgement call logged (band).** The 1-40 Hz bank band (TASK_PITCH1 §2) meets a 4-125 Hz zero-phase cache:
  6 of 16 oscillators (1.0-3.4 Hz) sit at or below the high-pass, and none covers high gamma. Phase 1 keeps the
  spec; a 4-112 Hz sensitivity run (rk4 sub-steps >= 3) is exploratory, after Phase 1. (The published bank,
  omega ~ U(0.5, 4) with h = 0.01, sat at about 0.2-1.6 Hz.)

| # | date | where | decision | reason |
|---|---|---|---|---|
| 24 | 2026-11-09 | **Calibration screen of the Phase 1 contrasts** (`resonet_sync/screens/calib_p1.py`; review board 2026-11-09). Synthetic only, CPU, lin bank. Decides what the Phase 1 contrasts may be called (DEVIATION 23); it changes no Phase 1 test, margin or analysis set. | The board showed the Phase 1 "phase" contrasts fire on source-power-only data and are inflated by dilution, and that the fixed-roll surrogate leaks. None of the contrasts has been checked against generators with known ground truth. Frozen after the difficulty pilot (seed 999) and before any evaluation seed. |

### DEVIATION 24: protocol (frozen 2026-11-09, after the seed-999 pilot, before any evaluation seed)
- **Data.** 16 channels, 4.5 s at 250 Hz, 4 classes; 80 train trials per class (the Phase 1 seed-42 15%
  validation carve inside them) and 40 test trials per class. Four narrowband Gaussian sources (10, 12, 20,
  24 Hz; 3 Hz wide; fixed normalising constant, so per-trial power varies) mixed by a fixed random matrix, plus
  white sensor noise and a 4 Hz high-passed brownian background. Channels standardised with train statistics.
- **Generators.**
  - G0, nothing: labels permuted independently of the data.
  - G1, source power only: class k = amplitude drop (ERD) of source k by 0.15.
  - G1r, rank-1 power only (Halvorsen's dilution null): class = amplitude of the 10 Hz source, 1 - 0.15 y.
  - G2, coupling only: source 2 = rho x (source 1 rotated by the class lag {0, 45, 90, 135} deg) +
    sqrt(1 - rho^2) x independent 10 Hz noise, rho = 0.4. Every source's spectrum is identical in every class.
  - Difficulty was set on pilot seed 999 only, so that the generator-truth oracle (source log-power, or the
    source 1-2 coherency) scores about 0.75 (`results/calib_p1/difficulty.json`: G1 0.68, G1r 0.73, G2 0.74).
- **Seeds.** Evaluation seeds 3000-3009 (never used before), one dataset per generator and seed.
- **Arms** (the Phase 1 training loop and config, lin bank, n_channels = 16): power, sync, surrogate,
  surrogate_random, sync+power, power+surrogate, power+surrogate_random. `surrogate_random` draws a fresh
  uniform circular roll per trial and oscillator (`readouts.py`, test `test_surrogate_random_rolls_each_trial_independently`).
- **Post hoc on the trained sync+power bank (frozen):** features sync, fixed-roll surrogate, fresh-roll
  surrogate, power; baselines FBLV (log-variance in 9 causal 4th-order Butterworth bands, 4-40 Hz, first 0.5 s
  dropped) and FBCSP (one-vs-rest CSP per band, 2 filter pairs per class, 10% shrinkage, fitted on train minus
  the carve). Head: L2 logistic regression, C chosen on the carve. Also the generator-truth oracle and the D2
  probes.
- **Contrasts** (paired over the 10 seeds): F1 sync - surrogate; F1r sync - surrogate_random; F2 (sync+power)
  - (power+surrogate); F2r (sync+power) - (power+surrogate_random); F3 (sync+power) - power; D3
  (power+surrogate) - power; F5a (FBCSP+sync) - (FBCSP+surrogate_random); F5b (FBLV+sync) -
  (FBLV+surrogate_random).
- **Criteria** (margin 0.05, as in DEVIATION 23). *Valid* on a generator: the 90% t-CI lies inside +/-0.05.
  *Sensitive* on G2: exact sign-flip p < 0.05 and mean >= 0.05. A contrast *passes* if valid on G0, G1 and G1r
  and sensitive on G2. (Conservative: with n = 10 and an arm-difference SD of about 0.05, a perfectly calibrated
  contrast fails the validity check on one generator about 14% of the time; failing restricts wording, never
  adds a claim.)
- **Pre-committed mapping to Phase 1 (DEVIATION 23).**
  - F2 and F3 both pass: the Phase 1 F2/F3 rule may be called "phase synchrony beyond the bank's log-powers".
    Otherwise it is worded "zero-lag cross-oscillator structure beyond the bank's 16 log-powers".
  - F5 uses B = FBCSP if F5a passes, else FBLV if F5b passes; if neither, F5 is descriptive and no Phase 1
    result is called "phase".
  - F1 is worded "sync vs the published fixed-roll surrogate" whatever happens here.
- **Excluded (proportion).** nl_coupled: its coupling is linear and far below locking (board, Volkov), and its
  RK4 loop costs about 5x on CPU. A 3-seed transfer check on Tom's GPU is optional.
- **Disclosure.** One timing job ran G1 on pilot seed 999 with the final settings before this freeze. It showed
  F2 = +0.19 and F5a = -0.05 (FBCSP + sync scored below FBCSP + fresh-roll surrogate). Pilot data only; the
  criteria above were set from the board's round-2 proposals, not from it.

| # | date | where | decision | reason |
|---|---|---|---|---|
| 25 | 2026-11-09 | **hc-5 Gate 1 v2** (`laminar.simulate_laminar_v2`, `generator_mix_fit`, `screens/gate1_v2.py`; recorded edit of `docs/PREREG_H3_GENERATOR_MIX_DRAFT.md`, section "Gate 1 v2"). The generator-mix statistic is computed from two fitted sink/source CSD templates (generator powers by projection), not from two picked sites. Forward model with disc sources, shared septal drift, CA3-EC coherence, perisomatic/DG theta, absolute noise, per-session probe offset, thickness and gains. Criteria: equivalence relative to the target effect, n = 7 resampled power, identification within one site, a depth-shift QC. | Review board 2026-11-09 (Echeverria, Novak, Volkov; verified by the judge): the v1 site picker never found the radiatum sink, a 10 um probe offset reversed the must-move response, and the replacement site rule also failed. Designed on dev seeds 2100-2109; the validation seeds 2200-2209 and the geometry pass map run on the frozen code. The v1 runner and its dry runs are kept unchanged. |

#### DEVIATION 24, outcome (2026-11-09; `results/calib_p1/results.json`, git af002bd; fig 9)
- **Pre-committed mapping applied:** F2/F3 are worded "zero-lag cross-oscillator structure beyond the bank's 16
  log-powers"; F5 has no baseline that passes, so **F5 is descriptive and no Phase 1 result is called "phase".**
- F2 fires on source-power-only data (G1: +0.181, 10/10) and on the dilution null (G1r: +0.117, while F3 =
  +0.011). D3 = (power+surrogate) - power is -0.09 to -0.18 on informative data, so F2 is inflated by dilution
  whenever classes are decodable. The intersection-union rule (F2 and F3) also fires on G1 (F3 +0.094).
- F5 (frozen bank) is sensitive to pure coupling (G2: FBCSP +0.136, FBLV +0.118; 10/10 each) and close to
  calibrated, but each misses one validity check narrowly: FBCSP on G1r (90% CI upper bound +0.053) and FBLV on
  G0 (lower bound -0.052). The criteria are not revisited.
- D2 on the frozen banks: the fixed-roll surrogate features decode 0.32-0.39 with a probe against 0.27-0.30 for
  the trained surrogate readout (partly a training failure, partly a weak signal); the fresh-roll surrogate is at
  chance (0.24-0.27).

#### DEVIATION 24, outcome addendum (board 2026-11-09, round 3; append-only corrections)
- **The "about 14%" false-invalid figure was wrong.** With n = 10, a 90% t-CI and margin 0.05, a contrast with zero
  bias passes all three validity checks only about 50% of the time at a paired SD of 0.05, and 26% at F5b's SD on
  G0 (Halvorsen, Asante simulations). The two narrow F5 misses therefore say nothing about F5's bias.
- **The misses go in opposite directions.**
  - F5b's miss is on G0 (permuted labels), where its expected difference is exactly 0: it is noise.
  - F5a's miss on G1r (+0.027, 7/10) is weak evidence of a liberal bias. On G1r FBCSP (0.57) is weaker than FBLV
    (0.65), so sync can pick up power that FBCSP misses. A future F5 should use the strongest power baseline
    (FBLV and FBCSP together, or a baseline chosen on the validation carve).
- **G2 is "coupling only" at the source level only.** The class lag's cosine enters the zero-lag channel covariance
  (FBLV 0.60, FBCSP 0.58 on G2). F5's G2 sensitivity is to quadrature (lagged) coupling beyond the zero-lag
  covariance the baseline already sees. A cleaner sensitivity generator (lags of +/-90 deg only, so the class
  lives in the imaginary cross-spectrum) is the design for any future calibration.
- **D2 wording.** DEVIATION 23 says the D2 probes are "train-CV". As implemented (here and in
  `scripts/phase1_posthoc.py`) they are fitted on train minus the carve and scored on test, like every other arm.
- **Not re-run, by design.** The Phase 1 wording stays as pre-committed. Any future "phase" claim needs a new,
  pre-registered calibration with >= 20 datasets per generator, so that its validity check has power.

| # | date | where | decision | reason |
|---|---|---|---|---|
| 26 | 2026-11-09 | **Gate 1 v2 status revised to FAIL** (review board round 3). The radiatum-depth-shift cell is gating again (it was made non-gating on dev seeds after failing: a forking path). Gate 1 v2 also fails with depth-extended generators and a strong CA3-EC coherence change. `laminar._pair_score` closed form corrected (wrong conjugation). The hc-5 documentation gives 8 sites per shank (16 anatomical groups, 128 channels). Gate 1 v3 is deferred until Gate 0 (probe geometry, from the hc-5 QC) is known. | Halvorsen, Echeverria, Volkov and Novak independently; the judge re-ran the failing cells. |

### DEVIATION 26: details
- **Validation seeds 2200-2209, 16 x 50 um, under the original rule (radiatum shift gating): FAIL.**
  - The radiatum shift moves Delta by +0.27 [-0.23, +0.78] (paired against the null cell: +0.23), about 75% of the
    target. The depth-shift QC detected it in 3/10 sessions (6/10 on dev).
- **New failing cells** (judge re-runs):
  - CA3-EC coherence PLV 0.95 -> 0.3: Delta -0.236 [-0.315, -0.157], 1/10 positive (PLV 0.75 gave -0.02).
  - Depth-extended generators (Gaussian sinks, SD 40-80 um; Echeverria): EC +20% falls to +0.208 (ideal 0.365) with
    identification 0.7; a CA3 sink spread of 40 vs 80 um gives -0.080 [-0.243, +0.084]. Because the simulator and
    the fitted templates share one point-depth family, v2 was only tested on shapes it can represent exactly
    (an "inverse crime").
  - The EC-asymmetry cell (fundamental fixed) and five other cells reproduce the null cell's draw within 0.002 under
    common random numbers; they test nothing.
- **Null bias claim rejected.** The null cell's +0.04-0.05 on dev and validation was seed luck: 40 fresh null
  sessions give -0.012 [-0.031, +0.007], and n = 7 subsets reject at 0.034. The A/B processing is exactly
  antisymmetric under a seed swap.
- **`_pair_score` bug** (Novak): the refinement score had the wrong conjugation (up to 64% error for a complex S;
  test against explicit QR added). At 16 x 50 um it moved only the lag cells (225 deg: per session up to 0.077); the
  coarse search was correct.
- **Requirements for a Gate 1 v3** (not built; only worth building if a dataset passes Gate 0):
  - generators with depth extent and distributed returns; templates from the data (slow-gamma CSD for the CA3 /
    radiatum profile, mid-gamma CSD for the EC3 / l-m profile) or with a width parameter;
  - graded CA3-EC coherence (0.9 to 0.3) plus a fitted-coherence QC;
  - graded DG weight and phase;
  - nuisance criteria on paired effects (cell - null, same seeds), with the bound fixed at ln(1.44)/3;
  - the radiatum shift gating, or a positive result pre-registered as "EC/CA3 ratio OR CA3 sink-depth change";
  - fresh validation seeds (2300+), disclosing that 2200-2209 have been seen.
- **hc-5 geometry.** The data description lists 16 anatomical groups over 128 channels, i.e. 8 sites per shank.
  Site spacing is not given. Buzsaki-type probes are 20 um, which would give 140 um, far short of the >= 400-500 um
  the generator-mix statistic needs; 8 x 20 um failed every cell in the (stopped) pass map. Gate 0 must still be
  run on the QC output, but the expected outcome is FAIL, i.e. the CSD route is infeasible on hc-5.

| # | date | where | decision | reason |
|---|---|---|---|---|
| 27 | 2026-11-09 | **ResoNet v2 design study** (`resonet_sync/v2.py`, `screens/v2_study.py`, `scripts/v2_hgd_dev.py`; `docs/CRITIQUE_AND_V2.md`). Exploratory, outside the frozen Phase 1: v2 arms are never in `CORE`/`ALL`, write to their own folders, and change nothing in DEVIATIONS 23-24. | Forensics on the published checkpoints (`scripts/published_forensics.py`): the trained oscillators sit at 0.36-1.55 Hz on 4-125 Hz data, so the published model does not resonate and its sync readout is a covariance-type feature of spatial projections. The Hz bank kept 0.16-0.40 Hz bandwidths. v2 is a redesign in which lagged coupling is separable from source power by construction. Criteria below frozen before the final synthetic run. |

### DEVIATION 27: v2 synthetic study, protocol (frozen 2026-11-09, before any evaluation seed 3600-3609)
- **Model** (defaults of `v2.py`, no tuning): 8 resonators log-spaced 4-40 Hz with constant-Q bandwidths; 4 real
  spatial filters per band, no tanh; within-band cross-spectral readout split into power, Re coherency (zero-lag)
  and Im coherency (lagged); first 0.5 s of each trial dropped; BatchNorm, dropout 0.25, linear head. Phase 1
  training loop unchanged.
- **Data.** Generators G0, G1, G1r, G2 of DEVIATION 24 (same frozen difficulty) plus **G2i** (lagged coupling
  only: two source pairs at 10 and 20 Hz; the class is which pair is coupled and the sign of its +/-90 deg lag;
  rho = 0.25 set on pilot seed 999 so the oracle scores about 0.75). Evaluation seeds 3600-3609, never used.
- **What was seen before this freeze.** Development seeds 3500-3502, stopped early
  (`results/v2_study/dev_partial_STOPPED.json`: G0 x2, G1 x3, G2i x2), and untrained-feature probes on seed 3500.
  No design or hyperparameter was changed after the first implementation. One change was made before the final
  run: the null arm draws its shifts from a private generator, so both arms of L share augmentation and dropout
  streams (variance reduction only).
- **Primary contrast.** L = (cov+lag) - (cov+lag_null): what the lagged block adds beyond band covariance, against a
  dimension-matched block whose lag information is destroyed by a fresh circular shift per trial and projection.
- **Criteria** (margin 0.05; n = 10 datasets per generator).
  - *Valid* on G0, G1 and G1r: the 90% t-CI of L lies inside +/-0.05.
  - *Sensitive* on G2i (primary) and G2: exact sign-flip p < 0.05 and mean >= 0.05.
  - L **passes** if all five hold. (Power: with the paired SD of about 0.03 seen in development, an unbiased
    contrast passes one validity check with probability above 0.99.)
- **Descriptive.** R = cov - power; D = (cov+lag_null) - cov (cost of null dimensions); Q = v2 cov+lag - v1
  `lin:sync+power` on G1, G1r, G2, G2i; v2 against FBLV and FBCSP with shrinkage LDA and against the oracle.
- **Mapping.** If L passes, L is the contrast to pre-register for a real-data test of "time-lagged coupling
  beyond band covariance" (with v2 tuned only by train-split cross-validation on development subjects,
  `scripts/v2_hgd_dev.py`). If L fails, v2 is reported as a decoder redesign only.
- **Scope of the claim.** Im coherency of real spatial projections is zero for instantaneous mixing of independent
  sources whatever their powers. A non-zero value means time-lagged structure between spatial components. That
  includes stimulus-locked sequences and lagged common input; it is not, by itself, coupling between oscillators
  (`docs/NONIDENTIFIABILITY.md`).

#### DEVIATION 27, addendum 1 (2026-11-10): first real-data development results and the closed-form path
Written before the closed-form path was run on its synthetic seeds or on HGD subjects 2-6.
- **Real data used.** Training recordings of HGD subjects 1-6 only (public data, `scripts/build_hgd_dev_cache.py`,
  published preprocessing). The test recordings were not downloaded. All numbers are k-fold cross-validation inside
  the training split (`scripts/v2_hgd_dev.py`), i.e. development results.
- **What was seen.**
  - Trained v2 (12 bands 4-100 Hz x 4 filters, random init), subject 1, 5 folds: v2 0.65-0.68; v1 `lin:sync+power`
    0.52; FBLV + LDA 0.74; **FBCSP + LDA 0.89**. Subject 3, 2 folds; subject 2, 1 fold (stopped).
  - CSP-initialised trained v2 (same size), subject 1 fold 0: power 0.75.
  - Diagnostic on subject 1, folds 0 and 2 (untrained bank, shrinkage LDA): with CSP filters, power scores 0.75-0.81
    at 4 filters per band, 0.92 at 8 and 0.91-0.94 at 16 (FBCSP 0.94-0.95). The lag block alone scores 0.50-0.78.
- **Decision.** The number of spatial filters per band, not gradient training, limits v2 on real data. Real-data
  development continues on the **closed-form path** (`resonet_sync/v2_probe.py`, `scripts/v2_probe_run.py`): CSP
  filters from the training trials, untrained constant-Q resonators, shrinkage LDA per feature set. HGD
  configuration fixed now: 12 bands 4-100 Hz x **8** filters. The slow trained runs were stopped; their partial
  files are kept in `results/v2_hgd_dev/`.
- **Controls added for the lag block.** `im_null` (fresh circular shift per trial and projection) and `im_evoked`
  (same-class partner trial: only structure time-locked to the trial start survives; needs labels, analysis only).
- **Closed-form path on synthetic ground truth.** Same generators and criteria as the trained path above, new
  seeds 3700-3709, 8 bands 4-40 Hz x 4 CSP filters: L = acc(power+re+im) - acc(power+re+im_null) must be valid on
  G0, G1, G1r and sensitive on G2i and G2.
- **Status of any real-data number here: descriptive development.** A confirmatory test needs a frozen
  configuration and untouched data (subjects 7-14, or the test recordings).

#### DEVIATION 27, addendum 2 (2026-11-10): outcomes
- **Synthetic, trained path** (`results/v2_study/results_final.json`, seeds 3600-3609): L passes. Valid on G0
  (-0.009 [-0.031, +0.013]), G1 (-0.022 [-0.035, -0.009]) and G1r (-0.013 [-0.043, +0.017]); sensitive on G2
  (+0.161, 10/10) and G2i (+0.466, 10/10). v2 cov+lag exceeds v1 `lin:sync+power` by +0.04 to +0.10 on G1, G1r, G2
  and G2i.
- **Synthetic, closed-form path** (`results/v2_probe/synthetic.json`, seeds 3700-3709): L passes. Valid on G0
  (+0.002), G1 (-0.007), G1r (+0.011); sensitive on G2 (+0.126) and G2i (+0.378). The lag block alone is at chance on
  power-only data (0.256, 0.267) and scores 0.696 on lag-only data.
- **Published model on the same kind of data** (`results/published_null/results.json`, seeds 3200-3209, 4 Hz
  high-pass): sync - surrogate = +0.309 [+0.266, +0.352] on power-only data.
- **HGD development subjects 1-6** (closed-form path, 5-fold CV inside the training split; descriptive):
  v2 power 0.881, FBCSP + LDA 0.891, FBLV + LDA 0.771; lag block alone 0.736, its shift-null 0.250, evoked control
  0.252; L = +0.036 (6/6 subjects positive); power + re + im 0.879.
  - Controls (`controls.json`): lag block after regressing out power and zero-lag coherency 0.327; lag information
    concentrated at 10-31 Hz; fitted inter-channel delays within about +/-0.2 ms.
- **Mapping applied.** L passed on ground truth, so L is the contrast to pre-register for a confirmatory real-data
  test. Proposed frozen configuration: closed-form path, 12 bands 4-100 Hz x 8 CSP filters, 0.5 s skip, shrinkage
  LDA; contrasts: v2 power vs FBCSP (equivalence), lag block alone vs its null, and L; data: subjects 7-14 or the
  test recordings. Not run; needs Tom's go-ahead because it uses up untouched data.

#### DEVIATION 27, addendum 3 (2026-11-13): review-board findings on v2, written before the new runs
Board session 2026-11-13 (`docs/REVIEW_BOARD.md`). Its checks, re-run by the judge, show:
- **Neither "lag block alone above its null" nor L > 0 is evidence that coupling differs between classes.**
  - A world with the *same* lagged coupling in every class, where only the amplitude of the coupled sources depends
    on the class, gives lag block alone 0.76 and L = +0.037 (Halvorsen, Zhao). The real-data pattern is 0.74 and
    +0.036.
  - L is positive (+0.055 to +0.075) when the lag block is only a noisy linear function of power, with no unique
    information (Halvorsen). Its noise null controls dimension, not redundancy.
  - The "regress out power" control (`im_resid`) scores about 0.32-0.35 both with and without unique lag
    information (Zhao; judge), so it decides nothing.
- **The immunity claim was too strong.** With coherency normalisation the lag block responds to the power of
  unrelated sources as soon as any lagged structure exists, even a class-independent one (Echeverria: 0.58-0.61).
  The unnormalised imaginary cross-spectrum stays at chance there (judge: 0.24).
- **Consequences.**
  - The mapping in addendum 2 is withdrawn: L is **not** pre-registered as a confirmatory contrast, and no
    confirmatory real-data test of the lag block is run. The untouched data stay untouched.
  - Added now: an unnormalised lag block (`im_raw`, `im_raw_null` in `v2_probe.py`) and two trap generators
    (`v2_study.make_trap`: `T_amp`, `T_bg`), run on new seeds 3800-3809 (`scripts/v2_probe_run.py --traps`).
  - The development analysis on HGD subjects 1-6 is repeated with the unnormalised lag block
    (`--hgd --tag raw`; same folds; descriptive). Reading fixed in advance: if `im_raw` alone falls to chance, the
    earlier 0.74 came from power changes of unrelated sources through the normalisation; if it stays well above
    chance, the sources that movement modulates have lagged structure themselves. Neither outcome shows a change
    in coupling.

#### DEVIATION 27, addendum 4 (2026-11-14): higher-order resonators, written before the decoding run
- **Why.** Board finding (Asante): on real EEG the one-pole resonator isolates its band poorly, because its skirt
  falls only 6 dB per octave and the spectrum falls faster. Reproduced on the training recordings of subjects 1-6
  (`scripts/resonator_leakage.py`, `results/v2_probe/resonator_leakage.json`): the share of a resonator's output
  power that comes from its own -3 dB band is 0.17-0.77, and for the four bands above 40 Hz 68-81% of the output
  comes from below 40 Hz.
- **Change.** `CrossSpectralBank(order=n)` cascades the pole n times (a complex gammatone), each pole widened so
  that the -3 dB half-width is unchanged. Default `order = 1` reproduces every earlier number bit for bit. With
  order 4 the in-band share is 0.56-0.86 and the share from below 40 Hz in the bands above 40 Hz falls to 10-56%
  (a 4th-order Butterworth band-pass: in-band 0.86-0.98).
- **Run.** The development analysis on HGD subjects 1-6 is repeated with `--order 4` (same folds, same 12 bands x 8
  CSP filters; descriptive, no inference; test recordings not used).
- **Reading fixed in advance.** This is a check of the front end, not a new test of anything.
  - If the feature-set accuracies stay within about 0.02 of the order-1 values, the leak did not drive the earlier
    development numbers, and order 4 becomes the recommended setting because its bands mean what their labels say.
  - If v2 power moves by more than that in either direction, the earlier "v2 matches FBCSP" statement is tied to
    the resonator shape and is reported as such.
  - Nothing in this run can support a statement about coupling (addendum 3).
- **Outcome** (2026-11-14; `results/v2_probe/hgd_b12s8_4-100Hz_o4/`; order-1 values from
  `hgd_b12s8_4-100Hz_raw/`, which reproduces the first run exactly on the shared keys).
  - v2 power: 0.881 -> **0.908** (+0.027; per subject +0.022, +0.031, -0.003, +0.013, **+0.113**, -0.012). This is
    above the 0.02 mark, so the second reading applies: **the decoder comparison depends on the resonator shape
    and is reported per order.** v2 power minus FBCSP: -0.010 (SD 0.033) with order 1, +0.017 (SD 0.047) with
    order 4. Most of the change is subject 5 (0.881 -> 0.994). Neither difference is claimed.
  - Lag block alone: 0.736 -> 0.732; unnormalised 0.693 -> 0.691; shift-nulls 0.250 -> 0.239 and 0.285 -> 0.277;
    evoked control 0.252 -> 0.254. All within 0.02: the band leak did not drive the lag results.
  - power + re + im: 0.879 -> 0.890, now below power alone (0.908). L: +0.036 -> +0.016 (6/6 positive). L is not a
    contrast any more (addendum 3).
  - Follow-up started after seeing this outcome (so it is exploratory): the per-band profile with order 4
    (`scripts/v2_probe_controls.py --order 4`), to see in which bands power gained.
- **Synthetic study rerun on the same seeds 3700-3709** to add the unnormalised block to
  `results/v2_probe/synthetic.json`. Every previously stored value is reproduced exactly (maximum difference 0);
  L still passes. Unnormalised block alone: G0 0.246, G1 0.260, G1r 0.248, G2 0.464, G2i 0.694.

#### DEVIATION 27, addendum 5 (2026-11-14): artefact control with the central sensors, written before the run
- **Why.** Board item accepted but not run (Echeverria): the movements are executed, so eye, cranial-muscle and
  movement artefacts can carry class information. The board asked for a control before any physiological reading.
  The sentence at stake is the reading of addendum 3: "the sources that movement modulates have lagged structure
  themselves."
- **Control.** The closed-form analysis on HGD development subjects 1-6 (same folds; training recordings only) is
  repeated with **only the 44 sensors over sensorimotor cortex** (the selection of Schirrmeister et al. 2017:
  FC, C, CP rows and their half-row neighbours). Frontal, temporal, parietal-occipital and edge sensors, where eye
  and muscle artefacts are strongest, are left out. Resonator order 4, 12 bands 4-100 Hz, 8 CSP filters
  (`scripts/v2_probe_run.py --hgd --order 4 --channels central44`). Descriptive; no inference.
- **Reading fixed in advance.**
  - If the unnormalised lag block alone stays at 0.50 or above (128 sensors, order 4: 0.691), sensors outside the
    sensorimotor region are not needed for it. Eye artefacts and muscle artefacts confined to the edge of the cap
    are then unlikely to be its main source.
  - If it falls below 0.40, the lag result depended on the non-central sensors, and artefacts become the leading
    explanation. Between 0.40 and 0.50 the control is inconclusive.
  - In no case does the control exclude muscle activity that reaches central sensors, or movement artefacts common
    to all sensors. It cannot support any statement about coupling (addendum 3; `docs/NONIDENTIFIABILITY.md`,
    construction 3).
  - The per-band profile with order 4 (running, started after the order-4 outcome) is read together with it: lag
    information concentrated at 10-31 Hz fits mu/beta rhythms; information concentrated above 40 Hz would fit
    muscle activity.

| # | date | where | decision | reason |
|---|---|---|---|---|
| 28 | 2026-11-13 | **Phase 1 is stopped at 6 of 14 subjects** (review board 2026-11-13, decision 3; Tom delegated the decision to the board). The interim section of `results/README_results.md` stays as a descriptive, incomplete record. No capacity-matched arm was ever run. The notebook keeps the code but its Phase 1 switches now default to off. | The reasons are independent of the interim outcome: (a) DEVIATION 24 fixed in advance that no Phase 1 result can be called "phase"; (b) the checkpoint forensics show that the published model, which Phase 1 replicates, does not resonate with the data, so the replication cannot inform Pitch 1; (c) the remaining GPU time is better spent on the control the manuscript lacks (`scripts/bandwidth_control.py`). Counter-position recorded in `docs/REVIEW_BOARD.md`. |
