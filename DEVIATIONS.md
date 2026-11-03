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
