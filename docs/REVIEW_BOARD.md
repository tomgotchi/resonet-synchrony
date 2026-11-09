# Review board, 2026-11-03

Six independent reviewer personas were run as agents with read-only access to both repos while screen 10 stage C
was running. Each returned ranked findings. The judge (the main session) then checked each claim:
- **VERIFIED:** reproduced or confirmed in the code.
- **ACCEPTED:** a reasoned claim, adopted without independent reproduction.
- **REJECTED:** not adopted.
- **PENDING:** needs an experiment.

## The board

| reviewer | persona | scope |
|---|---|---|
| I. Halvorsen | biostatistician, meta-science | inference, adaptivity, error rates |
| T. Echeverria | hippocampal systems neuroscientist | biological realism, hc-5 |
| M.-L. Zhao | information theorist | TE estimation, identifiability |
| A. Volkov | nonlinear-dynamics physicist | generators, dynamics, coupling inference |
| K. Asante | closed-loop signal-processing engineer | phasetrack baselines, real-time validity |
| P. Novak | adversarial research-software auditor | bugs, leakage, reproducibility |

## Points where several reviewers converged

1. **Rhythmic (narrowband) common input has never been put through the within-session rule** (Echeverria C2, Zhao
   C2; Volkov M1 implicitly).
   - `mixed` uses white common noise. In a latent LDS, a shared rhythmic drive needs its own state dimensions, so
     it lands in A, which is the place the method reads as coupling.
   - In hc-5 the septal/EC theta drive is exactly this case. **ACCEPTED as the top-priority missing trap.**
2. **Pairwise phase TE is confounded by common input by construction** when each oscillator is only
   half-observed: x = Re z, and a phase lag is not the full state (Zhao C1, with numbers; Volkov M1).
   - The S_inst_dn guard is therefore load-bearing, not optional. No phase-only estimator can do the separation.
   - **ACCEPTED.** This is consistent with every screen since 4.
3. **The patching is open-ended whack-a-mole, tuned on one generator family** (Halvorsen C1; Echeverria; Zhao M1:
   the rotation fix moves both regimes to a leakage minimum but does not remove the leakage).
   - **ACCEPTED.** The rule's false-positive rate is set by the bias magnitudes that no one has mapped (Halvorsen
     C2), not by the bootstrap.
4. **Noise matching with white noise based on a diagonal-R LDS will not transfer to 1/f, spatially correlated real
   noise** (Echeverria M7, Zhao M2, Halvorsen M3). **ACCEPTED** as a limit; it needs a coloured-noise trap.

## Individual findings and verdicts

### Code defects (Novak), all fixed in this round
- **MAJOR, VERIFIED, FIXED.** `screen10.decide`: one NaN (an unstable LDS fit) made every must-not-fire cell
  "correct". It now marks the cell invalid (`fires=None`) and the whole verdict `valid=False`. Regression test
  added. Stage C partial results checked: 0 non-finite values.
- **MAJOR, VERIFIED, FIXED.** phasetrack `phase2_hc2` integrity gate checked only code_hash, completed and r3_steps.
  It now also requires whl_units, seed, data sha256 and channel to match. Test added for the seed mismatch.
- **MAJOR, ACCEPTED, PARTLY FIXED.** Reproducibility:
  - `requirements.lock` (pip freeze) added to both repos.
  - Resumable `partial.json` has no code hash. Stage C runs at commit d90a214 and its directory was fresh. Future
    screens will store a code hash.
- **MINOR, VERIFIED, FIXED.**
  - `lds.simulate_observed` reused the state-noise RNG stream.
  - `data_hc5.runs_from_ids` merged adjacent runs that had different ids.
  - `train.py` reported a best epoch that could differ from the restored checkpoint.
  - The hard-coded 9/10 threshold now scales with the number of sessions.
- **MINOR, ACCEPTED, NOT FIXED.** `synth.py` `if p.get("stream")` treats 0 as "no stream". This is harmless,
  because numpy zero-pads seed lists, and it was verified.

### Dynamics (Volkov)
- **CRITICAL, VERIFIED.** Euler-Maruyama is not rotation-equivariant. The effective damping is lambda + w^2 dt / 2:
  -5.32, -5.03 and -4.69 s^-1 at 7, 8 and 9 Hz, against -6.28 nominal (judge recomputed this).
  - So the F cells also change bandwidth and amplitude by about 7%. They test an integrator artefact as well as
    the estimator.
  - Fix: exact co-rotating (Lawson/OU) integrator. **PENDING, to be implemented as an option**, so old datasets
    stay byte-identical.
- **CRITICAL, ACCEPTED.** Diffusive coupling also raises self-damping (Laplacian diagonal), so "K up" means
  bandwidth up and amplitude down. Required trap: self-damping matched, A = 0. **PENDING (screen 11).**
- **MAJOR, ACCEPTED.** Drift/diffusion estimation on complex amplitudes:
  - Phi = C(2) C(1)^-1 is immune to white observation noise at lags >= 1.
  - Off-diagonal drift reads coupling; off-diagonal diffusion reads common input.
  - It is invariant to a uniform frequency shift by construction. **PENDING as a candidate statistic.**
- **Missing traps ACCEPTED:** frequency-spread change, damping change, sigma change, delayed coupling (must fire),
  phase-only coupling (must fire), weak coupling (K about 5-10).

### TE estimation (Zhao)
- **MAJOR, ACCEPTED.** The bin-alignment derivation is consistent with the diagnostics. Bin dithering "adding
  nothing" was a *test design error*: `diag10b` shifted the past and future grids together, so the relative offset
  never changed (judge confirmed in the code). The README statement is corrected.
- **MAJOR, ACCEPTED.** LDS block norms are basis-dependent, which is already known (screen 5/6). S_inst_dn mixes
  off-diagonal Q with hidden-coordinate coupling.
- **Proposals PENDING:**
  - KSG-torus TE with a cross-trial surrogate (rotation-invariant, no f_c needed);
  - exact innovations correlation from the steady-state Kalman/DARE solution (replaces Monte Carlo + VAR(25));
  - latent likelihood-ratio coupling statistic (block-diagonal A vs full A).

### Inference (Halvorsen)
- **MAJOR, ACCEPTED.**
  - `boot_ci` is a plain percentile bootstrap. The reviewer's simulated coverage was 0.90 at n=10 and 0.86-0.88 at
    n=7.
  - The 6/7 sign rule has p = 0.0625. PREREG_H3_REDESIGN_DRAFT still names rule v2 and n=7.
- **MAJOR, ACCEPTED.** "All cells correct" with dz of about 7 positives and dz of about 0 nulls carries little
  information.
  - Replace it with a false-positive-rate and power map over *graded* nuisance and common-input magnitudes
    (resampling 10-of-60 sessions).
  - Pre-register the worst-case false-positive rate as alpha.
  - Use exact sign-flip permutation and a log-ratio TOST for the guard.
- **ACCEPTED.** Freeze the rule and stop patching. Confounds found on hc-5 become reported limitations.

### Biology (Echeverria)
- **CRITICAL, ACCEPTED (scope).** "Coupling between CA1 LFP sites" is not well defined: the LFP is a lead field
  times shared generators. The physiologically meaningful hypothesis is the generator mix (CA3 radiatum vs EC3
  l-m), via CSD/ICA, with **EC recorded as an observed covariate**.
- **CRITICAL, VERIFIED.** `data_hc5.bipolar` builds *overlapping* pairs (a-b, b-c). With white sensor noise these
  share an electrode (correlation -0.5), which violates diagonal R. QC-only today; must change before any
  hypothesis use. `parse_xml` ignores skip flags and `find_base` takes the first .xml (both verified).
- **MAJOR, ACCEPTED.** Further traps for real theta:
  - speed-locked frequency drift within windows;
  - sawtooth waveform asymmetry;
  - travelling waves;
  - per-electrode SNR with 1/f noise;
  - a forward-model null through the real pipeline.
- **Missing wheel speed**: maze-only speed gating builds an asymmetry into the design.

### phasetrack (Asante)
- **CRITICAL, ACCEPTED.** Asymmetric researcher degrees of freedom:
  - ecHT/cecHT have a fixed f_c +/- f_c/4 band and no off-centre correction, while the resonators got offset/WL
    correction and the post-hoc DC-zero fix.
  - The reviewer's check: cecHT at 6/7/9/10 Hz gave 54/36/19/40 deg with a 6-10 Hz band, and 30/26/8/21 deg with
    4-12 Hz.
  - Proposed baselines B2w (ecHT on the bank band) and B2f (ecHT with H(w_hat) correction) must be added **before
    hc-2 data are opened**, as an amendment.
- **CRITICAL, ACCEPTED.** NRO's 82 deg comes from the 296 ms prefilter delay. The README claim that the
  resonators beat every literature baseline must be qualified.
- **CRITICAL, ACCEPTED.** Zero-phase resample_poly gives every estimator 40 ms of look-ahead at 250 Hz. It is
  uniform across estimators but invalid for real-time claims. Decimation must be causal, with its delay counted
  in latency.
- **MAJOR, ACCEPTED.**
  - The PLV-argmax latency is not identifiable.
  - Trigger metrics drop false alarms outside the mask.
  - H3c compares gates at different realised coverages.
  - Max-power channel selection may pick SLM/EC rather than the pyramidal layer.
  - SSPE fits EM on the first 2 s of test, which conflicts with the train-only rule.
- **MAJOR, ACCEPTED.** The DC-zero "replication" held in only 7/12 contrasts. The wording should be "partially
  replicated".

## Judge's plan (in priority order)
1. Finish stage C as frozen and report it honestly, with the caveats above.
2. **Screen 11 (synthetic, frozen before running):** rhythmic common-input trap, self-damping-matched trap,
   coloured-noise trap. Use the exact integrator option. Run v3r alongside the new candidate statistics
   (drift/diffusion; exact innovations correlation). Use graded magnitudes and estimate the false-positive rate by
   resampling, not by pass/fail.
3. phasetrack amendment before hc-2: B2w, B2f, causal decimation, NRO qualification, trigger false alarms,
   coverage-matched H3c.
4. hc-5 code: non-overlapping bipolar or CSD, skip flags, unique xml base, an EC covariate; update the H3 draft.

---

# Round 2: the board criticises the judge (2026-11-03)

Each reviewer received the round-1 record, the judge's verdicts and the fix commits, and was asked to criticise
them and to cross-examine the other reviewers.

## Verdicts the board overturned or corrected

| round-1 verdict | round-2 outcome | evidence |
|---|---|---|
| Volkov's drift/diffusion statistic ACCEPTED as a candidate | **REJECTED as a separator of coupling and rhythmic common input** | Zhao: S_drift 3.24 for rhythmic input with no coupling vs 2.11 for true coupling (band-passed, 40 dB); falls 23% from 40 to 0 dB. Halvorsen: hidden rhythmic drive makes the process VARMA (off-diagonal Phi 0.001 -> 0.03). Volkov concedes about 15% leakage at matched synchrony. White common input does land in diffusion. |
| Zero-phase decimation = CRITICAL, 40 ms look-ahead | **MAJOR: latency-budget item** | Asante and Novak independently: the zero-phase FIR has gain 1.000-1.004 at 4-12 Hz (no look-ahead at theta). A causal anti-alias filter costs 4-8 ms (about 12-24 deg at 8 Hz). |
| Euler integrator = CRITICAL, "not rotation-equivariant" | **MAJOR** | Asante: the Euler step commutes with rotation; the defect is a frequency-dependent gain, analytic and compensable. Effect on F/X1/C3 cells stands. |
| "No phase-only estimator can separate" (attributed to Zhao) | **Corrected** | Zhao's claim is broader: no *observation-level* statistic separated them in her tests, including full analytic signals with conditioning. Halvorsen: this is shown for pairwise TE, not proven in general. Volkov: for narrowband units, Hilbert recovers Im z, so half-observation is less severe than stated. |
| `runs_from_ids` fix VERIFIED | **Reverted to the documented default** | Novak: whether NLapCW counts runs or laps is unknown. Splitting could put one run in both train and test. Splitting is now opt-in. |
| `ceil(0.9 n)` threshold as a bug fix | **Reclassified as a design decision** | Halvorsen: at n = 7 it silently selects 7/7 (power 0.29 at dz = 1). Recorded in DEVIATIONS 20 as open for hc-5. |

## Gaps in the judge's fixes (fixed in this round unless noted)
- **NaN tokens.** `results.json` could contain NaN tokens. Now strict JSON (non-finite -> null).
- **RNG pairing.** `kappa_nb` drew from the shared trial RNG, so rhythmic datasets were not paired with their
  baseline. It now has a dedicated stream, and labels are identical (test).
- **Unsupported options.** The new synth options were silently ignored for kinds other than `mixed`. They now
  raise an error.
- **Stage C verdict.** Stage C runs the pre-fix `decide()`. The verdict will be recomputed with the fixed code
  (pre-committed).
- **Coloured noise (not fixed).** The 1/f noise is independent per channel, so the accepted
  *spatially-correlated* noise trap (common reference, overlapping bipolar) is still missing.
- **Rhythmic drive (not fixed).** The drive reuses the white-noise loadings psi and has a fixed nb_f_hz, so
  frequency cells combined with rhythmic input become detuning cells (Echeverria, Volkov, Zhao). To be addressed
  in screen 11.
- **Weak test.** The exact-integrator test checks an identity, not the simulator (Novak, Volkov). An
  equivariance-in-law test (variance and bandwidth equal at 7 and 9 Hz) is still to do.
- **Gate coverage (phasetrack).** The gate should also hash PREREG and the config (Asante).

## Cross-examinations worth keeping
- **Volkov vs Echeverria.** ICA assumes independent sources, but coupled generators are dependent. Instantaneous
  mixing y = Wz alone produces off-diagonal drift proportional to (w_i - w_j), so **volume conduction is itself
  a coupling false positive**. It needs a trap (mixing W, K = 0). CSD is preferred over ICA.
- **Echeverria vs Volkov.** For LFP, mixing creates drift (Phi_x = L Phi_s L^-1). The statistic would measure
  volume conduction.
- **Zhao vs Halvorsen.** Rhythmic common input is not a nuisance; it is the rival hypothesis. Freeze the
  *evaluation protocol* (graded traps, FPR map, alpha), not the statistic.
- **Novak checked others' claims.**
  - Echeverria's bipolar correlation of -0.5 is right, and reaches -0.90 with one noisier shared electrode.
  - Halvorsen's bootstrap coverage reproduced: 0.91 at n = 10, 0.87 at n = 7, and 0.75-0.81 for skewed data.

## Positions on the programme
- **Halvorsen.** Stop developing the rule. Run screen 11 once as a terminal calibration (graded design about 7.9
  core-hours; P(fire) <= 0.05 per must-not-fire cell; power >= 0.8 at K 35). Pre-commit to retiring TE/LDS if
  any rhythmic or rho cell fails. She expects the rhythmic cell to fail.
- **Zhao.** The only candidate that targets the rival hypothesis is a held-out likelihood ratio of a latent model
  *augmented with shared driver oscillators*: null = no unit-to-unit blocks, full = free blocks.
- **Echeverria.**
  - Retire H3' ("coupling between sites") and replace it with a generator-mix hypothesis (CA3 radiatum vs EC3
    l-m), using CSD, one hemisphere, speed matched via the wheel encoder, and EC as an observed covariate.
  - Required first: a forward-model null through the real pipeline.
  - Forbidden before a freeze: any wheel-vs-maze S_TE, S_inst_dn or v3r computation; overlapping bipolar data;
    pooling hemispheres; maze-only speed gating.
- **Asante (phasetrack).** An amendment to freeze before the hc-2 train-only run:
  - causal Butterworth decimation (8th order; E0 uses zero-phase, so decimation delay counts as error);
  - a pre-declared baseline menu B2* selected on train data;
  - train-split bias calibration for every estimator;
  - H3a latency from group delay at f_c;
  - coverage-matched H3c;
  - NRO, NRO with a short IIR prefilter, SSPE fitted on train, and R1dz_uncorrected as descriptive comparisons.

  His own prototype shows that B2w and B2f trade noise for bias (0 dB: 56.6 / 56.9 vs B2 48.1). The DC-zero
  confirmation stands, but the baselines get the same pre-declared, train-selected allowance.

## Judge's revised plan
1. **Stage C:** finish as frozen and report under the pre-commitment (a pass is not confirmation).
2. **Screen 11 = terminal calibration** (Halvorsen's tiered design), with corrections:
   - exact integrator;
   - rhythmic drive with its own loadings, shifting with f_shift;
   - spatially correlated noise;
   - volume-conduction trap.

   Frozen before running, no revisions, retirement pre-committed. Candidate statistics, including Zhao's
   augmented likelihood ratio, are computed but quarantined.
3. **phasetrack:** Asante's amendment, frozen before any hc-2 run.
4. **hc-5:** move from H3' to a generator-mix hypothesis (Echeverria); descriptive and pre-registered only. Tom
   decides.

---

# Board session, 2026-11-05: Tom's questions on closed loop, FFT padding, ICA and the power+surrogate arm

**Header.**
- **Members:** code auditor (Novak), DSP engineer (Asante), statistician (Halvorsen); 1 round.
- **Mode:** single-conversation. The reviewer agents were launched twice and stopped on usage limits both times,
  so one model wrote the three sealed reviews (claims and "check:" lines only), and the judge ran all checks
  afterwards.
- **Code reviewed:** `vendor/resonance_network.py`, `resonet_sync/banks.py`, `readouts.py`, `run.py`,
  `train.py`; `/reso-net/build_hg_cache.py`, `build_nonemg_cache.py`, `run_surrogate.py`; the Colab notebook.

| sev | member | claim | verdict | evidence |
|---|---|---|---|---|
| - | Novak | ResoNet's FFT convolution is zero-padded (L >= 2T), so linear and causal, with no circular wrap | VERIFIED | impulse test on the exact `_linear` path: output before t0 < 5e-8 (float round-off), 1.000 at t0; `banks.py:131-139`, vendor `:419-424` |
| minor | Novak | ICA (non-EMG cache only) is fitted per file on a 400 s crop, separately for the train and test EDF; components are rejected by EOG/muscle/EMG correlation; **no labels** | VERIFIED | `build_nonemg_cache.py:55-63, 108-110`; `y` built only after `ica.exclude` |
| minor | Novak | The test file's cleaning is fitted on test data (unsupervised, transductive); train and test get different decompositions | VERIFIED | same lines. Report it as session-level preprocessing; not deployable online as is |
| - | Novak | The Colab Phase 1 run is unaffected: it uses `subj_cache_hg`, which has no ICA | VERIFIED | `build_hg_cache.py` (the two "ica" hits are "canonical"/"identical"); notebook path |
| - | Novak | Validation and early stopping come from train only | VERIFIED | `run.py:67-70` |
| major | Asante | ResoNet is not causal end to end: the cache uses zero-phase FIR, and the readout is per 4.5 s trial | VERIFIED | `build_hg_cache.py:47` `phase="zero"` |
| major | Asante | "A DC zero removes the separate high-pass" does not hold for ResoNet: the tanh encoder precedes the bank, so drift must be removed before the nonlinearity, and any prefilter on x changes what the encoder sees, so retraining plus a new reference model is required | VERIFIED | vendor `:299` `complex_tanh(x W_enc + b)` before the bank. A DC zero after the encoder is exact for the linear bank (Delta z = bank(Delta I)) but leaves drift in the tanh |
| minor | Asante | The DC zero is a differentiator, not a flat high-pass | VERIFIED | -38 dB at 0.5 Hz, -12 dB at 10 Hz, -2.7 dB at 30 Hz (fs 250) |
| major | Halvorsen | The Colab run has no power+surrogate arm; neither do "all arms" nor the published analysis | VERIFIED | `run.py:30-35`; `run_surrogate.py` `ARMS` |
| major | Halvorsen | `(sync+power) - power` is capacity-confounded (N + 2P vs N dims); power+surrogate is the matched control | VERIFIED | readout dims `readouts.py:46-48`; the published docstring makes the same argument for sync vs power |

**Fixed.** DEVIATIONS 22 (frozen before any all-arms result) adds the `power+surrogate` readout and arms, plus the
family `incremental_capacity_matched`. Tests pass.

**Pending (Tom).**
1. **The closed-loop target:** phase-locked stimulation (latency of about 10 ms; phasetrack's estimators) or online
   decoding (a sliding-window decision within a few hundred ms; ResoNet).
2. **Running the new arms:** about 4 extra arms (sync+power and power+surrogate for lin and nl_coupled). That is
   roughly +65% of the core-arm GPU time.

**Positions.**
- **Novak:** the code is sound for the questions asked. Document the non-EMG ICA as transductive in any paper.
- **Asante:** pick the closed-loop target first. Change nothing in ResoNet until the Phase 1 replication is
  done. The DC-zero shortcut belongs to phasetrack-style estimators, not to ResoNet.
- **Halvorsen:** make `incremental_capacity_matched` the confirmatory test of "phase beyond power".

**Judge's plan.**
1. Let the core run finish.
2. Run the 4 arms for lin and nl_coupled.
3. Report the capacity-matched family as primary for the phase-beyond-power claim.
4. Decide on closed loop separately, after Phase 1.

---

# Board session, 2026-11-09: what should the project do next, with no new real data?

**Header.**
- **Target:** the repo at ac481c4; direction for the project while the Phase 1 Colab run is stopped at 6/14.
- **Members:** all six. **Mode:** agent mode (six independent reviewer agents), 2 rounds. Every member ran its own
  checks; the judge re-ran the decisive ones.

## Findings and verdicts
VERIFIED = re-run or found by the judge; ACCEPTED = plausible, not reproduced.

| sev | member | finding | verdict | evidence |
|---|---|---|---|---|
| crit | Asante | On source-power-only data (4 sources, class = 30% ERD of one, random phases) sync beats power (0.98 vs 0.79); sync-surrogate +0.46; (sync+power)-(power+surrogate) +0.25 | VERIFIED | proto_surr.py re-run: 0.983 / 0.790 / 0.523 / 0.985 / 0.735 |
| crit | Zhao | That signal is source power carried by zero-lag cross-covariance, not phase: channel imaginary coherency is at chance (0.25), real coherency 0.99 | ACCEPTED | a1_check.py (not re-run) |
| crit | Halvorsen | The capacity-matched contrast is positive under "nothing beyond power" by dilution (noise-like surrogate dims) | VERIFIED (toy) | dil.py re-run: +0.130, 40/40; real-readout size PENDING |
| crit | Zhao | The fixed-roll surrogate leaks cross-oscillator phase | VERIFIED | phase-only synth: fixed roll 0.42-0.51, fresh roll 0.19-0.25 (surr_leak.py); 4.5 s stationary narrowband: 0.90-1.00 (Novak r2_both.py, re-run) |
| crit | Asante (r2) | The intersection-union rule still fires on source-power-only data; only (band power + sync) - (band power + fresh-roll surrogate) separated power-only (+0.01) from phase-only (+0.73) | ACCEPTED | r2_rules.py (not re-run) |
| crit | Echeverria, Novak, Volkov | Gate 1 site selection is broken: the ripple dipole is null at the true pyramidal site; "radiatum" = the reversal site or the CA3 return source, never the sink | VERIFIED | seed 2000: runner radiatum 5, true 7; z0 = -10 um gives EC+20% mean -0.185, 0/10; true sites +0.369, 10/10 |
| crit | Novak (r2) | The judge's proposed replacement rule (most negative CSD at the pyramidal trough) also fails: the EC return source is in phase with the CA3 sink (corr 0.93) | VERIFIED | r2_both.py re-run: trough CSD +0.91 at 350 um, +0.94 at 400 um |
| crit | Zhao, Volkov | Coupling and rhythmic common input are not identifiable from passive data in the linear-Gaussian regime (two constructions: directed pair with latent driver; symmetric pair = sum mode + private modes) | VERIFIED | spectral matrices equal to 6e-16 and 1e-15; GC, coherence, PLV match (ident2.py, equiv.py re-run) |
| crit | Volkov | Phase 2 emergent class information is purely second order: a Whittle (Gaussian spectral) classifier scores 1.000 | VERIFIED | whittle.py re-run, emergent seeds 0-2 (driven 0.73-0.81) |
| major | Volkov | nl_coupled is a damped linear bank with weak linear crosstalk: coupling `z @ W.t()`, Re(lambda) < 0 forced, max Re eig -0.51, h|W| ~ 6e-4, sync features corr 0.985 with W -> 0 | VERIFIED | banks.py:62-63, 150; bank_regime.py re-run (init) |
| major | Halvorsen, Novak | Holm ran over 4 banks when 2 ran (missing rows as p = 1) | VERIFIED; fixed | stats.py:57, run.py:103-111 |
| major | Novak | A resume overwrote git, config and device for earlier arms; changed data undetected | VERIFIED; fixed | q2_meta.py; run.py:139-151 |
| major | Halvorsen | summarize() and the notebook print Holm p at any n (peeking channel) | VERIFIED; fixed | run.py:201-226; notebook cell 15 |
| major | Halvorsen | Learning curves not stored, so "surrogate does not train" cannot be checked | VERIFIED; fixed | run.py:83-86 |
| major | Halvorsen | MDE at n = 14 is dz 0.92 (0.037-0.078 accuracy); TOST at margin 0.05 has 0.81 power at SD 0.06 | VERIFIED | pow.py re-run |
| major | Asante | The bank's power readout is a weak band-power estimator and f0/bw barely move in training | VERIFIED (synthetic) | SD(log P) 0.28-0.45 vs ~0.24 for a 4 Hz band; log f0 moves <= 0.014 in 36 steps |
| major | Asante | Published Shallow 0.749 vs nl_sync 0.530 | VERIFIED | results_hg/ablation_summary.json |
| major | Asante | The 1-40 Hz band meets a 4-125 Hz cache; it comes from TASK_PITCH1 §2, not the published bank (0.2-1.6 Hz) | VERIFIED | build_hg_cache.py:37,47; TASK_PITCH1:40; vendor:229 |
| major | Echeverria | HGD is not a test of Pitch 1 (no internally generated contrast); nl-lin +0.023 is inside Phase 2's generic-capacity band | ACCEPTED | |
| major | Echeverria | Gate 0 is likely to fail on hc-5 (32-channel blocks; about 140 um span) | PENDING | needs the hc-5 QC (crcns.org unreachable here) |
| major | Echeverria, Volkov | Speed drift is independent per generator; with shared (septal) drift it is harmless; the real nuisance is CA3-EC coherence (about -0.07 at full decoherence) | VERIFIED (code), ACCEPTED (numbers) | laminar.py:68-70; drift_truesites.py, r2_drift.py |
| major | Echeverria | qc_hc5 marks a shank usable without the span criterion | VERIFIED | qc_hc5.py:57 |
| major | Echeverria | "Travelling-wave 10 deg" is implemented as an inter-generator phase change | VERIFIED | gate1_laminar.py CELLS |
| minor | Novak | 12x50: l-m lands on the EC source (last site excluded) | VERIFIED | q4_gate1.py |
| minor | Novak | Zero Delta passes every null cell (i_rad == i_lm) | VERIFIED | sign_flip_p(zeros) = 1.0 |
| minor | Zhao | Label smoothing 0.05 makes ln 4 the floor | VERIFIED | train.py:67 |
| minor | Asante | ResoNet as configured is ruled out for phase-locked stimulation; fixed point is not a constraint | ACCEPTED | |

## Verdicts overturned or corrected in round 2
1. **The judge's A3 reconciliation** ("the leak is large at 1 s, small at 4.5 s") is wrong. Novak, Zhao and Asante
   each showed it: a linear head uses the best pairs (smallest lags) and BatchNorm undoes the attenuation, so the
   fixed-roll surrogate decodes 0.90-1.00 at 4.5 s on stationary narrowband phase classes. The leak depends on
   lag relative to how long the class structure stays coherent, and on SNR.
2. **Novak N7** ("val loss near ln 4 is what an informationless null looks like") was withdrawn by Novak and
   rejected by Halvorsen: the frozen-feature probe decodes what the trained readout does not, so the HGD
   surrogate's near-chance score is not explained. Diagnostic D2 (DEVIATION 23) decides it.
3. **Zhao's inference that the leak makes the contrasts conservative** was rejected by Halvorsen: with leak a =
   0.08 the dilution bias is unchanged (+0.143, 25/25), and for an equivalence claim the leak is anti-conservative.
4. **Echeverria E4 and Volkov V6** ("the amplitude cell is removed exactly"): rejected as stated, because the
   ripple is not scaled (phi differs by up to 0.48). Substance kept: the cell is a null replicate (true sites
   +0.0125) and is replaced by an absolute-noise cell.
5. **Echeverria E5** (perisomatic, absolute-noise and DG traps fail): withdrawn by Echeverria. At the true sites
   they pass (+0.011, +0.011, -0.035); the round-1 failures came from the broken site picker. Only the radiatum
   depth shift still moves Delta.
6. **Volkov V7** (layer-specific asymmetry moves Delta by about 0.15): it is a normalisation artefact. Each source
   is normalised to unit variance, so the 16 Hz harmonic takes power from the fundamental; with the fundamental
   fixed, Delta is +0.012 (Echeverria).
7. **Novak N1** (speed_drift fails at the true sites): the failure is real but is not caused by frequency drift. It
   comes from independent drift decohering CA3 and EC (Volkov, Echeverria).
8. **Asante A6** was relabelled: the 1-40 Hz band is the spec's, not the published configuration's.
9. **Upgraded:** H6, H7, A4, N2, N4, N8, V3 to VERIFIED.

## Fixed in this session (commit b0f24b4)
- DEVIATION 23 (Phase 1 analysis rules), frozen before any capacity-matched arm and before subjects 7-14.
- Code: families declared over lin and nl_coupled; exact sign-flip primary test, t CIs, TOST, outcomes; no
  inference below n = 14; provenance per arm; resume refuses config/data changes; atomic writes; learning curves
  and trained weights stored. A golden CPU smoke run of all 22 arms is identical before and after.
- **Declined (proportion):** installing `requirements.lock` on Colab (it pins this container's torch/CUDA 13 wheels
  and would swap the environment mid-study; provenance records versions instead); pinning the notebook to a commit
  (per-arm git hashes plus the golden-equality test cover it); hiding `wins` below n = 14 (descriptive, already
  disclosed in the interim).

## Pending
- DEVIATION 24 calibration screen (frozen before it runs).
- Gate 1 v2 and the amendment of the generator-mix draft.
- The non-identifiability note, the Whittle oracle on all Phase 2 seeds, and Volkov's supercritical check.
- Standard baselines and the frozen-bank analyses (F5, D2), which need the HGD data (Colab).
- Gate 0 on real hc-5 data (needs the QC run).

## Positions
- **Halvorsen:** P2 before P3 (done: DEVIATION 23); add a dilution null to the calibration.
- **Echeverria:** rebuild Gate 1 with shared septal drift, disc sources and per-session jitter before anything
  else; freeze only the rule for the Gate-0-fail branch now.
- **Zhao:** the calibration must include a coupling-only generator and source-power baselines; no "phase" wording
  until it passes on source-power-only data.
- **Volkov:** record the equivalence theorem; add a cheap supercritical oracle check so the note does not read as
  "coupling is never identifiable".
- **Asante:** only (band power + sync) - (band power + fresh-roll surrogate) may use the word "phase"; save the
  trained banks (done).
- **Novak:** Gate 1 needs a site/generator method validated against ground truth across lags and probe offsets.

## Judge's plan
1. **DEVIATION 24, a calibration screen of the Phase 1 contrasts** (frozen, then run on CPU): generators null,
   source-power-only, coupling-only (class-invariant source spectra) and both; arms through the real training
   loop; the frozen-bank F5 candidates and the D2 probe. It decides what the Phase 1 result may be called.
2. **Gate 1 v2**: a site/generator method validated against ground truth; forward-model fixes (distributed
   ripple, disc sources, absolute noise, per-site gain, probe-offset and thickness jitter, shared drift); the cell
   list from Echeverria's round 2; equivalence relative to the target effect; a geometry pass map that replaces
   the guessed 300 um in Gate 0.
3. **Non-identifiability note** (both constructions, assumptions, N-unit form, what breaks it), the Whittle
   oracle on all 10 Phase 2 seeds and a supercritical oracle check.
4. **Baselines module** (FBLV + shrinkage LDA, FBCSP, tangent space) and the F5/D2 frozen-bank analyses, ready for
   Colab.
5. Tom: resume the Colab run (the code now stores weights and curves); run the hc-5 QC.

## Round 3 (2026-11-09): review of what the judge built
**Mode:** agent mode; the six members were resumed with their context, one round. **Target:** commits b0f24b4..9dd30fd.

| sev | member | finding | verdict | evidence |
|---|---|---|---|---|
| crit | Halvorsen | Gate 1 v2 null cell biased (+0.049 pooled, n = 20); effective alpha 0.25 at n = 7 | REJECTED | 40 fresh null sessions: -0.012 [-0.031, +0.007]; n = 7 subsets reject 0.034; A/B swap exactly antisymmetric |
| crit | Echeverria | "Inverse crime": with depth-extended generators the target falls to +0.208, identification 0.7; sink spread moves Delta | VERIFIED | r3_extended.py re-run |
| crit | Volkov | CA3-EC coherence 0.95 -> 0.3 moves Delta by -0.236 | VERIFIED | judge re-run on validation seeds: -0.236 [-0.315, -0.157], 1/10 |
| major | Halvorsen, Echeverria, Volkov, Novak | Making the radiatum shift non-gating after it failed on dev is a forking path | ACCEPTED; reverted | DEVIATION 26: gating again -> Gate 1 v2 FAIL |
| major | Novak | `_pair_score` closed form wrong (conjugation) | VERIFIED; fixed | up to 64% error vs explicit QR; regression test |
| major | Novak | Null criterion on raw Delta with common random numbers; paired effects show real nuisances (coherence 0.75: -0.060) | ACCEPTED | requirement for v3 |
| major | Volkov | Asymmetry cell (fundamental fixed) and five others reproduce the null draw: test nothing | ACCEPTED | requirement for v3 |
| major | Zhao | G2 carries class information in zero-lag band power (FBLV 0.56-0.60) | VERIFIED | results/calib_p1: FBLV 0.60, FBCSP 0.58 on G2 |
| major | Zhao, Volkov | The note overclaims: screen 11's units were detuned with complex loadings, so construction 1 does not cover its R cells; "what breaks it" is not an identification result | VERIFIED (synth.py:56); fixed | NONIDENTIFIABILITY.md corrected |
| major | Halvorsen, Asante | The DEVIATION 24 validity check had about 50% false-invalid probability (not 14%) | ACCEPTED (simulations not re-run); corrected | DEVIATION 24 addendum |
| major | Halvorsen | Holm family shrinks when a declared row is incomplete; intersection-union rule and D3 not in code | VERIFIED; fixed | stats.phase1_rows (m = declared rows); run.decision_rules; test |
| major | Novak | Notebook not pinned to a commit; resume does not refuse code changes | DECLINED (proportion) | a code-hash refusal would block the legitimate resume (training code unchanged, golden run identical); per-arm git recorded |
| minor | Novak | p_wilcoxon_exact not exact under ties; complete flag true with missing arms; code_dirty ignores scripts; post-hoc not traceable; D2 wording | VERIFIED; fixed | stats.py, run.py, phase1_posthoc.py (weights sha256), DEVIATION 24 addendum |
| minor | Volkov | bank_regime units and criterion (RK4 gain, max coupling/detuning) | ACCEPTED; fixed | run.bank_regime |
| minor | Asante | 40-60 Hz band contains 50 Hz mains | ACCEPTED; fixed | HG_BANDS skip 48-52 Hz |
| minor | Asante | Baseline memory about 5-6 GB on 128 channels | ACCEPTED, not changed | fits Colab |
| - | Asante | Baselines match the P6 spec; calibration honestly reported apart from the 14% figure | ACCEPTED | |

**Overturned or corrected judge decisions:** the Gate 1 v2 PASS (now FAIL); the radiatum cell's non-gating status;
the "14%" figure; three sentences of the non-identifiability note.

**Positions.**
- **Halvorsen:** never cite the v2 pass; future calibrations need about 25 datasets per generator.
- **Echeverria:** templates must come from the data (gamma-band CSD profiles); expect hc-5 to fail Gate 0.
- **Zhao:** any future "phase beyond band power" claim needs a +/-90 deg (imaginary-only) sensitivity generator.
- **Volkov:** a supercritical oracle check only if Pitch 1's decoder question stays open.
- **Asante:** resume Phase 1 as frozen; nothing in his lane blocks it.
- **Novak:** pin the code before any further run.

**Judge's plan.**
1. Tom resumes Phase 1 (subjects 7-14, then the capacity-matched arms), then runs notebook §6b (post hoc) and §7
   (hc-5 QC).
2. hc-5 QC -> Gate 0. If Gate 0 fails as expected, the hc-5 CSD route is reported infeasible, and a laminar dataset
   is chosen by documented geometry before any of its data are opened.
3. Gate 1 v3 only for a dataset that passes Gate 0.
4. Phase 1 analysis at n = 14 exactly as frozen (DEVIATIONS 23-24).
