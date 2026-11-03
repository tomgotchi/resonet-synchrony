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
