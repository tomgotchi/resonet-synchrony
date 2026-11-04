# resonet-synchrony: results

## Phase 0: setup (2026-10-27)

- Repo scaffolded following TASK.md S0 conventions: `vendor/` holds the unmodified published code with hash checks,
  data is gitignored, paths go in `config.yaml`, and `config.example.yaml` is committed.
- Reused from phasetrack: the sanitization helpers and their tests, and the causality-test pattern (now applied to
  the banks) plus paired statistics with Holm correction.
- **66 tests pass**, covering:
  - Hz parameterization: init f0 is log-spaced and the measured free rotation matches f0 within 2%.
  - Causality of all four banks.
  - Equivalence with the published forward pass: with equivalent parameters, `nl_coupled` reproduces vendored
    `ResonanceNetwork` and `lin` reproduces `FastResonanceNetwork`, and the sync/surrogate features match. This is
    the S2 requirement to keep everything else identical.
  - S7 smoke tests for all 20 bank x readout arms: gradients reach every parameter group and the loss decreases.
  - Controls: the surrogate changes features but preserves each oscillator's spectrum, `trial_shuffle` reduces to
    `sync` when the partner is the trial itself, and `nl_uncoupled_wide` matches `nl_coupled`'s extra parameter
    count within 2N.
  - An end-to-end smoke run.

### Findings worth knowing before Phase 1
1. **HGD cache question answered:** `subj_cache/` is low-passed (~48 Hz), and `subj_cache_hg/` is the full-band 4-125 Hz
   rebuild (`build_hg_cache.py`). Phase 1 uses `subj_cache_hg` at decim 2 (250 Hz), which is the published
   high-gamma configuration (`results_hg/surrogate_summary.json`: decim 2, N 16, cap 80, 80 epochs).
   `data_hgd.band_check` verifies the band before any run.
2. **The published integrator can't carry Hz-correct high frequencies.** It takes one RK4 step per sample, which
   adds frequency-dependent damping: bandwidth is 1.02x at 20 Hz, 1.24x at 30 Hz and 2.3x at 40 Hz at fs = 250.
   Above about 60 Hz the oscillator decays to zero. The linear bank uses the exact exponential and is unaffected,
   so with correct frequencies `nl` vs `lin` also differ in effective bandwidth for the top oscillators (31 and
   40 Hz of the 16). `--rk4-substeps 2` brings 1-40 Hz within 1.04x. For hippocampus (2-150 Hz) the spec as written
   is not feasible at 250 Hz: 150 Hz exceeds Nyquist, and RK4 needs k >= 4. See DEVIATIONS #2-3.
3. **`trial_shuffle` needs the label at test time** to choose a same-class partner. It is therefore an analysis
   control, an upper bound on evoked phase information, not a decoder (DEVIATIONS #4; needs Tom's sign-off).
4. Published numbers for reference (read from `results_hg/surrogate_results.json`, full band, n = 14):
   sync - surrogate = **+0.163 (nl, 14/14)** and **+0.150 (lin, 12/14)**. TASK_PITCH1 quotes +0.098 / +0.147, which
   appear to be the low-passed-cache values. The Phase 1 comparison uses the full-band ones.

## Phase 1: recalibrated replication on HGD. Runs on Tom's GPU (decided 2026-10-27).

The code is complete and every arm is smoke-tested on tiny synthetic data. Tom runs it locally, where the
full-band cache exists, following `RUN_PHASE1_LOCAL.md`. The cache is not rebuilt here.
`--arms core` (the 6 published arms) answers the urgent question first: does sync > surrogate survive Hz
initialisation? Per-subject JSONs store the config, seed, git hash, data sha256, band check and chance threshold.
The evoked/induced decomposition (DEVIATIONS 4) is computed for every `bank:sync` model and summarised
separately from the decoder table.

## Phase 2 plan: synthetic validation of the nl-vs-lin contrast (H0). Frozen 2026-10-27, before any evaluation seed was run.

- **Data:** `synth.py`, with two generators (DRIVEN: uncoupled units, class = phase pattern of a common
  input; EMERGENT: noise-driven Stuart-Landau network, class = coupling graph, class-free input).
  8 channels, 1 s @ 250 Hz, z-scored per channel and trial. 100 train and 40 test trials per class, with the
  published 15% stratified validation carve from train used for early stopping.
  **Evaluation seeds 0-9**, with the seed as the unit; seed 999 was used only for calibration (DEVIATIONS 9).
- **Arms:** lin, nl_uncoupled, nl_coupled and nl_uncoupled_wide with the `sync` readout, plus `power` and `surrogate`
  for lin and nl_coupled. The published training loop is used unchanged (<= 80 epochs, patience 15), with N = 16
  and band 2-40 Hz.
- **Primary metric:** test balanced accuracy, compared with the Combrisson-Jerbi chance threshold for n_test = 160.
- **H0, emergent:** the contrast *detects* coupling-dependent information if nl_coupled:sync - lin:sync > 0
  with Holm p < 0.05 within the 4-contrast coupling family.
- **H0, driven:** the contrast correctly *does not detect* if the 95% bootstrap CI of nl_coupled - lin lies
  inside +/-0.03 (equivalence). If the CI crosses the margin without significance, the result is inconclusive.
- **Secondary:** the interaction (d_emergent - d_driven) > 0, the capacity control nl_coupled - nl_uncoupled_wide
  on emergent, H2 sync - surrogate per bank, and the evoked-vs-induced decomposition (reported separately).
- **Stop rule (TASK_PITCH1 S5):** if the emergent criterion fails, H3 is underpowered and any real-data null is
  uninterpretable. In that case the result is reported and Phase 3 does not start.

## Phase 2 results: synthetic H0 validation (10 seeds per generator; `results/phase2/summary.json`)

Test balanced accuracy (chance threshold for n = 160: 0.312):

| arm | emergent | driven |
|---|---|---|
| lin:sync | 0.860 | 0.759 |
| nl_uncoupled:sync | 0.868 | 0.766 |
| nl_coupled:sync | 0.879 | 0.771 |
| nl_uncoupled_wide:sync | **0.890** | 0.768 |
| lin / nl_coupled : power | 0.579 / 0.630 | 0.520 / 0.586 |
| lin / nl_coupled : surrogate | 0.621 / 0.647 | 0.601 / 0.629 |

### Against the frozen criteria
| criterion | result | verdict |
|---|---|---|
| **H0 emergent:** nl_coupled - lin > 0, Holm p < 0.05 | +0.019 [+0.007, +0.031], dz 0.94, raw p 0.043, **Holm p 0.17**, 8/10 seeds | **FAILS.** The contrast does not detect emergent coupling at the pre-registered threshold. |
| **H0 driven:** CI of nl_coupled - lin inside +/-0.03 | +0.012 [+0.003, +0.021] | **Passes** (correct null). |
| Interaction (d_emergent - d_driven) | +0.007 [-0.006, +0.019], p 0.37, 7/10 | Not significant. |
| Capacity control on emergent: nl_coupled - nl_uncoupled_wide | -0.011 [-0.025, +0.003], 3/10 | The wide encoder is at least as good as trainable coupling. |
| H2: sync - surrogate | +0.23 to +0.24 (emergent), +0.14 to +0.16 (driven), 10/10, Holm p 0.004 | Replicates in both. |

**Stop rule (TASK_PITCH1 S5) triggered:** `nl_coupled` does not beat `lin` on the emergent generator at the
pre-registered threshold. A real-data null in Phase 3 would therefore be uninterpretable. **Phase 3 does not start.**

### Why it fails (interpretation; for discussion with Tom before any redesign)
- **The phase information the coupling creates is in the data itself.** Class-specific coupling in the generator
  produces class-specific pairwise phase locking in the observed channels. A *linear, uncoupled* bank with the
  `sync` readout measures that coherence directly, so it reaches 0.86 vs 0.88 for the coupled bank. The decoder
  does not need coupled dynamics to read out coherence its input already carries.
  - The nl-vs-lin decoder contrast therefore asks "does the decoder benefit from coupled dynamics?" That is a
    different question from "is the class information emergent rather than driven?", and on this evidence the
    answer to the first is "barely, and not more for emergent data".
- **The small nl advantage (+1-2 points) appears for both generators** and does not exceed what the
  parameter-matched wide encoder gets. It looks like generic capacity, not coupling-specific.
- **The evoked/induced decomposition** shows both generators are almost entirely induced (evoked fraction about
  0.17; trial-shuffled CV at chance, about 0.25), as designed. The decomposition cannot separate them either.
- **Caveats:** a single generator family, one difficulty level (calibrated on seed 999), N = 16, 1 s windows.
  More seeds might make +0.019 significant, but the effect size relative to the driven case (the interaction) is
  what H3 needs, and that is +0.007 and non-significant.

### Options for redesign (not started; needs Tom)
1. Make the H3 contrast *generator-identifiable*: test a quantity that coupled dynamics produce and linear
   filtering of the observed channels cannot. Candidates are transient sequence structure, higher-order (beyond
   pairwise) phase interactions, and responses to perturbation. Re-validate H0 on it.
2. Keep the decoder contrast but drop the "emergent" interpretation. Report it as "does a coupled decoder help",
   as a benchmark, without the dissociation claim.
3. Increase sensitivity, with more seeds and pre-registered power analysis, *only* if the target effect is the
   interaction. The current interaction estimate (+0.007) suggests the required n would be large.

## Exploratory mechanism-statistic screen (DEVIATIONS 11; synthetic only; `results/screen1/screen_results.json`)

The screen used three generators: EMERGENT; DRIVEN (narrowband common input); and DRIVEN_COMMON (EMERGENT's units with
coupling replaced by an unobserved shared white-noise source, rho = 0.85, calibrated on seed 999 so that plain PLV
matches EMERGENT). Seeds 0-9, 400 train trials per dataset, offline 6-10 Hz phases. The criterion was frozen before
the run: >= 9/10 seeds and a 95% CI excluding 0 for both E-D and E-DC, with the same sign.

| statistic | E / D / DC mean | E - D [95% CI] (seeds) | E - DC [95% CI] (seeds) | passes |
|---|---|---|---|---|
| S_couple (Kuramoto-fit coupling, rad/s) | 0.522 / 0.282 / 0.786 | +0.240 [+0.211, +0.272] (10/10) | **-0.264** [-0.344, -0.200] (10/10) | no (opposite signs) |
| **S_TE (mean directed phase TE, bits)** | **0.0240** / 0.0069 / 0.0131 | +0.0172 [+0.0165, +0.0178] (10/10) | +0.0109 [+0.0101, +0.0116] (10/10) | **YES** |
| S_asym (TE asymmetry) | 0.068 / 0.056 / 0.089 | +0.012 [+0.006, +0.019] (9/10) | -0.022 [-0.028, -0.015] (10/10) | no (opposite signs) |
| S_resid (PLV after removing first complex PC) | 0.577 / 0.582 / 0.560 | -0.006 [-0.011, -0.001] (6/10) | +0.016 [+0.014, +0.019] (10/10) | no |
| S_plv (reference, not a candidate) | 0.673 / 0.527 / 0.685 | +0.145 (10/10) | -0.013 (8/10) | no, as designed |

**Reading:**
- **Only directed phase transfer entropy passes.** Emergent data carry more phase-predictive information between
  units than either driven generator. This includes the hardest comparison, DRIVEN_COMMON, which has the same units,
  the same total noise and matched PLV.
- **The coupling-function fit is fooled by unobserved common noise**: it reports *more* coupling for DRIVEN_COMMON
  than for EMERGENT. This is the textbook common-input confound of coupling inference, and it disqualifies this
  statistic.
- **Removing a single common mode does not help.** With 8 units at slightly different frequencies, the common input
  is not rank-1 after each unit's own filtering, and EMERGENT's coherence is not cleanly "beyond rank-1" either.

**Caveats; the result is not yet usable for H3:**
- One parameter setting, seed-level n = 10, plug-in TE (biased; the bias depends on the effective sample count, which
  differs little here because all three use the same windows and band). TE differences could partly reflect
  differences in phase predictability (spectral shape) rather than inter-unit influence. EMERGENT vs DRIVEN_COMMON is
  the cleanest comparison because the units are identical.
- Per the frozen criterion, a passing statistic must be **re-validated on fresh seeds and at other difficulty and N
  settings** (coupling K, rho, number of units, window length, SNR) before being used. Its class-specific use, i.e.
  whether TE carries *class* information beyond coherence, has not been tested.
- Synthetic only. No claim about real data.

## TE re-validation (DEVIATIONS 12; fresh seeds 10-19; `results/screen2/screen2_results.json`)

| condition | rho (DC) | S_TE E / D / DC | E - D (seeds) | E - DC [95% CI] (seeds) | pass |
|---|---|---|---|---|---|
| base | 0.85 | 0.0240 / 0.0069 / 0.0134 | +0.0171 (10/10) | +0.0106 [+0.0099, +0.0113] (10/10) | yes |
| K25 | 0.70* | 0.0164 / 0.0069 / 0.0088 | +0.0095 (10/10) | +0.0076 (10/10) | yes |
| K60 | 0.95* | 0.0327 / 0.0069 / 0.0244 | +0.0258 (10/10) | +0.0083 (10/10) | yes |
| rho70 | 0.70 | 0.0240 / 0.0069 / 0.0088 | +0.0171 (10/10) | +0.0152 (10/10) | yes |
| **rho95** | **0.95** | 0.0240 / 0.0069 / **0.0244** | +0.0171 (10/10) | **-0.0004 [-0.0016, +0.0007] (4/10)** | **no** |
| M4 | 0.95* | 0.0391 / 0.0069 / 0.0228 | +0.0322 (10/10) | +0.0163 (10/10) | yes |
| M16 | 0.70* | 0.0186 / 0.0069 / 0.0087 | +0.0117 (10/10) | +0.0099 (10/10) | yes |
| win05 | 0.85* | 0.0273 / 0.0115 / 0.0181 | +0.0158 (10/10) | +0.0092 (10/10) | yes |
| win2 | 0.85* | 0.0211 / 0.0034 / 0.0100 | +0.0177 (10/10) | +0.0111 (10/10) | yes |
| snr10 | 0.85* | 0.0239 / 0.0069 / 0.0134 | +0.0170 (10/10) | +0.0105 (10/10) | yes |
| snr0 | 0.85* | 0.0221 / 0.0068 / 0.0127 | +0.0153 (10/10) | +0.0094 (10/10) | yes |

\* rho recalibrated on seed 999 by the frozen PLV-matching rule.

**Verdict (frozen rule):** the base condition passes on fresh seeds, and 9 of 10 variants pass, so TE is **robust**.
The only failure is **rho95**.

**What the failure means; it matters more than the count:**
- At rho = 0.95, unobserved common noise produces **as much TE as K = 40 coupling** (0.0244 vs 0.0240).
- TE therefore does *not* separate coupling from common input in principle. It separates them only when the
  coupling-generated TE exceeds what the common input generates. The variants that kept passing at rho = 0.95 (K60,
  M4) did so because their emergent TE was larger, not because common noise stopped producing TE.
- The cross-trial control locates the effect.
  - **Driven (narrowband shared input):** TE is almost entirely estimator bias / own-phase predictability (excess
    about 0.0004).
  - **Common white noise:** it creates *genuine* lagged inter-unit predictability (excess 0.002-0.018, growing with
    rho). Units with different natural frequencies respond to the same noise kicks at different phases.
- So TE measures **inter-unit phase predictability**, not coupling as such.

**Consequence for an H3 redesign (interpretation, not a decision):**
- TE could only serve as an *emergent-coupling* index on real data if the strength of unobserved common input can be
  bounded independently. One example is showing that the TE pattern is not explained by a common-input model fitted
  to the same data.
- Without such a bound, a wheel > maze difference in TE would still be ambiguous between "more coupling" and "more
  shared input".
- This is a design question for Tom before any hc-5 work.

## Screen 3: lagged (conditional Granger) vs instantaneous dependence (DEVIATIONS 13; fresh seeds 20-29; `results/screen3/`)

| condition | statistic | E / D / DC | E - D (seeds) | E - DC [95% CI] (seeds) | pass |
|---|---|---|---|---|---|
| base | **S_GC** | 0.0027 / 0.0005 / **0.0048** | +0.0022 (10/10) | **-0.0021** [-0.0024, -0.0018] (10/10) | no (opposite signs) |
| rho70 | S_GC | 0.0027 / 0.0005 / 0.0023 | +0.0022 (10/10) | +0.0004 [+0.0003, +0.0005] (10/10) | yes |
| rho95 | S_GC | 0.0027 / 0.0005 / **0.0081** | +0.0022 (10/10) | **-0.0054** (10/10) | no |
| base / rho70 / rho95 | S_inst | E 0.027, D 0.004, DC 0.196 / 0.132 / 0.247 | +0.023 (10/10) | -0.17 / -0.10 / -0.22 (10/10) | no (opposite signs) |
| base / rho70 / rho95 | S_TE (reference) | as in DEVIATIONS 12 | +0.017 (10/10) | +0.011 / +0.015 / **-0.0007 (6/10)** | yes / yes / no |

**Verdict (frozen rule): S_GC fails.** It does not pass all three conditions; it passes only rho70.

**Reading:**
- **The caveat stated before the run materialised.** With an unobserved common input, only the real part observed,
  and zero-phase decimation, the shared noise produces *more* lagged Granger influence than the true coupling
  (0.0048-0.0081 vs 0.0027). This is the known failure of GC under latent confounding and partial observation.
- The TE results replicate on a third independent seed set: it passes base and rho70 and fails rho95, exactly as in
  DEVIATIONS 12.
- **Post-hoc observation (NOT a result; it would need its own frozen test):** instantaneous residual correlation is
  about 10x higher for common noise (0.13-0.25) than for coupling (0.027), even at rho95.
  - A *two-statistic* rule, "high TE **and** low instantaneous correlation", separates emergent from both driven
    generators in every condition in these numbers.
  - That rule was chosen after seeing the data, so it is a hypothesis for a new pre-registered screen on fresh
    seeds 30-39 and further variants. It is not a finding.

**Where this leaves the H3 redesign (for Tom):** none of the pre-registered single statistics separates coupling
from strong unobserved common input. The most promising lead is the post-hoc conjunction (lagged predictability
together with low instantaneous dependence). It needs a frozen confirmatory screen before it can be considered.

## Screen 4: confirmatory test of the TE-and-low-instantaneous rule (DEVIATIONS 14; fresh seeds 30-39; `results/screen4/`)

Rule: a dataset is **emergent** iff S_TE > 0.015295 **and** S_inst < 0.111262. Both thresholds were frozen from
seeds 20-29.

| condition | emergent called emergent | DRIVEN called not | DRIVEN_COMMON called not | success | mean S_inst E / D / DC |
|---|---|---|---|---|---|
| base | 10/10 | 10/10 | 10/10 | yes | 0.027 / 0.005 / 0.184 |
| rho70 | 10/10 | 10/10 | 10/10 | yes | 0.027 / 0.005 / 0.124 |
| **rho95** | 10/10 | 10/10 | **10/10** | **yes** | 0.027 / 0.005 / 0.231 |
| K25 | 9/10 | 10/10 | 10/10 | yes | 0.019 / 0.005 / 0.124 |
| K60 | 10/10 | 10/10 | 10/10 | yes | 0.038 / 0.005 / 0.231 |
| M16 | 10/10 | 10/10 | 10/10 | yes | 0.014 / 0.004 / 0.133 |
| win05 | 10/10 | 10/10 | 10/10 | yes | 0.028 / 0.005 / 0.182 |
| snr0 | 10/10 | 10/10 | 10/10 | yes | 0.024 / 0.004 / **0.027** |

**Verdict (frozen rule): CONFIRMED.** It succeeds in all 3 primary conditions, including rho95, the case TE alone
failed, and in 5/5 variants.

**Caveats; these limit how far this goes:**
1. **At 0 dB observation noise, the instantaneous leg collapses.** DRIVEN_COMMON's S_inst falls from 0.18 to 0.027,
   the same as emergent, because independent sensor noise dilutes the shared innovation. In snr0 the common-input
   datasets were rejected by **TE alone**. A combination this screen did not test (rho95 together with low SNR) would
   very likely fail, since TE alone cannot reject rho95.
2. **The margins are thin in places.** K25 lost one emergent dataset (S_TE 0.0161 vs threshold 0.0153).
3. **The absolute thresholds will not transfer to real data.** Units depend on the number of channels, window,
   bandwidth and SNR. Real-data use needs thresholds calibrated against a fitted null model of the same recording
   (for example, a common-input-only model fitted to it), not these numbers.
4. Synthetic, single generator family, Stuart-Landau units, linear diffusive coupling. Confirmed **only** as a
   mechanism-level discriminator in this setting.

**Status for the H3 redesign:** this is the first candidate that passed a pre-registered confirmatory test,
including the common-input confound. Before any hc-5 use, the remaining open items are:
- caveat 1: a low-SNR x strong-common-input test;
- caveat 3: a recording-specific null calibration;
- the 102-channel region-map question.

## Boundary test 5: strong common input x observation SNR (DEVIATIONS 15; fresh seeds 40-49; `results/screen5/`)

rho = 0.95 throughout. Rule and thresholds unchanged from DEVIATIONS 14.

| obs. SNR | emergent called emergent | DRIVEN called not | DRIVEN_COMMON called not | success | S_inst E / DC |
|---|---|---|---|---|---|
| 20 dB | 10/10 | 10/10 | 10/10 | yes | 0.028 / 0.206 |
| 10 dB | 10/10 | 10/10 | **2/10** | **no** | 0.029 / 0.106 |
| 5 dB | 10/10 | 10/10 | 0/10 | no | 0.028 / 0.068 |
| 0 dB | 10/10 | 10/10 | 0/10 | no | 0.025 / 0.042 |

**Verdict: the operating range under strong common input is observation SNR >= 20 dB only.**
- Below that, independent sensor noise dilutes the instantaneous residual correlation that marks a shared input.
  DRIVEN_COMMON then looks emergent, because its TE stays as high as coupling's.
- The failure is one-sided and dangerous: **common input gets called emergent coupling, not the other way round.**
  On real data that is the error that would produce a false positive for H3.

**Implications (for Tom; no redesign is decided here):**
- The confirmed rule (screen 4) holds across coupling strength, network size, window length and SNR *at moderate
  common input*. Against *strong* common input it needs high observation SNR.
- Using it on hc-5 would need a train-split estimate of the theta-band observation SNR and of the plausible
  common-input strength. Neither is directly measurable from the recording alone.
- **A possible fix to test next, not done here:** an SNR-invariant version of the instantaneous statistic. For
  example, normalise the residual correlation by each channel's estimated noise-free residual variance, or use partial
  coherence at lag 0 after a noise-floor correction. Any such fix would need its own frozen screen.

## Screen 6: SNR-robust rule with LDS-denoised instantaneous dependence (DEVIATIONS 16; fresh seeds 50-59; `results/screen6/`)

Rule: a dataset is **emergent** iff S_TE > 0.015295 **and** S_inst_dn < 0.115964. Both thresholds were frozen
before these seeds existed.

| condition | E called emergent | D called not | DC called not | success | S_inst_dn E / D / DC |
|---|---|---|---|---|---|
| **rho95, 20 dB** | 10/10 | 10/10 | 10/10 | yes | 0.032 / 0.008 / 0.219 |
| **rho95, 10 dB** | 10/10 | 10/10 | 10/10 | yes | 0.030 / 0.015 / 0.225 |
| **rho95, 5 dB** | 10/10 | 10/10 | 10/10 | yes | 0.033 / 0.020 / 0.224 |
| **rho95, 0 dB** | 10/10 | 10/10 | 10/10 | yes | 0.047 / 0.025 / 0.192 |
| base | 10/10 | 10/10 | 10/10 | yes | 0.035 / 0.009 / 0.195 |
| rho70 | 10/10 | 10/10 | 10/10 | yes | 0.035 / 0.009 / 0.132 |
| K25 (weak coupling) | **6/10** | 10/10 | 10/10 | **no** | 0.024 / 0.009 / 0.132 |
| M16 | 10/10 | 10/10 | 10/10 | yes | 0.020 / 0.008 / 0.130 |
| win05 | 10/10 | 10/10 | 10/10 | yes | 0.026 / 0.010 / 0.195 |

**Verdict (frozen rule): CONFIRMED.** It succeeds in all 4 primary conditions (strong common input from 20 dB down to
0 dB) and in 4/5 generalisation conditions.

**Reading:**
- **The screen-5 failure is fixed.** Denoising through a fitted latent LDS keeps the common-input signature at every
  SNR tested (S_inst_dn 0.19-0.23 vs 0.03-0.05 for emergent). The raw statistic had fallen to 0.04 at 0 dB.
- **Across all 270 datasets (90 per generator), no driven or common-input dataset was ever called emergent.**
  Within this synthetic family the rule makes no false positives, which is the error that would matter for H3.
- **Its only failure is conservative.** With weak coupling (K25) it misses 4/10 emergent datasets, because their TE
  falls just under the fixed TE threshold (0.0159 vs 0.0153). It under-claims coupling and never over-claims it.
- **One trend to watch.** At 0 dB, emergent S_inst_dn creeps up (0.047), still well below the threshold (0.116).

**Limits (unchanged in kind):**
- Synthetic Stuart-Landau networks only, and the LDS is a linear approximation of them.
- The thresholds are absolute and generator-specific. Real-data use needs recording-specific null calibration.
- This is a mechanism-level discriminator. It is not yet evidence about hippocampus, and it says nothing about class
  information (H3 asks about the *coupling advantage* for decoding).

**Status for the H3 redesign (Tom's decision):** this is the first rule shown, with a pre-registered confirmation,
to separate emergent coupling from common input across the SNR range. A redesigned H3 could use it as a
*regime-classification* step (wheel vs maze: is the inter-channel dependence coupling-like?) before, or instead of,
the nl-vs-lin decoder contrast that Phase 2 showed to be uninformative.

## Pilot (seed 999 only, not a screen): recording-specific null via a no-coupling LDS. NEGATIVE

- *Idea:* replace the absolute thresholds by a null fitted to the recording itself. Fit an LDS with block-diagonal A
  (no coupling), full Q (common input allowed) and diagonal R; simulate it; ask whether the observed S_TE exceeds the
  null's 95th percentile (`lds.null_te_test`).
- *Result:* **every** generator exceeded its null, DRIVEN and DRIVEN_COMMON included (null S_TE about 0.007-0.014 vs
  observed 0.007-0.025), so the test is anti-conservative.
- *Diagnosis:* even the *unconstrained* LDS, simulated back, reproduces only about 2/3 of the data's TE
  (0.017 vs 0.025) and PLV of 0.57 vs 0.80. A linear-Gaussian model cannot regenerate the phase structure of these
  weakly nonlinear (Stuart-Landau) networks, and DRIVEN's coloured, oscillatory shared input is outside its null
  family entirely.
- *Consequence:* linear-LDS *simulation* nulls are not usable. The LDS remains useful as a *denoiser* of
  instantaneous structure (screen 6), where misspecification did not matter empirically.
- Recording-specific calibration still needs either a nonlinear generative null (for example, fitted
  noise-driven Stuart-Landau units with common input and no coupling) or an empirical calibration within the
  recording (for example, contrasting epochs or channel pairs whose common-input exposure is known). Both are open
  design choices for Tom.
- The code is kept in `lds.py` (`em_fit(coupled=False)`, `null_te_test`) and marked as not validated.

## Screen 7: within-session contrast rule, no absolute thresholds (DEVIATIONS 17; sessions 60-69; `results/screen7/`)

Rule: A is "more coupling-dominated than B" iff Delta S_TE > 0 (>= 9/10 sessions, CI > 0) **and** Delta S_inst_dn
shows no increase (upper CI < +0.03). A and B are two regimes of the same session: same units, graphs and loadings,
independent trials.

| scenario @ SNR | Delta S_TE [CI] (sessions > 0) | Delta S_inst_dn [CI] | fires | should | correct |
|---|---|---|---|---|---|
| S1 coupling up @20 dB | +0.0077 [+0.0069, +0.0086] (10) | +0.004 [+0.003, +0.005] | yes | yes | yes |
| S1 coupling up @0 dB | +0.0068 [+0.0059, +0.0077] (10) | +0.007 [+0.003, +0.012] | yes | yes | yes |
| **S2 common-input up (trap) @20 dB** | **+0.0066 [+0.0044, +0.0090] (9)** | **+0.147** [+0.132, +0.162] | no | no | yes |
| **S2 trap @0 dB** | **+0.0048 [+0.0028, +0.0068] (9)** | **+0.140** [+0.121, +0.159] | no | no | yes |
| S3 null @20 dB | +0.0001 [-0.0003, +0.0006] (4) | +0.000 | no | no | yes |
| S3 null @0 dB | +0.0001 [-0.0003, +0.0006] (4) | +0.003 | no | no | yes |
| S5 coupling vs common @20 dB | +0.0074 (10) | -0.169 | yes | yes | yes |
| S5 coupling vs common @0 dB | +0.0069 (10) | -0.147 | yes | yes | yes |

**Verdict (frozen rule): all 8 cells correct, with no false positives and no misses.**

**Reading:**
- **The trap shows why the guard is needed.** When only common input increased (S2), TE rose significantly in 9/10
  sessions. TE alone would have reported "more coupling", a false positive. The rise in S_inst_dn (+0.14) blocked it,
  at both 20 dB and 0 dB.
- **The null session is clean.** Delta was about 0 for both statistics, and the rule did not fire.
- **No absolute threshold is involved.** Only the *direction* of within-session differences plus one pre-fixed
  margin on S_inst_dn is used. This is the form H3 needs: wheel vs maze within the same session.

**Limits:**
- Synthetic Stuart-Landau networks with linear diffusive coupling.
- The S_inst_dn margin (+0.03) was set from the pilot's effect size. Real recordings may have different scales, and
  the margin may need re-expressing relative to a within-session null (for example, split-half differences of the
  *same* regime).
- 10 sessions here, against 7 hc-5 sessions available in reality.

**Proposal for the H3 redesign (Tom's decision; nothing is frozen for real data):** replace the "nl_coupled - lin
decoder advantage" contrast, which Phase 2 showed cannot detect emergence, with this two-statistic within-session
regime contrast. Wheel would count as more coupling-dominated than maze iff its TE is higher *and* its LDS-denoised
instantaneous dependence is not, across sessions. The margin would be calibrated within each session from
split-half nulls.

## Screen 8: scale-free relative guard (DEVIATIONS 18; fresh sessions 70-79; 0 dB throughout; `results/screen8/`)

Rule v2: Delta S_TE > 0 (>= 9/10 sessions, CI > 0) **and** the relative change in S_inst_dn
(A - B) / B has an upper CI < **0.25**. No absolute threshold is used anywhere.

| cell | Delta S_TE [CI] (n > 0) | relative Delta S_inst_dn [CI] | fires | should | correct |
|---|---|---|---|---|---|
| S1 coupling up, M16 | +0.0058 [+0.0054, +0.0062] (10) | +0.017 [-0.011, +0.042] | yes | yes | yes |
| S1 coupling up, win05 | +0.0053 [+0.0047, +0.0059] (10) | +0.094 [-0.019, **+0.220**] | yes | yes | yes |
| S2 trap, M16 | **+0.0056 (10)** | +1.89 | no | no | yes |
| S2 trap, win05 | **+0.0039 (10)** | +1.91 | no | no | yes |
| S3 null, M16 | +0.0001 (6) | -0.023 | no | no | yes |
| S3 null, win05 | -0.0003 (4) | +0.037 | no | no | yes |
| S5 coupling vs common, M16 | +0.0019 (10) | -0.84 | yes | yes | yes |
| S5 coupling vs common, win05 | +0.0066 (10) | -0.74 | yes | yes | yes |
| **S2b mild trap (rho 0.5 -> 0.7)** | **+0.0012 [+0.0005, +0.0019] (9)** | **+0.77** [+0.65, +0.90] | no | no | yes |

**Verdict (frozen rule): all 9 cells correct, with no false positives and no misses.** The rule held at 0 dB in
16-unit networks and 0.5 s windows, where absolute statistic scales differ.

**Reading:**
- **Both traps would have fooled TE alone.** Even the *mild* common-input increase (rho 0.5 -> 0.7) raised TE
  significantly (9/10 sessions, CI > 0). The relative guard blocked it (+77% in S_inst_dn). Pure coupling increases
  move S_inst_dn by only about 2-9%.
- **One near-miss to note.** For S1 in short windows (win05), the guard's upper CI reached +0.22, against the
  0.25 margin. With noisier statistics (0.5 s windows, 0 dB) the guard comes close to blocking a true coupling
  increase. Its errors are therefore conservative (missing coupling, not inventing it), but its sensitivity drops in
  short or noisy data.

**Cumulative status of the within-session rule (screens 7 + 8):**
- 17/17 cells correct across 20 sessions.
- Every trap and every null was correctly rejected, at 20 dB and at 0 dB, across 8-unit and 16-unit networks and
  1 s and 0.5 s windows.
- It uses only within-session differences and a scale-free margin, which is the form needed for wheel vs maze in
  hc-5.
- Still synthetic, with one model family (noise-driven Stuart-Landau, linear diffusive coupling). Application to hc-5
  remains Tom's decision and would need its own PREREG.

## Screen 9: nuisance-confound traps for rule v2 (DEVIATIONS 19; fresh sessions 80-89; `results/screen9/`)

Both regimes have the same mechanism (`mixed`, K 25, rho 0.5). Only one nuisance differs per cell; C1 is the
positive control.

| cell | Delta S_TE [CI] (n>0) | rel Delta S_inst_dn [CI] | fires | should | correct |
|---|---|---|---|---|---|
| F1 freq up (+1 Hz) | **+0.0027 [+0.0021, +0.0033] (10)** | -0.003 [-0.040, +0.034] | **yes** | no | **no** |
| F2 freq down (-1 Hz) | -0.0001 [-0.0007, +0.0005] (6) | -0.004 [-0.036, +0.028] | no | no | yes |
| N1 SNR up (10 vs 0 dB) | **+0.0007 [+0.0003, +0.0009] (9)** | -0.209 [-0.253, -0.152] | **yes** | no | **no** |
| N2 SNR down (0 vs 10 dB) | -0.0009 [-0.0013, -0.0006] (0) | +0.245 [+0.184, +0.303] | no | no | yes |
| C1 control: coupling up (K 40 vs 25) | +0.0063 [+0.0058, +0.0068] (10) | +0.026 [-0.018, +0.070] | yes | yes | yes |

**Verdict (frozen criterion): FAILED, with 2 nuisance-caused false positives (F1, N1) and no misses.** Under
DEVIATIONS 19, H3' as drafted is invalid unless theta frequency and SNR are matched or adjusted between regimes.

**Reading:**
- **Both confounds act in one direction.** A regime with a *higher* theta frequency, or a *higher* SNR, gets a
  higher S_TE without any change in coupling. The opposite shifts lower S_TE, so the rule (which needs Delta > 0)
  stays silent there.
- **Frequency.** +1 Hz alone produced 43% of the TE increase that a 60% coupling increase produced (+0.0027 vs
  +0.0063), in 10/10 sessions. The likely cause is that the TE delay is fixed in *samples* (8 = 32 ms), so it spans
  more of a cycle at a higher frequency. The guard cannot help: S_inst_dn does not move (-0.3%).
- **SNR.** Higher SNR raises S_TE slightly (+0.0007, 9/10) and *lowers* S_inst_dn by 21%. The relative guard only
  blocks increases in S_inst_dn, so this change passes it.
- **Relevance to hc-5.** Theta frequency is known to differ with running speed and between wheel and maze, and
  signal quality differs between behavioural states. Both confounds are therefore plausible in the real
  comparison, in the direction that would produce a positive result.
- **Next step: candidate fixes, tested on fresh sessions (DEVIATIONS 20).** (a) Set the TE delay in cycles of each
  regime's own train-free spectral peak. (b) Degrade the higher-SNR regime to match, using the latent-LDS noise
  fraction.

## Screen 10, stage D: rule v3 development check (DEVIATIONS 20; used sessions 80-89; `results/screen10/stage_D/`)

| cell | Delta S_TE_cyc [CI] (n>0) | rel Delta S_inst_dn [CI] | Delta f_c | Delta R | v3 fires | should | v2 fires |
|---|---|---|---|---|---|---|---|
| F1 freq up | +0.0002 [-0.0003, +0.0008] (5) | -0.002 | +1.02 | 0.000 | no | no | yes |
| F2 freq down | **+0.0019 [+0.0013, +0.0025] (10)** | -0.003 | -1.01 | -0.001 | **yes** | no | no |
| N1 SNR up | -0.0002 [-0.0005, +0.0000] (3) | -0.059 | +0.01 | -0.426 | no | no | yes |
| N2 SNR down | -0.0001 [-0.0004, +0.0003] (5) | -0.017 | +0.01 | +0.425 | no | no | no |
| C1 coupling up | +0.0062 [+0.0057, +0.0066] (10) | +0.028 | +0.01 | -0.004 | yes | yes | yes |

**Verdict: stage D FAILED (F2 false positive).** As allowed by DEVIATIONS 20, one documented revision is permitted
before stage C.

**Reading:**
- **Noise matching works.** The SNR confound is gone in both directions (N1: +0.0007 -> -0.0002). The
  relative-guard asymmetry from screen 9 disappears with it (rel Delta S_inst_dn -21% -> -6%).
- **Cycle normalisation over-corrects.** It fixed +1 Hz but turned -1 Hz into a false positive. Under the old
  sample delay only +1 Hz was biased. This asymmetry points to the fixed 6-10 Hz analysis band, not the delay, as
  the main frequency confound (a 9 Hz peak sits close to its upper edge). A development diagnostic of band/delay
  variants on these used sessions decides the revision (`resonet_sync/screens/diag10.py`).
- The coupling control is unaffected by either fix (+0.0062 vs +0.0063 in screen 9).

### Development diagnostics and revision 1 (used sessions 80-89)
Each value is Delta S_TE (A - B), averaged over 10 sessions; the number of sessions with a positive Delta is in
brackets.

| variant | F1 +1 Hz | F2 -1 Hz | C1 coupling |
|---|---|---|---|
| fixed 6-10 Hz band, fixed delay (v2) | +0.0027 (10) | -0.0001 (6) | +0.0063 (10) |
| fixed band, delay in cycles (v3) | +0.0002 (5) | +0.0019 (10) | +0.0062 (10) |
| centred band, fixed delay | +0.0032 (10) | +0.0019 (9) | +0.0064 (10) |
| centred band, cycle delay | -0.0009 (0) | +0.0011 (9) | +0.0062 (10) |
| fixed band, fixed delay, phase-advance rotation | +0.0020 (10) | +0.0004 (7) | +0.0062 (10) |
| **centred band, fixed delay, phase-advance rotation (v3r)** | **-0.0001 (5)** | **+0.0001 (6)** | **+0.0061 (10)** |
| bin dithering (4 grids), any combination | same as without (**invalid test**: past and future grids were shifted together, so the relative offset never changed; review board, Zhao) | | |

**Interpretation.** A uniform frequency shift with white-noise input does not change the physics in real time
(rotating frame), so the delay must stay fixed in time. The confound comes from the estimator: the fixed band, plus
the interaction between the mean phase advance and the 8-bin grid. Both must be removed. The fix was frozen as
DEVIATIONS 20 revision 1 before stage C.

## Screen 10, stage C: rule v3r on fresh sessions 90-99 (DEVIATIONS 20 + revision 1; `results/screen10/stage_C/`)

The verdict was recomputed with the fixed `decide()`, as pre-committed (`results_recomputed.json`). It is valid
(no non-finite values) and identical to the verdict the run wrote.

| cell | Delta S_TE_al [CI] (n>0) | rel Delta S_inst_dn [CI] | Delta f_c | Delta R | v3r fires | should | v2 (unmatched) fires |
|---|---|---|---|---|---|---|---|
| F1 freq up | -0.0002 [-0.0005, +0.0002] (4) | -0.003 | +1.01 | 0.000 | no | no | **yes** |
| F2 freq down | +0.0001 [-0.0002, +0.0004] (5) | -0.001 | -1.02 | 0.000 | no | no | no |
| N1 SNR up | -0.0004 [-0.0007, -0.0001] (2) | +0.028 | -0.01 | -0.427 | no | no | no |
| N2 SNR down | -0.0002 [-0.0006, +0.0001] (3) | -0.020 | -0.02 | +0.426 | no | no | no |
| X1 both nuisances up | -0.0003 [-0.0006, +0.0000] (3) | +0.020 | +1.00 | -0.426 | no | no | **yes** |
| T1 mild common trap (rho 0.5 -> 0.7) | +0.0007 [-0.0001, +0.0016] (6) | **+0.940** [+0.842, +1.027] | -0.01 | +0.003 | no | no | no |
| C1 coupling up | +0.0070 [+0.0064, +0.0076] (10) | +0.025 | -0.02 | -0.004 | yes | yes | yes |
| C2 coupling up, A noisier | +0.0060 [+0.0054, +0.0066] (10) | +0.049 | -0.02 | +0.427 | yes | yes | **no (miss)** |
| C3 coupling up, A slower | +0.0070 [+0.0064, +0.0076] (10) | +0.025 | -1.02 | -0.004 | yes | yes | yes |

**Verdict: rule v3r, all 9 cells correct.** No false positives and no misses. On the same data, the unmatched v2
rule fails 3/9:
- false positives in F1 and X1 (the screen 9 frequency confound, replicated on fresh sessions);
- a miss in C2, a real coupling increase hidden because regime A was noisier.

**What this does and does not show (frozen before the result, DEVIATIONS 20 pre-commitment):**
- **It is not confirmation of v3r.** The F1/F2/X1/C3 cells run on the Euler-Maruyama generator, whose damping
  depends on frequency. That mixes the estimator confound with a small change in the physics, so the result is
  reported as **no counter-example found** within this generator family.
- **Noise matching does both jobs.** It removes the SNR confound (N1 +0.0007 under v2, -0.0004 here) and rescues
  sensitivity when the coupled regime is noisier (C2: missed by v2, found by v3r). This is the first time the
  S_TE_al + matching combination was run.
- **The mild common-input trap** did not even pass the TE leg (6/10 positive). The guard would have blocked it
  anyway (+94% in S_inst_dn).
- **Untested by stage C:** *rhythmic* common input. The review board named it the rival hypothesis, and expects
  this rule to read it as coupling. Rule development is closed. The only remaining synthetic step is the terminal
  calibration (screen 11), with retirement pre-committed if any rhythmic or rho cell has an estimated
  P(fire) > 0.05.

## hc-5 generator-mix plan, Gate 1 runner: DRY RUNS on hypothetical geometry (`results/gate1_laminar/*_DRYRUN.json`)

These are not Gate 1. The real Gate 1 uses the probe geometry that Gate 0 (hc-5 QC, `laminar_coverage`) reports.
The dry runs check that the runner and the analysis path behave.

| geometry | outcome |
|---|---|
| 16 sites x 50 um | **passes**. No nuisance moves Delta (all \|mean\| <= 0.026, sign-flip p >= 0.18): +1 Hz, speed-locked drift, sawtooth asymmetry, amplitude x1.3, 1/f background, 10 deg EC phase lag. EC drive +20% gives mean Delta +0.21, 8/10 positive (exactly at the threshold). |
| 8 sites x 20 um | **fails the coverage gate** (no theta reversal with >= 300 um below the pyramidal layer), as expected. |

**Reading.**
- With adequate laminar coverage, the CSD generator-mix statistic is robust to every nuisance the board listed.
- Its power for a 20% EC change is marginal at 10 sessions. hc-5 has 7, so only effects of about 30% or more are
  realistic.
- Short 8-site shanks make the whole route infeasible. Gate 0 will decide this on the real data.

## Screen 11: terminal calibration of rule v3r (DEVIATIONS 21; `results/screen11/`; fig 8)

**Pre-committed outcome: the TE/LDS within-session rule is RETIRED.** It must not be used on any real data. This is
reported as a negative methods result.

**Calibration.** The rhythmic-drive levels were kappa 4.25 / 5.91 / 9.22, which raise synchrony by 0.5x / 1x / 2x
the synchrony increase of K 25 -> 40 (Delta PLV 0.055).

| cell | Delta PLV | Delta S_TE_al | rel Delta S_inst_dn | P(TE leg) | **P(rule)** | must |
|---|---|---|---|---|---|---|
| null / +1 Hz / self-damping x2 | <= 0 | <= 0 | - | 0.00 | <= 0.00 (TE bound) | not fire |
| -1 Hz | -0.011 | +0.0005 | +0.05 | 0.16 | **0.34** | not fire |
| **rhythmic common input low / mid / high** | +0.032 / +0.056 / +0.111 | +0.0029 / +0.0054 / +0.0115 | +0.06 / +0.08 / +0.11 | 1.00 | **1.00 / 1.00 / 0.94** | not fire |
| white common rho 0.55 / 0.60 / 0.65 | +0.005 / +0.012 / +0.020 | small | +0.46 / +0.70 (0.60 / 0.65) | 0.01 / 0.10 / 0.15 | 0.01 (bound) / 0.00 / 0.00 | not fire |
| volume conduction | +0.166 | +0.0197 | +4.27 | 1.00 | 0.00 | not fire |
| correlated 1/f noise, A 10 dB vs B 0 dB | +0.029 | +0.0017 | -0.25 | 1.00 | **1.00** | not fire |
| coupling K30 / K35 / K40 | +0.023 / +0.044 / +0.064 | +0.0021 / +0.0043 / +0.0065 | +0.08 (K35) | 1.00 | 0.99 (K35) | fire |
| coupling K35 with rhythm in both | +0.052 | +0.0056 | +0.09 | 1.00 | 1.00 | fire |

**Verdict (frozen criteria).**
- Effective alpha = **1.00** (calibration fails).
- Power at K35 = 0.99.
- Retirement is triggered by all three rhythmic cells (DEVIATIONS 20/21 pre-commitment).

**What the numbers say.**
1. **Rhythmic common input is indistinguishable from coupling for this rule.** Per unit of synchrony gained, it
   raises TE as much as coupling does (+0.0054 at Delta PLV 0.056, vs +0.0043 at 0.044 for K35). It also moves
   the denoised instantaneous correlation by the same small amount (+8% vs +8%).
   - A shared rhythm is a latent oscillator with memory. The LDS absorbs it into the lagged dynamics A, not into
     the instantaneous covariance Q, exactly as the information theorist and the statistician predicted in
     board round 2.
   - This is the rival hypothesis for hc-5 (septal/EC theta drive), so the rule cannot answer the question it was
     built for.
2. **What the guard does catch.** It works for instantaneous sources: white common input (P 0.00 at rho 0.60 and
   0.65) and volume conduction (rel Delta +427%, P 0.00).
3. **Noise matching fails under spatially correlated 1/f noise.** Q1 fires with P 1.00, because the diagonal-R
   model mis-estimates correlated noise and the guard then sees a 25% *drop*. This confirms the board's concern
   about real sensor noise.
4. **The frequency fix is incomplete with the exact integrator** (-1 Hz: P 0.34). Part of the stage D/C
   frequency "fix" was compensating the Euler damping artefact.

**Consequences (already in place).**
- **hc-5:** the adopted plan is the generator-mix hypothesis (`docs/PREREG_H3_GENERATOR_MIX_DRAFT.md`), not a
  coupling statistic.
- **Synthetic programme:** screens 1-11 are closed as a methods result. Within-session phase-dependence statistics
  (TE plus instantaneous-correlation guard, with frequency and noise adjustments) separate coupling from
  *instantaneous* common input. They do not separate it from *rhythmic* common input, which is the physiologically
  relevant alternative.
