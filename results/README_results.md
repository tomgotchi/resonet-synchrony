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

## Exploratory mechanism-statistic screen (DEVIATIONS 11; synthetic only; `results/screen/screen_results.json`)

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
