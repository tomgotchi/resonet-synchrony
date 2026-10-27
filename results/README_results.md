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
