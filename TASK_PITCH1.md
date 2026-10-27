# TASK: Emergent vs. Driven Synchrony — a Controlled Dissociation (ResoNet follow-up, "Pitch 1")

Read this whole file before writing code. Work in phases, and **stop and report after each phase**
before starting the next one. If the TASK.md phase-tracker work is in progress, finish its Phase 0 and
stop there. Its tests and sanitization helpers can be reused here.

## 1. The question

ResoNet found that inter-oscillator phase synchrony carries class information beyond band power (capacity-
matched surrogate: +0.098 in the nonlinear bank, +0.147 in the linear bank, on HGD). It also found that
nonlinear coupling added nothing. The conclusion was that the phase effect is **stimulus-driven**,
not emergent.

**Pitch 1 asks:** does coupling help where the brain's rhythm is *internally generated*?
- If yes, the result is a dissociation: coupled oscillator decoders earn their cost only for internally
  generated dynamics.
- If no, coupling buys nothing anywhere. That's also a clear, useful answer for the oscillatory-network field.

### Pre-registered hypotheses (freeze in `PREREG.md` and commit BEFORE Phase 3)
- **H0 (method validity):** On synthetic data, the nl-vs-lin contrast detects coupling-dependent class
  information when it exists, and does NOT detect it when class information is purely input-driven.
- **H1 (positive control):** In rat CA1, spike theta phase carries position information beyond firing rate,
  tested under a capacity-matched phase shuffle.
- **H2 (replication):** `sync > surrogate` holds in every dataset.
- **H3 (primary):** The coupling advantage `(nl_coupled − lin)` is larger during **wheel running**
  (internally generated sequences) than during **maze running** (sensory-driven), within the same animal and
  electrodes (hc-5).
- **H3b (secondary):** The coupling advantage is larger in hippocampus than in HGD motor cortex.

## 2. Prerequisite fix: oscillator frequency parameterization

In `resonance_network.py`, per-sample rotation is `omega*h`, with `omega ~ U(0.5, 4)` and `h = 0.01`. That's
about **0.1–0.8 Hz at fs = 125 Hz**, so the bank is initialized far below the bands of interest. The paper's
"oscillators settle in delta/theta" finding may be an initialization artifact, and the nl-vs-lin comparison
may be too.

Fix it in a new module rather than editing `vendor/`:
- Parameterize each oscillator by `f0` (Hz) and bandwidth `bw` (Hz).
- `omega_per_sample = 2*pi*f0/fs`, `mu_per_sample = -2*pi*bw/fs`.
- Initialize `f0` log-spaced across the analysis band: 1–40 Hz for HGD, 2–150 Hz for hippocampus.
- Keep everything else (encoder, readouts, surrogate) identical to the published code.

## 3. Arms (identical across every dataset)

Bank factorial (nonlinearity × coupling), with three readouts each:

| Bank | Cubic saturation | Coupling W | Isolates |
|---|---|---|---|
| `lin` | no | no | stimulus-driven baseline |
| `nl_uncoupled` | yes | frozen at 0 | nonlinearity alone |
| `nl_coupled` | yes | trainable | nonlinearity + coupling (the target) |

Readouts: `sync` (pairwise coherence), `power`, `surrogate` (fixed per-oscillator circular shift; capacity-
matched). Add `sync+power` against `power` for the incremental-value contrast the ResoNet paper is missing.

Extra arm for trial-structured data:
- `trial_shuffle`: compute coherence between oscillator i from trial n and oscillator j from a different
  trial m of the same class. This keeps evoked (class-locked) structure and destroys trial-specific
  (induced) coupling.

**Capacity control for coupling:** `nl_coupled` has about N² more parameters than `nl_uncoupled`. Add
`nl_uncoupled_wide`, which gets the same extra parameter count in the encoder, so that a coupling win
can't be explained by parameter count.

Baselines: EEGNet and ShallowConvNet (HGD), plus a linear decoder on binned rates (hippocampus).

## 4. Data

- **HGD:** existing `subj_cache` (14 subjects, official split). Confirm whether the cache is low-passed or
  full-band. Use full band.
- **hc-5 (CRCNS):** Pastalkova et al. 2008. One rat, silicon probe in CA1. Each session contains
  left/right alternation on a maze **and wheel running between trials**, where cell-assembly sequences are
  internally generated. Sessions: `i01_maze05.005`, `maze06.002`, `maze06.005`, `maze08.001`, `maze08.004`,
  `maze13.003`, `maze15.002` (2.5–16 GB each). Downloading requires Tom's CRCNS login, so **ask him**.
  Read the dataset's description PDF for file formats (it documents `Track.eeg` at 1250 Hz and theta-phase
  fields). Verify, don't assume.
- **hc-2 (CRCNS):** Mizuseki et al. 2009. Multiple rats. Tom has spikes and position for 5 sessions and raw
  `.dat` for ec013.527/528. Use it for H1 and for multi-animal replication of the maze side. Additional
  sessions with LFP need a CRCNS download, so ask first.

### Task definitions (matched across regimes)
- **Maze (hc-5, hc-2):** 4 position bins along the track, 1 s windows during running.
- **Wheel (hc-5):** 4 elapsed-time bins within each wheel run, 1 s windows. Secondary task: upcoming L/R choice.
- **HGD:** existing 4-class task.
- Same number of classes, window length, and oscillator count N wherever possible.
- Report balanced accuracy and the chance threshold for each test-set size (Combrisson & Jerbi 2015;
  `chance_threshold()` already exists).

## 5. Phases

**Phase 0: setup.** Create the repo `resonet-synchrony` using the same conventions as TASK.md Section 0:
private repo, `vendor/`, data gitignored, `config.yaml`. Reuse the causality and sanitization tests.

**Phase 1: recalibrated replication on HGD.** Rerun all ResoNet arms with Hz parameterization plus the new
arms (`nl_uncoupled`, `nl_uncoupled_wide`, `trial_shuffle`, `sync+power`).
- **STOP and report.** If the published phase effect (sync > surrogate) disappears or reverses under correct
  initialization, Tom needs to know before anything else, because it affects the paper under review.

**Phase 2: synthetic validation (H0).** Two generators, power-matched across classes:
- *Driven:* class determines the phase relationships of a common external input. Units are uncoupled.
  Prediction: `lin ≈ nl_coupled`.
- *Emergent:* class determines the coupling matrix of a noise-driven Stuart-Landau/Kuramoto network. There is
  no class information in the input. Prediction: `nl_coupled > lin`.
- If `nl_coupled` doesn't beat `lin` on the emergent generator, the H3 test is underpowered and any real-data
  null is uninterpretable. Report this and stop.

**Phase 3: hc-5 within-animal test (H3, primary).** Same electrodes and sessions, wheel vs. maze.
- LFP: ≥8 CA1 channels spread across shanks. Use bipolar or local re-referencing (a common reference inflates
  coherence). Decimate causally or zero-phase (offline decoding is fine here) to 250 Hz.
- Split: temporal, within session, before windowing (as in `hc2_loader.split_and_window`). Never let windows
  from the same wheel run or maze trial land in both train and test.

**Phase 4: hc-2 (H1 and replication).**
- H1: spike theta phase vs. rate. Keep spike times at 20 kHz; phase comes from LFP (`.dat`), not from 25.6 ms
  bins. Surrogate: shuffle spike phases across theta cycles within each position bin, keeping counts.
- Maze-side replication of the Phase 3 arms across animals.

**Phase 5: combined analysis.** H3b cross-dataset comparison, figures, `README_results.md`.

## 6. Statistics

- Units: session (hc-5, within one animal), animal with sessions nested (hc-2), subject (HGD), seed (synthetic).
  Never treat windows as independent samples.
- **H3 interaction:** per session, `d_wheel = acc(nl_coupled) − acc(lin)` on wheel and `d_maze` the same on
  maze. Test `d_wheel − d_maze` across sessions (Wilcoxon plus bootstrap CI), and also within-session
  bootstrap across held-out windows.
- hc-5 has 7 sessions from ONE rat. Report this as a within-animal mechanistic result, not population-level.
- Cross-dataset (H3b): compare standardized effects (dz), not raw accuracy, and state the confounds (species,
  recording type, SNR, task).
- Holm-Bonferroni within each hypothesis family. Paired effect sizes and 95% CIs everywhere.

## 7. Leakage and integrity rules (non-negotiable)

- All fitted choices (channels, referencing, band, normalization stats, hyperparameters) come from the train
  split only.
- No epoch or checkpoint selection on test data. Early stopping uses a validation split carved from train.
  (This is exactly how the HPC-GCM leak happened.)
- `PREREG.md` is committed before Phase 3 runs. Any later deviation is logged in `DEVIATIONS.md` with the reason.
- Fixed seeds. Every result JSON stores its config, seed, git commit hash, and data file checksums.
- Smoke-test every new arm on tiny data first: loss decreases, gradients reach every parameter group, and the
  surrogate actually changes the features.

## 8. Deliverables

```
resonet_sync/
  banks.py         # lin, nl_uncoupled, nl_uncoupled_wide, nl_coupled (Hz-parameterized)
  readouts.py      # sync, power, surrogate, trial_shuffle, sync+power
  synth.py         # driven vs emergent generators (Phase 2)
  data_hgd.py  data_hc5.py  data_hc2.py
  train.py  stats.py  run.py  # CLI: --phase {1..5}
  tests/
PREREG.md  DEVIATIONS.md
results/phase{1..5}/  figures/  README_results.md
```

## 9. Do not

- Don't modify the published ResoNet code paths in `vendor/`.
- Don't report best-seed or best-session results.
- Don't call the wheel result "intrinsic dynamics" unless the controls hold. Use "internally generated
  (wheel-running) regime" as the hc-5 dataset's own terminology does.
- Don't start Phase 3 if Phase 2 shows the test can't detect emergent coupling.
