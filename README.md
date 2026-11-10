# resonet-synchrony

ResoNet follow-up "Pitch 1": does nonlinear coupling between oscillators earn its cost where the rhythm is
generated internally (hippocampal wheel running), and not where it is stimulus-driven?

**Where things stand (2026-11-10):**
- **Critique of the published model** ([docs/CRITIQUE_AND_V2.md](docs/CRITIQUE_AND_V2.md), fig 10):
  - In all 14 published checkpoints the trained oscillators sit at 0.36-1.55 Hz, while the data are 4-125 Hz.
  - The paper's "phase" contrast (sync - surrogate) is +0.31 on simulated data with no phase information.
  - The high-gamma comparison also changed the sampling rate.
- **ResoNet v2** (`resonet_sync/v2.py`, DEVIATION 27): resonators that tile the band, real spatial filters per
  band, and a readout split into power, zero-lag and lagged blocks.
  - Ground truth: its lag contrast passes all pre-set criteria (ignores source power, finds lagged coupling).
  - Real EEG, HGD subjects 1-6, cross-validation inside the training recordings (development only): v2 0.88 against
    FBCSP 0.89; the lag block alone 0.74 with its nulls at chance; it adds little once power is in the model.
  - A confirmatory test on untouched data is specified but not run.
- **Phase 1** (replication of the published arms) is stopped at 6/14 subjects on Colab. Its wording is fixed in
  advance (DEVIATIONS 23-24): no Phase 1 result can be called "phase".
- **Synthetic mechanism programme (screens 1-11)** is closed; **hc-5** waits for its QC (Gate 0), and Gate 1 v2
  fails (DEVIATION 26).

## Start here

| document | what it is |
|---|---|
| [docs/TASK_PITCH1.md](docs/TASK_PITCH1.md) | the original spec |
| [DEVIATIONS.md](DEVIATIONS.md) | every departure and judgement call; **frozen protocols** for each screen (append-only) |
| [results/README_results.md](results/README_results.md) | all results, phase by phase and screen by screen |
| [results/figures/](results/figures/) | figures (regenerate: `scripts/make_figures.py`) |
| [docs/CRITIQUE_AND_V2.md](docs/CRITIQUE_AND_V2.md) | **start here**: what the published model computes, what that means for the manuscript, and ResoNet v2 with its ground-truth and real-data results |
| [docs/REVIEW_BOARD.md](docs/REVIEW_BOARD.md) | review board sessions: findings, verdicts, overturns, plans (latest 2026-11-09) |
| [docs/NONIDENTIFIABILITY.md](docs/NONIDENTIFIABILITY.md) | why passive phase statistics cannot separate coupling from rhythmic common input (two constructions, checked by `scripts/nonident_check.py`) |
| [docs/PREREG_H3_GENERATOR_MIX_DRAFT.md](docs/PREREG_H3_GENERATOR_MIX_DRAFT.md) | **adopted** hc-5 plan: generator mix (CA3 radiatum vs EC3 l-m CSD), frozen after Gate 0 (laminar coverage) + Gate 1 (forward-model null) |
| [docs/PREREG_H3_REDESIGN_DRAFT.md](docs/PREREG_H3_REDESIGN_DRAFT.md) | superseded H3' draft (coupling between sites) |
| [docs/RUN_PHASE1_LOCAL.md](docs/RUN_PHASE1_LOCAL.md) | exact commands for the Phase 1 GPU run |
| [notebooks/resonet_synchrony_full_colab.ipynb](notebooks/resonet_synchrony_full_colab.ipynb) | **standalone Colab pipeline**: Drive data, tests, Phase 1 GPU training (resumable to Drive), results, hc-5 QC, synthetic figures, push |

## Layout

```
resonet_sync/            library
  banks.py               oscillator banks: lin, nl_uncoupled, nl_coupled, nl_uncoupled_wide (Hz-parameterised)
  readouts.py            sync, power, surrogate (fixed roll), surrogate_random (fresh roll), sync+power, power+surrogate[_random]
  train.py               published training loop + same-split shuffle partners
  decomposition.py       evoked-vs-induced decomposition (analysis control, not a decoder arm)
  run.py                 Phase 1 (HGD) and Phase 2 (synthetic) runners
  synth.py               synthetic generators: emergent / driven / driven_common / mixed (+ screen-11 options)
  mechanism.py           phase TE, Granger, residual correlation
  lds.py                 latent linear dynamical system (EM) and denoised statistics
  nuisance.py            rule v3/v3r adjustments: centred band, phase rotation, LDS noise matching
  laminar.py             CA1 laminar forward model v1 and v2 (disc sources), CSD, layer localisation, generator-template fit
  stats.py               paired tests, Holm, bootstrap CI, exact sign-flip, TOST (DEVIATION 23)
  baselines.py           FBLV / FBCSP / tangent-space baselines and frozen-bank analyses (DEVIATION 23)
  v2.py, v2_probe.py     ResoNet v2: cross-spectral bank and readout, CSP init; closed-form path with controls
  data_hgd.py            HGD cache loader;   data_hc5.py, qc_hc5.py: hc-5 loading and QC (no hypothesis test)
  sanitize.py            shared with phasetrack;   _vendor.py: imports vendor/ unmodified
resonet_sync/screens/    frozen synthetic screen runners screen1..11, diag10*, calib_p1 (DEVIATION 24), gate1_laminar (v1), gate1_v2
scripts/make_figures.py  all figures in results/figures/
scripts/                 phase1_posthoc.py (F5, D2, baselines), nonident_check.py, whittle_phase2.py
                         published_forensics.py, published_null_test.py (critique of the published model)
                         build_hgd_dev_cache.py, v2_hgd_dev.py, v2_probe_run.py, v2_probe_controls.py (v2 on HGD)
notebooks/               Colab: full pipeline (resonet_synchrony_full_colab) and hc-5 QC only
vendor/                  unmodified published code (hashes checked by tests)
tests/                   pytest suite
docs/                    spec, review board, drafts, run instructions
results/                 phase1/, phase2/, screen1..11/, calib_p1/, gate1_laminar/, gate1_v2/, figures/
```

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # exact versions: requirements.lock
cp config.example.yaml config.yaml                                       # set paths.hgd_cache_hg (FULL-BAND cache)
.venv/bin/python -m pytest -q tests                                      # ~5 min on CPU
.venv/bin/python -m resonet_sync.run --phase 1 --smoke --arms all        # tiny synthetic end-to-end
.venv/bin/python -m resonet_sync.run --phase 1 --arms core --subjects 1-14   # published 6 arms (GPU)
.venv/bin/python -m resonet_sync.screens.screen10 --stage C --workers 4  # a frozen screen (resumable)
.venv/bin/python scripts/make_figures.py                                  # all figures
```

Rules kept throughout:
- **Data:** never committed (`.gitignore`).
- **`vendor/`:** never edited.
- **Protocols:** frozen in DEVIATIONS before each run.
- **Seeds and sessions:** fresh for every confirmatory step.
- **Resumable checkpoints:** `results/**/partial.json` is not tracked; the final `results.json` of each screen
  holds every per-dataset value.

## Review board
Six reviewer personas ship with the repo as agents (`.claude/agents/board-*.md`), with a skill that runs the
whole procedure (`.claude/skills/review-board/SKILL.md`). In Claude Code, opened in this repo, type
`/review-board <what to review>`. The steps are independent reviews, judge verification, cross-examination,
and a record in `docs/REVIEW_BOARD.md`.
