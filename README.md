# resonet-synchrony

ResoNet follow-up "Pitch 1": does nonlinear coupling between oscillators earn its cost where the rhythm is
generated internally (hippocampal wheel running), and not where it is stimulus-driven?

**Where things stand (2026-11-03):**
- **Phase 1** (HGD replication) is written and smoke-tested. It runs on Tom's GPU.
- **Phase 2** (synthetic H0) hit its stop rule.
- **Synthetic mechanism-statistic screens 1-10** are documented in DEVIATIONS 11-20. Screen 10 stage C is
  running.
- **A six-persona review board** sent the TE/LDS within-session rule to a final calibration (screen 11). **It
  failed and is retired** (2026-11-04): it reads rhythmic common input as coupling (P(fire) 0.94-1.00). See
  [docs/REVIEW_BOARD.md](docs/REVIEW_BOARD.md) and results "Screen 11".
- **hc-5** now follows the generator-mix plan (CSD of the CA3 vs EC3 theta sinks), gated by laminar coverage.

## Start here

| document | what it is |
|---|---|
| [docs/TASK_PITCH1.md](docs/TASK_PITCH1.md) | the original spec |
| [DEVIATIONS.md](DEVIATIONS.md) | every departure and judgement call; **frozen protocols** for each screen (append-only) |
| [results/README_results.md](results/README_results.md) | all results, phase by phase and screen by screen |
| [results/figures/](results/figures/) | figures (regenerate: `scripts/make_figures.py`) |
| [docs/REVIEW_BOARD.md](docs/REVIEW_BOARD.md) | review board rounds 1-2: findings, verdicts, revised plan |
| [docs/PREREG_H3_GENERATOR_MIX_DRAFT.md](docs/PREREG_H3_GENERATOR_MIX_DRAFT.md) | **adopted** hc-5 plan: generator mix (CA3 radiatum vs EC3 l-m CSD), frozen after Gate 0 (laminar coverage) + Gate 1 (forward-model null) |
| [docs/PREREG_H3_REDESIGN_DRAFT.md](docs/PREREG_H3_REDESIGN_DRAFT.md) | superseded H3' draft (coupling between sites) |
| [docs/RUN_PHASE1_LOCAL.md](docs/RUN_PHASE1_LOCAL.md) | exact commands for the Phase 1 GPU run |
| [notebooks/resonet_synchrony_full_colab.ipynb](notebooks/resonet_synchrony_full_colab.ipynb) | **standalone Colab pipeline**: Drive data, tests, Phase 1 GPU training (resumable to Drive), results, hc-5 QC, synthetic figures, push |

## Layout

```
resonet_sync/            library
  banks.py               oscillator banks: lin, nl_uncoupled, nl_coupled, nl_uncoupled_wide (Hz-parameterised)
  readouts.py            sync, power, surrogate, sync+power, trial_shuffle readouts
  train.py               published training loop + same-split shuffle partners
  decomposition.py       evoked-vs-induced decomposition (analysis control, not a decoder arm)
  run.py                 Phase 1 (HGD) and Phase 2 (synthetic) runners
  synth.py               synthetic generators: emergent / driven / driven_common / mixed (+ screen-11 options)
  mechanism.py           phase TE, Granger, residual correlation
  lds.py                 latent linear dynamical system (EM) and denoised statistics
  nuisance.py            rule v3/v3r adjustments: centred band, phase rotation, LDS noise matching
  laminar.py             CA1 laminar forward model (CA3/EC3 generators), CSD, layer localisation, generator mix
  stats.py               paired tests, Holm, bootstrap CI
  data_hgd.py            HGD cache loader;   data_hc5.py, qc_hc5.py: hc-5 loading and QC (no hypothesis test)
  sanitize.py            shared with phasetrack;   _vendor.py: imports vendor/ unmodified
resonet_sync/screens/    frozen synthetic screen runners screen1..screen10 + diagnostics diag10, diag10b
scripts/make_figures.py  all figures in results/figures/
notebooks/               Colab: full pipeline (resonet_synchrony_full_colab) and hc-5 QC only
vendor/                  unmodified published code (hashes checked by tests)
tests/                   pytest suite
docs/                    spec, review board, drafts, run instructions
results/                 phase2/, screen1..screen10/ (one folder per frozen screen), figures/
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
