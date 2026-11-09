# resonet-synchrony

ResoNet follow-up "Pitch 1": does nonlinear coupling between oscillators earn its cost where the rhythm is
generated internally (hippocampal wheel running), and not where it is stimulus-driven?

**Where things stand (2026-11-09):**
- **Phase 1** (HGD replication) is stopped at 6/14 subjects on Colab; re-running the notebook resumes it. The
  interim is descriptive only.
- **Review board 2026-11-09** (six agents, two rounds; [docs/REVIEW_BOARD.md](docs/REVIEW_BOARD.md)):
  - The Phase 1 "phase" contrasts are not phase-specific: they fire on source-power-only data, are inflated by
    dilution, and the fixed-roll surrogate leaks phase.
  - **DEVIATION 23** freezes the Phase 1 analysis rules before subjects 7-14: exact sign-flip test, TOST margin
    0.05, an intersection-union rule, wording limits, and a frozen-bank "phase beyond band power" family F5.
  - **DEVIATION 24**, a calibration screen with known ground truth, is running and decides that wording.
- **Synthetic mechanism programme (screens 1-11) is closed.** The formal reason is in
  [docs/NONIDENTIFIABILITY.md](docs/NONIDENTIFIABILITY.md): in the linear-Gaussian regime, coupling and rhythmic
  common input have identical laws. The Phase 2 emergent data are purely second order (Whittle oracle 0.999).
- **hc-5:** the generator-mix plan now uses **Gate 1 v2** (DEVIATION 25). It fits generator templates instead of
  picking sites, because the v1 site picker never found the radiatum sink. The dev dry run passes every cell
  except DG crosstalk (marginal). The validation run and the geometry pass map follow. Gate 0 needs the hc-5 QC.

## Start here

| document | what it is |
|---|---|
| [docs/TASK_PITCH1.md](docs/TASK_PITCH1.md) | the original spec |
| [DEVIATIONS.md](DEVIATIONS.md) | every departure and judgement call; **frozen protocols** for each screen (append-only) |
| [results/README_results.md](results/README_results.md) | all results, phase by phase and screen by screen |
| [results/figures/](results/figures/) | figures (regenerate: `scripts/make_figures.py`) |
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
  data_hgd.py            HGD cache loader;   data_hc5.py, qc_hc5.py: hc-5 loading and QC (no hypothesis test)
  sanitize.py            shared with phasetrack;   _vendor.py: imports vendor/ unmodified
resonet_sync/screens/    frozen synthetic screen runners screen1..11, diag10*, calib_p1 (DEVIATION 24), gate1_laminar (v1), gate1_v2
scripts/make_figures.py  all figures in results/figures/
scripts/                 phase1_posthoc.py (F5, D2, baselines), nonident_check.py, whittle_phase2.py
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
