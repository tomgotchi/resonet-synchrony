# resonet-synchrony

ResoNet follow-up "Pitch 1": does nonlinear coupling between oscillators earn its cost where the
rhythm is internally generated (hippocampal wheel running) and not where it is stimulus-driven? Spec:
[TASK_PITCH1.md](TASK_PITCH1.md). Deviations and judgement calls: [DEVIATIONS.md](DEVIATIONS.md).
Results: [results/README_results.md](results/README_results.md). `PREREG.md` will be written together
with Tom and committed before Phase 3.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml                       # set paths.hgd_cache_hg (FULL-BAND cache)
.venv/bin/python -m pytest -q tests                      # ~3 min on CPU
.venv/bin/python -m resonet_sync.run --phase 1 --smoke --arms all     # tiny synthetic end-to-end
.venv/bin/python -m resonet_sync.run --phase 1 --arms core --subjects 1-14   # published 6 arms first
.venv/bin/python -m resonet_sync.run --phase 1 --arms all  --subjects 1-14   # + new arms (resumes)
.venv/bin/python -m resonet_sync.run --phase 1 --arms core --rk4-substeps 2  # integrator sensitivity
```

Package `resonet_sync/`: `banks.py` (lin, nl_uncoupled, nl_coupled, nl_uncoupled_wide; Hz-parameterized),
`readouts.py` (sync, power, surrogate, sync+power, trial_shuffle), `train.py` (published training loop
plus same-class partners), `data_hgd.py`, `stats.py`, `run.py`, `sanitize.py` (shared with phasetrack).
`vendor/` holds the unmodified published code, and the tests check its hashes.
