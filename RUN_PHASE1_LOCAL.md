# Running Phase 1 on your laptop (HGD full-band cache, GPU)

The cache already exists on your machine (`C:\Users\asusd\Documents\RESEARCH\subj_cache_hg`, built by
`build_hg_cache.py`). Do **not** use `subj_cache` (low-passed); the runner refuses a low-passed cache anyway.

```powershell
git clone https://github.com/tomgotchi/resonet-synchrony
cd resonet-synchrony
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # install a CUDA build of torch if pip gives the CPU one
copy config.example.yaml config.yaml
#   edit config.yaml:  paths.hgd_cache_hg: C:\Users\asusd\Documents\RESEARCH\subj_cache_hg
#                      paths.published_results_hg: C:\...\ResonanceNetwork\results_hg   (for the comparison)
.venv\Scripts\python -m pytest -q tests               # ~3 min; must be all green
.venv\Scripts\python -m resonet_sync.run --phase 1 --smoke --arms all

# 1) the urgent question first: the 6 published arms (~ published runtime, ~45-90 min/subject on the 3070)
.venv\Scripts\python -m resonet_sync.run --phase 1 --arms core --subjects 1-14
# 2) the remaining arms (resumes per subject; ~2.5-3 days total for everything)
.venv\Scripts\python -m resonet_sync.run --phase 1 --arms all --subjects 1-14
# 3) integrator sensitivity (DEVIATIONS #2): written to results/phase1_rk4x2/
.venv\Scripts\python -m resonet_sync.run --phase 1 --arms core --rk4-substeps 2 --subjects 1-14
```

Outputs: `results/phase1/subj_XX.json` (written after every arm, so you can stop and resume) and
`results/phase1/summary.json` (decoder contrasts with Holm correction, the evoked/induced decomposition kept
separate, and a by-subject comparison with the published `results_hg` numbers). Commit the `results/phase1/`
JSONs (they are small) and send them back for the Phase 1 report. Never commit the cache.
