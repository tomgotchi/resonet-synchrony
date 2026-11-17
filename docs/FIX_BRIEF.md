# Fix brief for the implementing model (from the review board of 2026-11-17)

You are implementing fixes the board already decided on. Do not redesign anything. If a fix would need a decision that is not
written here or in `DEVIATIONS.md`, stop and ask Tom. Read first: `docs/REVIEW_BOARD.md` (session 2026-11-17),
`docs/MANUSCRIPT_REDLINE.md`, `docs/NOTE_TO_COAUTHORS.md`, `DEVIATIONS.md` (29 and addendum 1, 30 and addendum 1, 31).

## Hard rules
- Do **not** edit `/home/user/reso-net/PAPER_DRAFT.md` or anything in `vendor/`. The authors decide the manuscript text.
- Do **not** open HGD test recordings or subjects 7-14 for any v2 work. (The control and DEVIATION 31 use the published
  checkpoints and the authors' own test trials only; they run on Tom's Colab, not here.)
- Never commit data or checkpoints. No tokens in chat or in files. Commits are authored as `tomgotchi
  <tomoliverpresado@gmail.com>`; **no Co-Authored-By and no Claude-Session lines**. No PR unless Tom asks.
- `DEVIATIONS.md` is append-only. Do not change a frozen entry; add an addendum if something is wrong.
- Do not read `results/published_null_nl/` until `results/published_null_nl/results.json` exists. Do not touch the outputs of
  Tom's Colab run (Drive folder `ResonanceNetwork_OFFICIAL/resonet_synchrony_runs/`).
- Machine restarts kill background jobs. After a restart run `/tmp/claude-0/-home-user/eae2bd7c-1832-5027-8478-885374150389/scratchpad/relaunch_nl.sh`
  (resumable). Run it from a command line that does not itself contain the string `published_null_test`.

## Step 0: preconditions (do these first, change nothing)
1. `ps -eo pid,etime,args | grep "[.]venv/bin/python"`: is the nonlinear null test still running? If not and
   `results/published_null_nl/results.json` is absent, relaunch it.
2. Ask Tom: "Does the Colab you are running have a section 4a (the oscillator frequencies in Hz)?" If not, he opened an older copy.
   Then the control's JSONs will be on Drive under `.../resonet_synchrony_runs/bandwidth_control/` and are **not** pushed.
   Tell him to download that folder (or re-run section 9 of the new notebook) and put the JSONs in
   `results/bandwidth_control/<tag>/`. Do not do it for him.
3. `.venv/bin/python -m pytest -q tests` passes before you start (about 5-12 min).

## Step 1: scripts and notebook
**1a. `scripts/published_forensics.py` and notebook section 4a: all 14 subjects.**
- Compute `min_correlation_between_gain_curves` and `plus_minus_selectivity_db` for every checkpoint (function `response`),
  store them in `per_subject` of the JSON, and keep `response_subject_1` for compatibility.
- Notebook 4a prints the minimum over the 14 subjects and labels it. Verified values to expect: minimum correlation 0.9996 and
  +f/-f selectivity 1.9 dB at 10 Hz in all 14 (a real 10 Hz resonator: 36 dB).
- Also report, per subject, the largest eigenvalue of the Hermitian part of `diag(mu + i omega) + W` (expect -0.258 to -0.377).
- Acceptance: `scripts/published_forensics.py --ckpt-dir /home/user/reso-net/results_hg` finishes in about 10 s; the stored
  `results/published_forensics.json` is regenerated and differs only by the added fields; a test checks the new fields.

**1b. `scripts/bandwidth_control.py` and notebook 4b: the extra pipeline cell (DEVIATION 29 addendum 1).**
- Add `repro_lp48_decim4`: `--run --cache <lp48 cache> --decim 4 --tag repro_lp48_decim4 --subjects 1-4` (all six arms). Put it
  after the existing `repro_full_decim2` line and after `--make-lowpass`.
- Acceptance: run it end to end on a tiny synthetic cache with `--epochs 1` (see how it was smoke-tested: a synthetic
  `subj_1.npz` with `X_tr, y_tr, X_te, y_te`, 6 channels, 2250 samples; results go to `results/bandwidth_control/<tag>/`; **delete
  that folder afterwards**). The notebook still parses and every code cell compiles once `!` lines are replaced by `subprocess`.

**1c. New `scripts/bandwidth_analysis.py` implementing the amended rule of DEVIATION 29 addendum 1.**
- Inputs: the published restricted cell (`/home/user/reso-net/RESULTS_n14_COMPLETE/ablation_results.json`), the published full cell
  (`/home/user/reso-net/results_hg/ablation_results.json`), the new cells (`results/bandwidth_control/<tag>/subj_XX.json`, field
  `acc`), tags `lp48_decim2` (A), `full_decim4` (B), `repro_full_decim2` (F'), `repro_lp48_decim4` (R').
- Outputs, per arm: T = F - R, Bn = F - A, Rt = A - R (mean, 95% t interval, exact sign-flip p from `resonet_sync/stats.py`), the
  reading (mostly band / mostly sampling rate / mixed / no gain to attribute), the same computed with R' where it exists, B as a
  descriptive line, and the pipeline-check differences (12 cells, mean signed and mean absolute). The ICA-cleaned run
  (`results_nonemg/ablation_results.json`) is printed beside F.
- Acceptance: `tests/test_bandwidth_analysis.py` on synthetic stand-ins covers: the identity T = Bn + Rt; an arm with T <= 0 gives
  "no gain to attribute"; one example each of mostly band, mostly rate and mixed; the script runs when R' and F' are missing.
  **Do not run it on real new cells until Tom says the Colab run has finished.**

**1d. New `scripts/surrogate_retention.py` for DEVIATION 31.**
- For each of the 14 checkpoints (`RNsync_S*.pt`) and its subject's test trials from a cache directory given on the command
  line, run the published `ResonanceNetwork` with `return_traj=True`, build unit phasors, and compute rms |C_ij| over the 120
  pairs with and without the surrogate shift (`surr_frac * T` roll, as in `vendor/resonance_network.py`), plus the per-bin ratio
  by relative shift (under 0.5 s, 0.5-1 s, over 1 s), as in `docs/prototypes/surrogate_retention_prototype.py` (a working
  prototype on synthetic input; run it from the repository root with the project's `.venv`).
- Acceptance: runs on the synthetic prototype input and reproduces: rms |C| 0.170 unshifted and 0.154 shifted for checkpoint 1
  (seed 0 input as in the prototype); a test checks the shapes and that a zero shift gives ratio 1. A notebook cell 4c runs it on
  `LOCAL_CACHE` (test split of each subject) for Tom's Colab; default off (`RUN_SURROGATE_RETENTION = False`). It uses the
  authors' own test trials with the published checkpoints only.

**1e. A stub test for `published_null_test.py --model nl`:** monkeypatch `rn()` and assert the class (`ResonanceNetwork`) and the
arm set (`power, sync, surrogate`) are chosen, and that the default (`lin`) path is unchanged.

**1f. A notebook test:** all code cells compile after replacing `!cmd` by `pass`; the intro table mentions 4a, 4b.

## Step 2: documents
**2a. `docs/MANUSCRIPT_REDLINE.md`.** Make these changes; keep the HELD marking; cite section headings, not line numbers.
1. Terminology table: "capacity-controlled" becomes "dimension-matched" everywhere.
2. 3.2, surrogate description: replace "destroys all cross-oscillator structure, including the zero-lag covariance..." by: "keeps
   each oscillator's own series and the 240-dimensional feature space, and removes the time-aligned part of the cross-terms
   only for pairs whose relative shift exceeds the oscillators' memory (0.4-0.9 s). In the published checkpoints 44% of the 120
   pairs have a relative shift under 1 s (5% under 0.1 s), the same in all 14 subjects. [Add the DEVIATION 31 result when it is in.]"
3. Abstract, 4.4, 5, 6: "band-power-equivalent" becomes "the raw sync - power gap is -0.004 (95% CI -0.052 to +0.045), equivalent
   within +/-0.05 (TOST p = 0.03) but not within +/-0.03"; "nonlinear coupling adds nothing" becomes "adds no detectable gain
   (nl - lin sync +0.002, 95% CI -0.041 to +0.044)".
4. The +0.31: "shows that the contrast is not specific; its size depends on the generator (ERD depth 0.15, set so an oracle reaches
   about 0.7 on a pilot seed) and is not comparable with +0.163". Replace "[+0.31, linear bank]" by the nonlinear result
   only when `results/published_null_nl/results.json` is complete, and say which model each number is for.
5. "Contractive": "the linear part of the trained network is contractive in all 14 subjects (largest Hermitian eigenvalue
   -0.258) and the cubic term is 0.5-20% of the damping (subject 1)".
6. 3.2: "the 16 gain curves correlate above 0.9995" holds in all 14 subjects (minimum 0.9996). Add the intermodulation
   caveat: 14% of the output power lies below 3 Hz, from the tanh encoder (subject 1).
7. **Remove limitation (vi)** (the non-identifiability sentence). It stays in `docs/NONIDENTIFIABILITY.md`.
8. **Add the authors' own ICA-cleaned run**, which the draft omits: nl_sync 0.502 (restricted 0.393, full 0.530); the surrogate
   contrast is +0.106 (14/14) restricted, +0.163 (14/14) full, **+0.098 (13/14) cleaned**. Consequence: delete or qualify the
   draft's 4.4 sentence "the phase effect strengthens relative to the low-passed condition (+0.163 vs +0.106)". Mention that the
   authors' audit found the discriminative 70-125 Hz activity at frontopolar and inion electrodes.
9. Under HELD, add these ICA numbers to both prepared outcomes and say "survives ICA cleaning" (ICA removes only part of the muscle
   signal).
- Acceptance: every number in the file appears in this brief, in `docs/REVIEW_BOARD.md`, or in a result file you can name.

**2b. `docs/NOTE_TO_COAUTHORS.md`.**
- "Three of the interpretations do not hold" becomes "Two of the interpretations do not hold, and a third is untested".
- Add one bullet under "What I found": their own ICA-cleaned run is in the repository but not in the draft; with it the
  phase contrast is +0.098 (13/14), the same as the restricted run (+0.106), so "strengthens with high gamma" does not hold.
- State the gain-curve and selectivity numbers as "in all 14 checkpoints".
- Keep the "AI-assisted analysis" sentence and the placeholders for names and date. Do not send anything.

**2c. `docs/NONIDENTIFIABILITY.md`, construction 3, and `scripts/nonident_check.py`, check 4.**
- Scope: for Gaussian stationary data the cross-spectrum fixes the joint law, so **no statistic** separates the worlds (not only
  second-order ones). Say that real EEG is non-Gaussian and non-stationary, which is where identification could come from.
- Say it is the class-dependent version of the common-input confound of construction 1.
- Add the side conditions: lag case needs g1 + g2 <= 1 (g1 + g2 = g cos(phi_c - mid) / cos(half-separation)); with class-dependent powers
  one shared generator works iff g_c max(r_c, 1/r_c) <= 1 with r_c = sqrt(p_b / p_a) / rho. Add numeric checks of both to the
  script (infeasible case g = 0.95 at the midpoint lag must report infeasible) and to `tests/test_nonident_check.py`.

**2d. `docs/TODO.md`:** replace "it cannot be done" by "it cannot be done for Gaussian stationary data without further
assumptions"; add the new items (nonlinear null result, `repro_lp48_decim4`, DEVIATION 31, notebook-version question).

## Step 3: close
1. Full test run: `.venv/bin/python -m pytest -q tests` (about 5-12 min); all pass.
2. One commit per step (scripts and notebook; documents), authored as tomgotchi, no co-author lines; push to `main`.
3. Append a short "Fixed" section to `docs/REVIEW_BOARD.md` listing what was done and what remains pending (the null test, the
   Colab run, DEVIATION 31's real-data run, the notebook-version answer).
4. Tell Tom: what is done, what is waiting on his Colab, and that the redline and the note are drafts that are still his to send.

## Numbers you may use (all re-computed on 2026-11-17)
| item | value | source |
|---|---|---|
| trained frequencies | 0.36-1.55 Hz; half-bandwidth 0.16-0.41 Hz (14/14) | `results/published_forensics.json` |
| gain-curve correlation (min) | 0.9996 in all 14 | board check |
| Hermitian part, largest eigenvalue | -0.258 to -0.377 (14/14) | board check |
| surrogate pairs with relative shift < 0.1 / 0.5 / 1 s | 6 / 25 / 53 of 120 | board check |
| oscillator e-folding time | 0.41-0.87 s | board check |
| nl_sync - nl_power | -0.004, 95% CI [-0.052, +0.045]; TOST +/-0.05 p = 0.031 | `results_hg/ablation_results.json` |
| nl_sync - lin_sync | +0.002, 95% CI [-0.041, +0.044] | same |
| accuracies nl_sync restricted / full / ICA-cleaned | 0.393 / 0.530 / 0.502 | `RESULTS_n14_COMPLETE`, `results_hg`, `results_nonemg` |
| nl_sync - nl_surrogate restricted / full / ICA-cleaned | +0.106 / +0.163 / +0.098 | same |
| decimator at decim 4, rms kept | 70 Hz 0.054, 90 Hz 0.018, 110 Hz 0.013 | board check |
