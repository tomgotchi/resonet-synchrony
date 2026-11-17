# To-do (2026-11-17)

**The Colab to run:**
https://colab.research.google.com/github/tomgotchi/resonet-synchrony/blob/main/notebooks/resonet_synchrony_full_colab.ipynb

The repository is private. On first use Colab asks to authorise GitHub. If the link says "not found", use *File > Open
notebook > GitHub* or download `notebooks/resonet_synchrony_full_colab.ipynb` from the repository and upload it.

## Tom
1. **Run the Colab** (the manuscript's missing control, plus the hc-5 data-quality check)
   - [ ] *Runtime > Change runtime type > GPU* (T4 is enough).
   - [ ] The Colab secret `GH_TOKEN` is set (key icon, *Notebook access* on): a fine-grained token with Contents
         read/write on `tomgotchi/resonet-synchrony` only. Never paste it into a cell or into chat.
   - [ ] About 5 GB free on Drive: the 48 Hz copy of the cache is written to `Documents/RESEARCH/subj_cache_lp48`.
   - [ ] *Runtime > Run all*. Order: tests, **section 4a** (10 seconds: the trained oscillator frequencies in Hz from your
         own checkpoints), **pipeline check** (a published cell rerun on subjects 1-2), low-passed copy, the two new
         cells, hc-5 QC, figures, push.
   - [ ] Section 4a is the evidence for the co-authors: it should print 0.36-1.55 Hz for every subject. Save its output.
   - [ ] If Colab disconnects, run all again. Finished subjects and arms are skipped.
   - [ ] Tell me when the push has finished.
   - Not timed on a GPU: 168 trainings (6 arms x 14 subjects x 2 cells) plus 12 for the check. The two oscillator arms
     with the nonlinear model are the slow ones. Watch the log for the first subject. If it is too slow, set
     `SUBJECTS = '1-4'` (then `'5-9'`, `'10-14'`) in section 2.
2. **Tell the co-authors** about the published-model findings: Parts 1-2 of `docs/CRITIQUE_AND_V2.md`. Whether to
   correct the interpretation or withdraw is their decision, together with how to approach the editor.
3. **After the control:** decide with them what the paper says about high gamma (it depends on the result).
4. **Older questions, still open:**
   - [ ] the 10 Jul or the 13 Jul run as the reference;
   - [ ] the raw hc-2 files;
   - [ ] the closed-loop target.

## Added by the review board of 2026-11-17
- [ ] **Tom:** does the Colab copy you are running have a section 4a? If not, it is an older copy: the control's JSONs stay on
      Drive (`ResonanceNetwork_OFFICIAL/resonet_synchrony_runs/bandwidth_control/`) and are not pushed. Then either re-run
      section 9 from the current notebook, or download that folder and put it in `results/bandwidth_control/`.
- [ ] **Tom, next Colab session (current notebook):** the second pipeline check `repro_lp48_decim4` runs by itself in 4b. For
      DEVIATION 31 set `RUN_SURROGATE_RETENTION = True` (section 4c; GPU, no training).
- [ ] Nonlinear null test (DEVIATION 30): running on CPU here; read only when complete.
- [ ] Read the control with `scripts/bandwidth_analysis.py` (DEVIATION 29, addendum 1) once the Colab run has finished.

## Claude (no input needed)
- [ ] Analysis script that applies the reading rule of DEVIATION 29 to the four cells and the pipeline check.
- [ ] Read the hc-5 QC reports when they arrive (Gate 0: probe geometry and laminar coverage).
- [ ] Draft the note to the co-authors, if you want it.
- [ ] Artefact control on artefact-cleaned data (the non-EMG cache); asked for by the board, only partly covered by the
      central-sensor control.
- [ ] Board pass on construction 3 of `docs/NONIDENTIFIABILITY.md`, which it has not seen.

## Decided: not doing
- A confirmatory v2 lag test. The HGD test recordings and subjects 7-14 stay untouched for any v2 work.
- Phase 1 (stopped at 6 of 14 subjects, DEVIATION 28). Its switches in the notebook are off.
- A test of whether coupling changes with the class: it cannot be done for Gaussian stationary data without further
  assumptions (`docs/NONIDENTIFIABILITY.md`, construction 3).
