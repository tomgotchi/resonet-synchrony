# Draft note to the co-authors (for Tom to edit and send)

**To:** [co-authors]  **From:** Tom  **Date:** [ ]
**Subject:** Problems with the interpretation in the ResoNet paper: please read before the review goes further

Hi all,

I re-checked the paper's model against its own checkpoints and against simulated data with known ground truth. The
accuracies and statistics reproduce. **Two of the interpretations do not hold, and a third is untested.** The checks were run with
AI-assisted analysis (Claude) and then reviewed in a structured critique round. Every number below comes from a
script you can rerun.

## What I found
1. **Nothing resonates with the EEG.**
   - In all 14 trained checkpoints the oscillators sit at 0.36-1.55 Hz (half-bandwidth 0.16-0.41 Hz). The data are
     band-passed to 4-125 Hz.
   - The 16 gain curves are effectively one low-pass filter (correlation above 0.9995 in all 14 checkpoints). A
     resonator at 10 Hz would pass +10 Hz 36 dB above -10 Hz; these pass it 1.9 dB above, in all 14.
   - "Resonance", "Hopf oscillator" and "PLV" therefore describe the equations, not what the trained model does.
2. **The surrogate control does not isolate phase.**
   - I simulated data in which the classes differ only in source power, with no phase information. The published
     code (its linear bank) still gives sync - surrogate = **+0.31** (95% CI +0.27 to +0.35; 10 of 10 datasets). The
     same test on the nonlinear network, which produced the headline numbers, is running now; I will send it.
   - Source power reaches the readout through the cross-terms between learned spatial projections, and the surrogate
     disturbs those terms. So the +0.163 cannot be attributed to phase. The size of +0.31 depends on my simulation and
     should not be compared with +0.163.
   - Our own ICA-cleaned full-band run (`results_nonemg/`) is in the repository but not in the draft. With it the
     contrast is +0.098 (13 of 14 subjects), against +0.106 in the restricted run and +0.163 in the full run. So "the
     phase effect strengthens with high gamma" does not hold after cleaning.
   - It does **not** show that the high-gamma effect is a power effect.
3. **The high-gamma comparison changed two things at once (untested, not refuted).** The restricted run used decim 4
   (125 Hz), the full run decim 2 (250 Hz). The oscillator frequencies in Hz depend on the sampling rate. For
   reference, nl_sync is 0.393 restricted, 0.530 full and 0.502 on the ICA-cleaned full band.

## What stands
The decoding accuracies, the CNN baselines, the per-subject statistics and the reproducibility. Also that nonlinear
coupling adds no detectable gain (nl - lin sync +0.002, 95% CI -0.041 to +0.044; on subject 1's checkpoint the
nonlinear model's sync features correlate 0.994 with those of an uncoupled linear bank).

## What I suggest
- **Do not let the review proceed on the current claims.**
- Replace "phase synchronization carries information beyond band power" with "a fixed-lag surrogate that removes the
  cross-terms between learned spatial projections costs 0.15-0.16 accuracy; the control cannot separate phase from
  power".
- Report the oscillator frequencies in Hz. Drop "resonance" and "tuned resonators".
- Hold the high-gamma claim until the missing cell is run: the model on data low-passed at 48 Hz with 250 Hz
  sampling. (A full-band run at 125 Hz is also made, but decimation removes everything above about 60 Hz there.) The script is ready and I am running it on the Colab GPU
  now. I will send the result with a rule for reading it that was written before the run.
- Then decide together: correct the interpretation, or withdraw and resubmit. How to approach the editor is our
  decision as authors.

## To check it yourselves
`scripts/published_forensics.py` runs in about 10 seconds on the published checkpoints (section 4a of the Colab
notebook runs it on your own copy). `scripts/published_null_test.py` is
the simulation. Both are in `tomgotchi/resonet-synchrony`; the full account is `docs/CRITIQUE_AND_V2.md` (Parts 1-2).
I can give you access or send the outputs. Can we speak this week?

Tom
