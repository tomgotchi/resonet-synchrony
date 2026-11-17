# Draft note to the co-authors (for Tom to edit and send)

**To:** [co-authors]  **From:** Tom  **Date:** [ ]
**Subject:** Problems with the interpretation in the ResoNet paper: please read before the review goes further

Hi all,

I re-checked the paper's model against its own checkpoints and against simulated data with known ground truth. The
accuracies and statistics reproduce. **Three of the interpretations do not hold.** The checks were run with
AI-assisted analysis (Claude) and then reviewed in a structured critique round. Every number below comes from a
script you can rerun.

## What I found
1. **Nothing resonates with the EEG.**
   - In all 14 trained checkpoints the oscillators sit at 0.36-1.55 Hz (half-bandwidth 0.16-0.41 Hz). The data are
     band-passed to 4-125 Hz.
   - The 16 gain curves are effectively one low-pass filter (correlation above 0.9995).
   - "Resonance", "Hopf oscillator" and "PLV" therefore describe the equations, not what the trained model does.
2. **The surrogate control does not isolate phase.**
   - I simulated data in which the classes differ only in source power, with no phase information. The published
     code still gives sync - surrogate = **+0.31** (95% CI +0.27 to +0.35; 10 of 10 datasets).
   - The surrogate removes the cross-terms between learned spatial projections, and source power reaches the sensors
     through those terms. So the +0.163 shows the control cannot tell phase from power.
   - It does **not** show that the high-gamma effect is a power effect.
3. **The high-gamma comparison changed two things at once.** The restricted run used decim 4 (125 Hz), the full run
   decim 2 (250 Hz). The oscillator frequencies in Hz depend on the sampling rate.

## What stands
The decoding accuracies, the CNN baselines, the per-subject statistics and the reproducibility. Also "nonlinear
coupling adds nothing" (on subject 1's checkpoint the nonlinear model's sync features correlate 0.994 with those of
an uncoupled linear bank).

## What I suggest
- **Do not let the review proceed on the current claims.**
- Replace "phase synchronization carries information beyond band power" with "a fixed-lag surrogate that removes the
  cross-terms between learned spatial projections costs 0.15-0.16 accuracy; the control cannot separate phase from
  power".
- Report the oscillator frequencies in Hz. Drop "resonance" and "tuned resonators".
- Hold the high-gamma claim until the missing cells are run: the model on data low-passed at 48 Hz with 250 Hz
  sampling, and on the full band with 125 Hz sampling. The script is ready and I am running it on the Colab GPU
  now. I will send the result with a rule for reading it that was written before the run.
- Then decide together: correct the interpretation, or withdraw and resubmit. How to approach the editor is our
  decision as authors.

## To check it yourselves
`scripts/published_forensics.py` runs in about 10 seconds on the published checkpoints (section 4a of the Colab
notebook runs it on your own copy). `scripts/published_null_test.py` is
the simulation. Both are in `tomgotchi/resonet-synchrony`; the full account is `docs/CRITIQUE_AND_V2.md` (Parts 1-2).
I can give you access or send the outputs. Can we speak this week?

Tom
