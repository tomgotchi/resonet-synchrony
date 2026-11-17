# Proposed changes to `PAPER_DRAFT.md` for issues 1 and 2 (draft for the authors; the manuscript itself is not edited)

Issue 1: the oscillators do not resonate with the data. Issue 2: the surrogate contrast cannot separate phase from
power. **Neither depends on the Colab result.** Issue 3 (high gamma, band against sampling rate) is **held**, and its
passages are marked HELD. Evidence for each change: `docs/CRITIQUE_AND_V2.md` Part 1; `results/published_forensics.json`;
`results/published_null/results.json`. One number is still open: the null test used the published *linear* bank. The
nonlinear network is being run (DEVIATION 30); replace "[+0.31, linear bank]" with that result when it is in.

## Terminology
| now | use instead | why |
|---|---|---|
| "resonance", "resonator", "tuned resonators" (as behaviour) | "driven, damped oscillator bank"; for the trained model "a shared low-pass filter" | trained frequencies 0.36-1.55 Hz against 4-125 Hz data; 16 gain curves correlate > 0.9995 |
| "phase synchronization", "phase-locking", "PLV" (as what the readout measures) | "cross-oscillator coherence terms" | the readout is a covariance-type feature of learned spatial projections of the sensors |
| "genuine", "airtight", "real EEG signal" | drop | the control cannot attribute the effect to phase |
| "named oscillators" / "oscillator pairs" as units of interpretation | "pairs of learned projections" | no oscillator is tuned to a band; they all act alike |
Keep the name "Resonance Network" only as a label for the architecture, with one sentence that it does not describe the
trained behaviour. Whether to rename it is the authors' decision.

## Title
Now: *Does Inter-Oscillator Phase Synchronization Decode Motor EEG? A Surrogate-Controlled Evaluation of Resonance
Networks on the High-Gamma Dataset.*
Option: *What does a surrogate control isolate? A critical evaluation of an oscillator-bank EEG decoder on the
High-Gamma Dataset.*

## Abstract, finding (1) (lines 20-25)
Now: "Phase synchronization carries genuine, capacity-controlled class information beyond band-power: destroying
inter-oscillator phase costs +0.163 accuracy in all 14 subjects ... the effect replicates in the linear bank, so it is
stimulus-induced, not emergent."

Proposed: "**Removing the cross-oscillator terms of the readout costs 0.163 accuracy in all 14 subjects** (Cohen's
d_z = 2.04, Holm-corrected p = 2.3e-5), an effect the raw sync-versus-power ablation hides because readout capacity
independently hurts. **The control cannot attribute this to phase**: on simulated data in which the classes differ
only in source power, the same contrast is +0.31 [linear bank; 10 of 10 datasets]. The linear, uncoupled bank
reproduces the effect (+0.150), so nonlinear coupling is not needed."

## Abstract, last sentence (lines 29-33)
Now: "...inter-oscillator phase synchronization is a real, capacity-controlled EEG signal that is nonetheless
band-power-equivalent as a decoder and stimulus-induced, and the value of oscillatory decoders is contingent on
retaining their native high-frequency band."

Proposed: "We conclude that a capacity-matched surrogate is necessary but not sufficient to isolate phase: it must also
be shown to stay at chance on data without phase information. The trained oscillators sit at 0.4-1.5 Hz, below the
4-125 Hz input, and act as a shared low-pass filter; the decoder is band-power-equivalent. **[HELD: the sentence about
high-frequency content.]**"

## Introduction, contributions 1-2 (lines 64-70)
Proposed 1: "A **capacity-matched surrogate control** (after Theiler et al., 1992) that separates the cross-oscillator
terms of a readout from readout capacity. It costs 0.163 accuracy (14/14 subjects). **We show that it also fires when no
phase information exists**, so it identifies cross-terms, not phase."
Proposed 2: "A **linear-versus-nonlinear ablation** (uncoupled FFT bank against coupled Hopf network): the linear bank
reproduces the effect, so nonlinear coupling adds nothing here."
Drop the evoked-versus-induced mapping (David et al., 2006) unless a separate test supports it.

## 3.2 The Resonance Network: add after the dynamics paragraph
"**Trained frequencies.** With h = 0.01 and fs = 250 Hz an oscillator's frequency in Hz is omega h fs / 2 pi. After
training, omega lies at 0.36-1.55 Hz and the half-bandwidth |mu| h fs / 2 pi at 0.16-0.41 Hz in all 14 subjects, while the
input is band-passed to 4-125 Hz. The oscillators do not resonate with the data. They act as a shared low-pass filter on
the encoder output (the 16 gain curves correlate above 0.9995), and for a 1/f input 82% of the output power lies at
4-8 Hz."
Correct the description of `surrogate` (lines 151-155): "...destroys inter-oscillator phase alignment while preserving
each oscillator's spectrum" becomes "...destroys **all** cross-oscillator structure, including the zero-lag covariance
through which differences in source power reach the readout, while preserving each oscillator's spectrum and the
240-dimensional feature space."

## 4.4 (title and text, lines 236-259)
Title: "High-gamma surrogate control: the cross-oscillator terms carry capacity-controlled information".
Replace "Destroying inter-oscillator phase ... costs +0.163" by "Removing the cross-oscillator terms (dimension held fixed) costs
+0.163"; replace "**PHASE**" and "PHASE (linear)" in the table by "CROSS-TERMS"; and replace the last sentence of the first
paragraph ("so it is stimulus-induced, not emergent") by "so nonlinear coupling is not needed". Add:
"**Calibration.** On simulated data without phase information (classes differ only in source power) the published code
gives sync 0.63, power 0.58, surrogate 0.32, so sync - surrogate = +0.31 (95% CI +0.27 to +0.35; 10/10) [linear bank].
The contrast therefore does not identify phase. It also does not show that the effect on real EEG is a power effect."

## 4.5 Interpretability (lines 261-273)
Keep: all oscillators learned mu < 0; output rhythms low-frequency. Add: "consistent with the trained frequencies of
0.4-1.5 Hz". Replace the "functional-connectivity fingerprint" sentence: "The `sync` readout is an explicit matrix of
cross-terms between 16 learned projections. Because every oscillator acts alike, the pairs are pairs of projections, not
of rhythms, and the per-class matrices (Figure 5) should not be read as connectivity between brain rhythms."
**HELD:** the sentence on the high-gamma envelope.

## 5 Discussion (lines 280-316)
1. Heading "Phase synchronization carries genuine, capacity-controlled class information beyond band-power ... airtight":
   replace by "**The cross-oscillator terms carry capacity-controlled class information, and the control cannot say
   that this is phase.** With the feature dimension held identical, removing them costs about 0.16 accuracy in every
   subject. The same contrast fires on data without phase information, so the effect may be power-related structure
   expressed through the cross-terms. Any study that compares coherence with band-power readouts needs a capacity-matched
   control *and* a world in which the control stays at chance."
2. "The effect is stimulus-induced, not emergent ... captured by tuned resonators": replace by "**Nonlinear coupling adds
   nothing.** The linear, uncoupled bank reproduces the effect, and the trained nonlinear network is contractive in
   all 14 subjects (its sync features correlate 0.994 with those of an uncoupled linear bank on subject 1). The
   cross-oscillator terms are therefore a property of the encoder and the input, not of coupling."
   Drop "tuned resonators" and "common-input (evoked-like)": not tested.
3. "On the black-box question": delete "the surrogate control provides causal validation that this interpretable
   quantity is genuinely used". Replace by "The surrogate shows that the cross-terms are used by the classifier, not
   that they are phase."
4. Limitation (iv), "partially, not fully, volume-conduction-robust": the `sync_imag` readout decodes power-only data at
   0.61 (published code, simulated data), so it is **not** robust here. Say so.
5. Add limitation (vi): "A change in coupling between classes cannot be told from a change in source amplitudes using
   second-order statistics of the sensors." (Optional: this rests on `docs/NONIDENTIFIABILITY.md`, construction 3, which
   the review board has not yet seen.)
6. "Defensible summary claim": "A dimension-matched surrogate that removes the cross-oscillator terms of a Hopf-oscillator
   bank's readout costs about 0.16 accuracy on HGD in all 14 subjects. The same contrast is +0.31 on simulated data
   with no phase information, so it cannot separate phase from source power. The trained oscillators sit at 0.4-1.5 Hz
   and act as a shared low-pass filter. Nonlinear coupling adds nothing over an uncoupled linear bank, and the decoder is
   band-power-equivalent. **[HELD: high-frequency band.]**"

## 6 Conclusion (lines 320-330)
Replace "we established that inter-oscillator phase synchronization carries genuine class information beyond band-power"
by "we showed that the cross-oscillator terms of the readout carry capacity-controlled class information, and that the
control cannot separate phase from power". Replace the methodological takeaway by: "**claims that 'synchronization
decodes X' need a capacity-matched control that is also shown to stay at chance when no synchronization is present.**"
**HELD:** the sentence on the high-gamma band.

## Keywords
Replace "phase synchronization, phase-locking value, Hopf oscillator" by "oscillator networks, surrogate control,
calibration".

## HELD until the band x sampling-rate control is read (DEVIATION 29)
Abstract finding (2); contribution 3; section 4.2-4.3; the high-gamma sentences of 4.5, 5 and 6. Two prepared outcomes:
- *Mostly band:* "Restoring 48-125 Hz content raises every oscillatory arm at both sampling rates. This does not show that
  the model uses it through resonance or phase."
- *Mostly sampling rate:* "The gain from the restricted to the full run is mostly due to the change in sampling rate,
  which also changes the oscillator frequencies in Hz. We do not attribute it to high gamma."
