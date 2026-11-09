# ResoNet: a critique from its own artefacts, and a redesign (v2)

2026-11-09. Independent second look at the published Resonance Network (`reso-net/`, manuscript under review)
and at this follow-up repository. Every number below comes from a script in this repo, run on the published
checkpoints, the published code or synthetic data with known ground truth. No HGD recording was used.

## Bottom line
1. **The published model does not resonate with the EEG.** In all 14 published checkpoints the trained
   oscillators sit at 0.36-1.55 Hz. The data are band-passed 4-125 Hz.
2. **Its "phase synchronization" readout is a second-order (covariance-type) feature of learned spatial
   projections,** not a synchrony between brain rhythms.
3. **The surrogate control cannot separate phase from power.** It removes exactly the cross-terms through
   which source *power* shows at the sensors.
4. **The reported high-gamma effect is confounded with the sampling rate.**
5. The results themselves are reproducible and carefully evaluated; the *interpretation* (resonance, phase
   synchronization, high gamma) is what does not hold. **ResoNet v2** (`resonet_sync/v2.py`) is a redesign in
   which the phase question becomes testable.

## Part 1. What the published model computes

### 1.1 Nothing resonates
`scripts/published_forensics.py` reads `reso-net/results_hg/RNsync_S*.pt`.

| | frequency | half-bandwidth |
|---|---|---|
| initialisation (`omega ~ U(0.5, 4)`, `h = 0.01`, 250 Hz) | 0.20-1.59 Hz | 0.16-0.40 Hz |
| after training, all 14 subjects | **0.36-1.55 Hz** | 0.16-0.41 Hz |
| data (cache) | **4-125 Hz** | |

The oscillator frequency is `omega * h` radians *per sample*, so its value in Hz depends on the sampling rate.
At 250 Hz every oscillator lies below the 4 Hz high-pass, and training does not move it. The code comment says
as much: "the abstract omega timescale does not align with literal EEG-band Hz".

### 1.2 Consequences for the trained model (subject 1; the others are alike)
- **One filter, not a bank.** Over 4-124 Hz the 16 gain curves correlate above 0.9995. Each oscillator is the
  same low-pass with a 1/f slope: +22.8 dB at 4 Hz, +13.1 at 10 Hz, +3.0 at 30 Hz, -5.5 at 100 Hz.
- **Almost all of the output is 4-8 Hz.** For a 1/f input, 82% of the output power comes from 4-8 Hz, 94% from
  below 13 Hz and 1.2% from above 30 Hz.
- **The 16 frequencies hardly matter.** Give all oscillators one shared frequency and damping: the sync features
  correlate 0.98 with the original (median 0.82 per feature across trials).
- **The "phase" is not an oscillation phase.**
  - A resonating complex oscillator passes +f and rejects -f (36 dB for a 10 Hz resonator of this bandwidth).
    Here the difference is 4.9 dB at 4 Hz and 0.6 dB at 30 Hz.
  - The oscillator state rotates *backwards* in 42% of samples (median instantaneous frequency 2 Hz, interquartile
    range -6 to +11 Hz).
  - What the phase is: each oscillator's state is a low-passed complex number whose real and imaginary parts are
    two different spatial projections of the EEG (the complex encoder). Its angle is the ratio of those two
    projections.

So `C_ij = <u_i conj(u_j)>` is an amplitude-normalised cross-moment between pairs of learned spatial projections
of low-passed EEG. That is the information class that CSP and Riemannian (covariance) decoders use. It is how
**source power** appears at the sensors: a source that gets stronger changes the covariance between every pair of
projections that see it.

### 1.3 The surrogate control does not isolate phase
The paper's central contrast is `sync - sync_surrogate` (+0.163, 14/14, nonlinear bank). The surrogate rolls
each oscillator's series by a fixed lag (74-1056 samples). Because the states are broadband, any such lag removes
all cross-terms between projections. So the contrast shows that *cross-covariance between projections is
informative*, which is true whenever classes differ in source power.

Test on synthetic data with known ground truth: PENDING (filled in below, section 1.6).

### 1.4 The bandwidth finding changed two things at once
The paper reads "restricted vs full band" as the effect of high gamma ("the same experiment ... changing only the
input frequency band"). But:
- the restricted run used `decim = 4` (125 Hz) and the full-band run `decim = 2` (250 Hz)
  (`reso-net/results/ablation_summary.json`, `results_hg/ablation_summary.json`);
- because oscillator parameters are per sample, halving the sample interval also halves every oscillator
  frequency in Hz (0.1-0.8 Hz -> 0.2-1.6 Hz), halves every time constant in seconds and doubles the samples per
  trial;
- the trained oscillators pass about 1% of their output power from above 30 Hz (1.2).

The +0.12 to +0.17 gain is real, but it cannot be attributed to the 70-125 Hz band. The missing control is cheap:
low-pass the full cache at 48 Hz and keep `decim = 2`.

### 1.5 What stands
- The evaluation is careful: leakage-safe splits, per-subject paired statistics, honest baselines
  (ShallowConvNet 0.749 against 0.53 for the RN arms).
- "Linear is as good as nonlinear": the coupled Hopf dynamics do not earn their cost. This repo's later checks
  agree (the bank is a damped linear filter with weak crosstalk).
- The instinct to control for readout capacity is right. The control chosen does not do it (1.3, and it adds a
  dilution penalty: DEVIATION 24).

### 1.6 The published code on data with known ground truth
PENDING

## Part 2. What this means for the manuscript
These are the authors' decisions. My reading of what the evidence supports:
- **Finding 1** ("phase synchronization carries capacity-controlled class information beyond band power") is not
  supported as worded. Supported: *pairwise second-order features of learned spatial projections carry class
  information that per-projection power does not; a fixed-lag surrogate that removes them costs 0.10-0.16.*
- **"Resonance", "Hopf oscillator", "PLV"** describe the equations, not the trained model's behaviour on this
  data. State the oscillator frequencies in Hz.
- **Finding 2** (high gamma) needs the same-sampling-rate control before it can be claimed.
- The checks are quick to reproduce (`scripts/published_forensics.py` takes seconds). I would raise them with the
  co-authors before the review goes further.

## Part 3. Critique of this follow-up repository
- **It inherited the flaw it set out to fix, in milder form.** The Hz-parameterised bank put the frequencies in
  the band (1-40 Hz) but kept the published 0.16-0.40 Hz bandwidths. Sixteen such lines cover a few percent of the
  band, six of them lie below the cache's 4 Hz high-pass, and each oscillator still has its own frequency *and* its
  own complex spatial filter. So "coherence between oscillators" still mixes a frequency difference with a spatial
  difference.
- **Phase 1 replicates a model that is a weak decoder by construction,** and DEVIATION 24 already established
  that its contrasts cannot be called phase. The remaining GPU run will mostly confirm that.
- **Process outgrew results.** 26 deviations, 11 synthetic screens and three review-board rounds went mainly
  into a coupling-detection rule that could not work (the formal reason came late:
  `docs/NONIDENTIFIABILITY.md`) and into an hc-5 plan whose probes very likely lack laminar coverage. The
  controls were sound; the order was wrong. The checks in Part 1 take minutes and should have come first.
- **A fidelity gap in the synthetic data.** The calibration generators add white sensor noise down to 0 Hz. The
  real cache has no power below 4 Hz. This matters for any bank with oscillators below 4 Hz (the published one
  entirely; six of the Hz bank's sixteen). `scripts/published_null_test.py` high-passes at 4 Hz for that reason.

## Part 4. ResoNet v2
PENDING
