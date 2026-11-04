# hc-5 generator-mix hypothesis: pre-registration DRAFT

**Status (2026-11-04).**
- **Adopted, not yet frozen.** Tom delegated the choice to the review board, which recommended this reframing
  (`docs/REVIEW_BOARD.md`, rounds 1-2). It replaces `PREREG_H3_REDESIGN_DRAFT.md` (H3', "coupling between CA1
  sites"), which is **superseded**.
- **Freezing order.** This document is frozen, unchanged or with recorded edits, only after Gate 0 and Gate 1
  below have passed. It is frozen **before any wheel-vs-maze contrast of the target statistic is computed**.

## Why the construct changed
LFP at a site is a lead field times a few shared current generators. Within ~1 mm of the CA1 pyramidal layer the
generators are common to all sites, so "coupling between sites" cannot be told apart from a change in the
generator mix (Echeverria). Two further results point the same way:
- Phase-based dependence statistics read rhythmic common input and volume conduction as coupling (Zhao, Volkov,
  Halvorsen; screen 11 tests this directly for rule v3r).
- The physiologically meaningful question is how strongly the **CA3-driven (stratum radiatum)** and **EC3-driven
  (stratum lacunosum-moleculare)** theta current sinks contribute during wheel running vs maze running.

## Hypothesis
**H-GM (primary, two-sided).** Within session, the theta-band current-source-density (CSD) power ratio between the
l-m sink and the radiatum sink differs between wheel running and maze running.
- **Statistic per session:** Delta = G(wheel) - G(maze), with G = log( P_CSD,theta(l-m) / P_CSD,theta(radiatum) ),
  using speed-matched 1 s windows (`resonet_sync/laminar.py: generator_mix`).
- **Secondary (descriptive):** Delta of the l-m minus radiatum theta CSD phase difference (deg).

**Interpretation is fixed in advance.**
- Delta > 0 means "relatively more EC3-layer theta current during the wheel".
- It is **not** "intrinsic dynamics", and not "coupling". Those words are not used.

## Gate 0: data feasibility (QC only; no wheel-vs-maze statistic)
`qc_hc5.py` reports `laminar_coverage` per anatomical group (shank).
- **Usable shank.** A shank is usable if its sites contain the pyramidal layer (ripple-power peak) **and** a theta
  phase reversal below it.
- **Span.** It must also span >= 300 um below the pyramidal layer.
  - The site spacing must come from the probe documentation, not from assumption. It is still to be confirmed for
    hc-5.
  - With about 20 um spacing and 8 sites per shank, no shank will qualify.
- **Hemisphere.** One hemisphere only: right CA1, unless it has no usable shank in a session, in which case left
  CA1 for that session. The choice is recorded per session from Gate 0 only.
- **Pass.** Gate 0 passes if at least 5 of the 7 sessions have a usable shank.
- **If Gate 0 fails, the CSD route is infeasible on hc-5.** That is reported as the result, and nothing is run
  instead. There is no fallback to coupling statistics; a different dataset with laminar coverage would be needed.

## Gate 1: synthetic forward-model null (before any real contrast)
`resonet_sync/laminar.py` simulates laminar LFP from two theta generators (CA3 -> radiatum, EC3 -> l-m), each
driven by its own narrowband current. The simulation uses **the real probe geometry from Gate 0** and the real
analysis path (layer localisation on neutral data, CSD, generator_mix).

**Must NOT move Delta** (|mean Delta| < 0.10 and an exact sign-flip test not significant over 10 simulated
sessions). Each nuisance is set to the wheel-vs-maze difference measured in QC (Gate 0 data), or to the values
below if larger:
- theta frequency +/-1 Hz;
- speed-locked frequency drift (random walk 0.5 Hz);
- waveform asymmetry (2f harmonic 0.4);
- amplitude x1.3;
- 1/f background exponent 1.0 vs 1.6, with noise 5% vs 15%;
- a travelling-wave lag of 10 deg per shank.

**Must move Delta.** A 20% change in EC3 drive must be detected in >= 8/10 simulated sessions.

Gate 1 passes only if both hold. If a nuisance moves Delta, that nuisance is matched by window selection before
freezing, and Gate 1 is re-run. Each re-run is logged in DEVIATIONS.

## Data and preprocessing
- **Sessions:** the 7 TASK_PITCH1 sessions. One animal, so the result is within-animal.
- **Signal:** monopolar `.eeg` at 1250 Hz. CSD is reference-free, so no bipolar or common-reference
  re-referencing is used.
- **Channel handling:** skip-flagged and TTL/sync channels are excluded.
- **Depth order:** taken from the xml group order, after verifying that the theta phase profile is monotonic
  near the reversal.
- **Layer and sink sites:**
  - chosen on **neutral data**: the first half of each session, pooling all behaviour (not one regime);
  - radiatum sink = the theta-CSD minimum between the pyramidal layer and the reversal;
  - l-m sink = the theta-CSD minimum below the reversal.
- **Epochs:**
  - wheel = inside wheel runs (`Laps.NLapCW/CCW > 0`; documented definition: a contiguous stretch);
  - maze = inside maze laps, running only.
- **Speed matching:**
  - wheel speed comes from the wheel encoder TTL channels (to be verified on one session in QC; if absent, the
    analysis is restricted to maze-speed-equivalent wheel periods and this is reported as a limitation);
  - maze speed comes from `Track.speed_MMsec`;
  - 1 s windows are histogram-matched on speed, with equal window counts per regime (random subsample, fixed
    seed);
  - no window spans two runs or laps.

## Statistics
- **Unit:** the session (n = 7).
- **Primary test:** exact two-sided sign-flip permutation on the 7 Delta values (128 sign patterns; minimum
  p = 0.0156). Report all 7 values, the mean, and the exact p.
- **Within-session robustness:** an odd/even-window half-split replicate. Reported, not a gate.
- **Negative controls** (must be null; reported alongside the primary result):
  1. maze first half vs maze second half;
  2. wheel CW vs wheel CCW runs;
  3. a within-session label shuffle of regime across windows (null distribution of Delta).
- **Covariates reported for every session:** Delta f_c, Delta waveform asymmetry, Delta theta amplitude, the speed
  distributions, and **EC layer-3 theta power and EC-CA1(l-m) theta coherence** (EC is recorded; observed
  covariate, descriptive).

## Forbidden before the freeze (board, Echeverria)
- Computing S_TE, S_inst_dn, rule v3r or any dependence statistic on wheel vs maze.
- Overlapping-bipolar or common-reference data in the CSD analysis.
- Pooling hemispheres.
- Channel or site selection that uses one regime.
- Window-level statistics as if windows were independent.
- Maze-only speed gating.
- The words "coupling" and "intrinsic" in any conclusion about H-GM.

## Known limits
- **One animal, 7 sessions:** no population claims.
- **CSD assumptions:** a 1-D CSD assumes laminar homogeneity and constant conductivity. Lateral generators and
  probe tilt bias the sink estimates.
- **Generator ascription:** the "CA3" and "EC3" labels are anatomical sink locations, not direct measurements of
  those inputs.
- **Statistical power:** the minimum achievable two-sided p is 0.0156, so the test has low power for small effects.
