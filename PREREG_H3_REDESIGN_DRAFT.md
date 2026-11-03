# H3 redesign: pre-registration DRAFT (NOT frozen; for Tom)

Phase 2 (synthetic H0) showed that the original H3 contrast, the nl_coupled - lin *decoder* advantage, cannot detect
emergent coupling. The stop rule was triggered and Phase 3 was not started. Screens 1-8 (DEVIATIONS 11-18) then
developed and confirmed, on synthetic data, a *mechanism* contrast that does separate coupling from unobserved
common input within a session. This draft proposes it as the replacement H3. **Nothing here is binding until Tom
edits it and commits it as `PREREG.md`, before any hc-5 wheel-vs-maze contrast is computed.**

## Proposed hypothesis
**H3' (primary):** within each hc-5 session, CA1 theta-band dependence is *more coupling-dominated during wheel
running* (internally generated sequences) than during maze running (sensory-driven). Operationally, the
within-session rule v2 (DEVIATIONS 18) fires for A = wheel and B = maze:
- (i) Delta S_TE = S_TE(wheel) - S_TE(maze) > 0 in >= 6/7 sessions, with a 95% bootstrap CI over sessions > 0;
- (ii) the relative change (S_inst_dn(wheel) - S_inst_dn(maze)) / S_inst_dn(maze) has an upper 95% CI < 0.25.

The session counts are adapted from the synthetic 9/10 to the 7 available sessions. **To decide:** 6/7 or 7/7.

**Interpretation is fixed in advance:**
- Firing means "the wheel regime shows more lagged inter-channel phase predictability *without* a matching rise in
  instantaneous (common-input-like) dependence".
- It does **not** mean "intrinsic dynamics". Terminology follows TASK_PITCH1 S9.
- Not firing because (i) fails is reported as **no evidence for more coupling**.
- Not firing because (ii) fails is reported as **an increase confounded by common input**. That is a distinct,
  informative outcome.

## Data and preprocessing (proposals; most are already implemented in `data_hc5.py` / `qc_hc5.py`)
- **Sessions:** the 7 TASK_PITCH1 sessions. Session is the unit, all from one animal, so this is a within-animal result.
- **Channels:**
  - >= 8 CA1 channels spread across shanks. They are chosen from the QC report on **maze data of the first half of
    each session only**, by theta/delta ratio and absence of flags.
  - The region map is confirmed per session. **Blocker:** `i01_maze08.001` reports 102 channels, and the documented
    map is for 129.
- **Referencing:** local bipolar within shank (QC shows the common reference inflates coherence).
- **Filtering:** 6-10 Hz for TE phases; the LDS sees broadband z-scored 250 Hz data (decimated zero-phase; offline,
  as allowed by S5).
- **Epochs:**
  - wheel = samples inside wheel runs (`Laps.NLapCW/CCW > 0`);
  - maze = inside maze laps (`Laps.startLfpInd`-`endLfpInd`), running only (speed above a train-split threshold);
  - both cut into **1 s windows** (the synthetic window).
- **Matching:**
  - the same number of windows per regime per session (random subsample with a fixed seed);
  - windows never span two runs or laps.

## Statistics
- S_TE and S_inst_dn are computed exactly as in DEVIATIONS 11/16, using code at the commit that freezes this
  document.
- Bootstrap over sessions (10,000 resamples). Wilcoxon signed-rank on Delta S_TE is reported alongside.
- **Within-session robustness:** split each regime's windows into halves and require the rule's direction to hold
  in the half-split replicate. **To decide:** whether this is a gate or descriptive.
- **Sensitivity:** the same analysis restricted to correct trials, and with the opposite hemisphere's CA1 if
  available.

## Known limits (state them in any report)
- The rule was validated only on synthetic noise-driven Stuart-Landau networks with linear diffusive coupling.
  Real CA1 may violate the model (non-oscillatory components, volume conduction after re-referencing,
  behaviour-locked evoked structure).
- The relative guard loses sensitivity with short or noisy windows (screen 8, S1 win05). A null result may be a
  sensitivity limit.
- Wheel and maze also differ in speed, sensory input and theta frequency. **Screen 9 (DEVIATIONS 19) showed that
  rule v2 is not safe against these:** +1 Hz theta in one regime, or +10 dB SNR, each triggered it with no coupling
  change (10/10 and 9/10 sessions). H3' must not use rule v2 as written. The candidate rule v3 (DEVIATIONS 20:
  TE delay in cycles of each regime's own f_c, plus LDS noise-fraction matching) is being tested on synthetic
  data. **To decide after screen 10:** adopt v3, or match f_c and SNR by window selection instead. Delta f_c and
  Delta R are reported either way.
- 7 sessions from one animal; no population-level claims.

## What must be true before freezing
1. The hc-5 QC notebook has run: channel counts, region map, and wheel/maze epoch counts per session.
2. Tom has decided the open points above.
3. The H1/H2 plans for hc-2/HGD are aligned (separate documents).
