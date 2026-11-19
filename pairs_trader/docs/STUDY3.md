# Study 3: pre-registration (frozen 2026-11-19, before any Study 3 run)

**Study 2 FAILED at gate 0, as pre-registered** (`results/study2_gate0.json`):
- P1–P4 passed. P5 failed: kept nulls rejected 2.2% of the time, filtered-out nulls 34.0%.
- The board (round 3, unanimous: Halvorsen, Zhao, Volkov) found that P5 was mis-specified. It compared two
  different null processes:
  - The filtered-out nulls were all `clone_null`: a random-walk tracking error of 1–4 bp/day plus 2 bp of iid bounce.
    That is an IMA(1,1) process with a large MA root, on which a one-lag ADF/EG is known to be over-sized
    (Schwert 1989). Judge check: 5.2% rejection at 0 bp bounce and 32% at 2 bp (`scripts/p5_bounce_check.py`).
  - The kept nulls were independent pairs.
  - P5 therefore did not test whether the filter depends on the test statistic. The result is consistent with the
    filter *removing* anti-conservative nulls.
- **Disclosed as a post-result deviation (Halvorsen r3).** P5′ below was written knowing that kept nulls reject at
  2.2%. Two things keep it informative:
  - it runs on **fresh seeds**;
  - it adds a null kind that straddles the threshold, so the independence check can still fail.
- **One re-specification only.** If gate 0 fails again, the line of work stops; there will be no Study 4 on the
  simulator.

## Unchanged from Study 2 (`docs/STUDY2.md`)
- The strategy is unchanged. The config hash **6ce6d3e4de60** (study 2, cost_filter_m 1, integer shares) is pinned.
- Data, run order, the development veto, the holdout rule, N and the real-money conditions are all unchanged.
- N is unchanged because no config has touched real data since Study 1.
- **Board flag, reported prominently:** P3's FDR was 0.113 against q = 0.10, passing only through the MC allowance.
  With the filter, the null share among selected pairs rose from 11.2% to 14.3%. This is the cost of removing the
  clones, which were true positives and kept the null share down (π0 effect). It is not dependence between the
  filter and the test.

## Gate 0′ (`scripts/validate_study3_sim.py`)
**Panels**
- 200 fresh panels with seeds 20261120 + i. No Study 2 seed is reused.
- Same panel structure as Study 2, plus 4 `straddle_null` pairs per sector:
  - β U(0.97, 1.03);
  - a random-walk tracking error with σ_e log-uniform 5–60 bp/day, so it falls on both sides of the threshold;
  - bounce drawn per pair from {0, 1, 2} bp. That draw is the stratum.
- `clone_null` with 2 bp bounce stays in as a **reported trap**.

**Gates**

| gate | requirement |
|---|---|
| P1–P4 | As in Study 2. |
| P5a | Kept-null size (Halvorsen, Volkov P5″, Zhao a). For each null kind k ∈ {null, cross, straddle_null} with n_k ≥ 1000 kept: the kept rejection rate (raw p < 0.05) is r_k ≤ 0.05 + 2·√(0.05·0.95/n_k). |
| P5b | Independence within one process (Volkov P5′, Zhao b). Within straddle_null, filter-pass and filter-fail rejection rates do not differ: the Cochran-Mantel-Haenszel test across the three bounce strata has p ≥ 0.05, with ≥ 1000 pairs in each arm overall. |

**Reported, not gated**
- The pooled pass − fail difference.
- Rejection rates by stratum and by decile of the filter statistic.
- clone_null rejection with and without the filter.
- P6 as in Study 2.

If any of P1–P4, P5a or P5b fails, stop and record the failure.
