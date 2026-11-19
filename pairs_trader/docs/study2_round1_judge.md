# Study 2, round 1: judge's verdicts and the v2 proposal (2026-11-19)

## Verdicts on round-1 claims
- V = verified (I reproduced it). A = accepted (reasoned, not reproduced). P = pending.

| member | claim | verdict | evidence |
|---|---|---|---|
| Zhao C1, Volkov C2 (converged) | A post-BH sd screen enriches false discoveries | V | Re-ran Zhao's sel.py. At sd0 = 0.3%, R≥3 kept 7 of 81 pairs, and all 7 were null (8.6% null before). Null EG passers have median formation sd 6.2% against 0.3% for the true pairs. Volkov, separately: 0% → 5.7% null among pairs that pass the screen |
| Zhao | σ_e and the level sd carry the same information (corr 0.996) | V | sel.py output |
| Zhao M3 | The screen must be applied inside the calibration bootstrap | A | logic. calibrate.py conditions only on p |
| Zhao M1, Volkov m5 | The closed-form capture is too high (window end, estimated parameters, delay φ·z) | A | Zhao's ou.py ratios 0.78–1.05. Coint pairs realised/predicted 0.71–0.81 in sel.py output: V |
| Zhao M2 | z_scale in the capture term rewards pairs that calibrate badly | A | derivation |
| Volkov C1 | The simulator has one global spread_sd = 0.03 → the screen cannot fail | V | simulate.py:27,79 |
| Volkov M3 | β estimation drift ∝ √hl, independent of width; structural β drift untested | A | derivation |
| Volkov M4 | The innovation CUSUM misses level steps when hl > ~22 | A | derivation + simulation table. Out of scope for the one-change study; recorded as pending |
| Halvorsen C1 | "Holdout SR > 0" passes a null 50% of the time; power 0.22 at SR 0.5 | V | Φ(0) = 0.5; Φ(0.5/0.58 − 1.645) = 0.22 |
| Halvorsen C2 | The DSR gate needs n_trials ≥ 2 per universe label → fails automatically on new universes | V | cli.py:54, ledger.py; sp500_dev n_trials = 1 |
| Halvorsen M1 | The 6-block bootstrap is anti-conservative on the holdout | A | their simulation (7.2% vs 2.5%) |
| Halvorsen M2, M3, M4, m1, m2 | INCONCLUSIVE floor; R sensitivity on dev only; dev = veto only; minimum economic size; freeze hygiene | A | — |
| Novak C1 | New config fields change the Study 1 hash | V (logic) | config.py hashes the whole asdict |
| Novak C2 | Calibration seed truncated to 8 bytes | V | strategy.py:99, `b'RWR|VNQ|'` |
| Novak M1 | The stress test would change selection if the screen reads cost_bps | A | — |
| Novak M2 | random_pair_picker would skip the screen | A | — |
| Novak M3 | Ledger data_sha256 is the universe label; runs/ untracked | A | ledger shows "etf-dev" |
| Novak M4 | A book holding only the B leg is never flattened | V | paper.py:227 `have = sign(held[p.a])` |
| Novak M5 | Order ids collide across studies | A | — |
| Asante C1 | Alpaca fractional/notional orders: DAY only, no short → integer shares | A (*unverified* vs docs) | to check against the docs before coding |
| Asante C2 | A missed day fills a stale decision | V (logic) | paper.py fills every open order at the next run |
| Asante C3, M1–M4, m1–m2 | Non-atomic legs; 15:20 ET MOC loop driven by the calendar; paper fills ≠ evidence; broker as source of truth; costs; heartbeat; dashboard | A | — |

## v2 proposal (judge) for cross-examination
**The one strategy change is an *independent pre-BH* cost filter.**
- Candidates whose spread cannot pay for its own trading are removed *before* any test.
  - Statistic: the daily innovation sd of the formation OLS spread, σ_Δ = sd(Δs).
  - This statistic is a scale statistic. The EG t is scale-free, so under a Gaussian null σ_Δ is approximately independent of t (a Bourgon-style independent filter).
- Keep a candidate iff σ_Δ ≥ m × 2·c·(1+β̂), with c = 5 bp fixed (`screen_cost_bps`) and m = 1 a priori.
  - In words: one typical day's spread move must be at least the round-trip cost per dollar in leg A.
  - ETF clones (σ_Δ of a few bp against 20 bp) are excluded. Most stock pairs pass.
  - BH then runs on the filtered family, per group.
- The same filter is applied to every bootstrap copy in calibrate.py.
- The placebo uses the same filter.
- Nothing after BH changes. The model-based expected net P&L per pair is reported as a diagnostic only, not a screen (so that this stays one change).
- Sensitivity runs at m ∈ {0.5, 2} on dev only, counted in N, never on a holdout.

**Validation**
1. Simulator extension (Volkov C1): log-uniform per-pair σ, a clone kind, sd growing with hl, drifting β, level steps.
   - Pre-registered pass: the false-discovery share among traded pairs is ≤ the share without the filter, within MC error.
   - The filter removes ≥ 90% of clones.
2. Dev ETF and S&P 2010–2023-10: veto only.
3. Confirmation: the S&P 500 holdout 2023-11 → now (unopened) AND a never-touched universe, frozen and hashed first.
   - Halvorsen's three-outcome rule at one-sided α = 0.05 on the pooled holdout.
   - A per-trade sign-flip test as the primary test.
   - INCONCLUSIVE if < 60 pooled holdout round trips (< 150 on dev).
   - Minimum economic size: net return ≥ 2%/yr on gross capital.
4. Global N = distinct config hashes ever run on real data.

**Engineering (not part of the strategy test, but frozen before data):**
- Integer-share sizing in both the backtest and paper. A pair is skipped if the rounding error on the hedge exceeds 2%.
- Fixes for Novak C2 (seed), M4 (one-legged book) and M5 (order id salt). Asante C2: orders expire at t+1.
- Broker interface with SimBroker and an Alpaca paper adapter; reconciliation stops on any mismatch.
- Daily loop driven by the exchange calendar, plus a heartbeat.
- Study 1 hash pinned by a test.
