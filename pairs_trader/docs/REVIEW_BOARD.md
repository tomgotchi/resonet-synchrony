# Review board record: pairs_trader

## Session 2026-11-18
- **Target**: design of an ML trading app for "two connected things used proportionally, like two stocks".
  - v0 read the request as binary event contracts; the user corrected it. The v0 reviews were stopped unread.
  - v1 design (pairs trading / statistical arbitrage); then the v1 code; then the real-data decisions.
- **Members**:
  - Halvorsen (statistician), Zhao (information theorist), Volkov (dynamicist), Asante (DSP/real-time) and
    Novak (code auditor).
  - Echeverría (neuroscientist) was not convened: his lane (biology) does not apply.
- **Mode**: agent mode, with independent parallel reviewers.
- **Rounds**:
  1. Design.
  2. Cross-examination of the verdicts plus an audit of the code.
  3. Real-data decisions, asked for by the user.
- **Judge**: the main assistant.

### 1. Findings and verdicts
Abbreviations used below:
- EG: the Engle-Granger cointegration test. MacKinnon EG uses the critical values specific to that test.
- ADF: the augmented Dickey-Fuller test.
- BH: Benjamini-Hochberg false-discovery control.
- hl: half-life of the spread's reversion. z: the standardised spread.
- CUSUM: the break detector. FDR: false discovery rate. DSR: deflated Sharpe ratio.

Verdicts: V = verified (the judge reproduced it), A = accepted (reasoned, not reproduced), R = rejected,
P = pending.

| sev | member | finding | verdict | evidence / action |
|---|---|---|---|---|
| crit | Halvorsen, Zhao, Novak | Naive ADF p-values on EG residuals over-reject | V | null pairs, n=252: 14.2% naive vs 5.4% MacKinnon EG. Fixed in `coint.py`; test `test_eg_null_false_positive_rate_is_nominal` |
| crit | Halvorsen, Novak | Hundreds of candidates → "top K" fills with noise | V | ~5% of null candidates pass. Fixed with BH |
| crit | Zhao, Volkov, Novak | EG depends on the regression direction | V | 8.1% disagreement. Now p = max of both directions; Zhao (round 2) showed this is a valid intersection-union test, 2.8% size |
| crit | Halvorsen, Zhao, Volkov | The half-life filter screens nothing | V | 54 of 54 null passes kept |
| crit | Volkov | AR(1) half-life biased fast | V | true 20 → median 14.4. Round 2 (Zhao): Kendall's correction is not enough on *selected* pairs → selection-aware bootstrap (`calibrate.py`) |
| crit | Zhao | The planned ML label "reverts before stops" is meaningless | V | true on 96.4% of random-walk trades, mean P&L ≈ 0. Label replaced with net P&L; ML off |
| crit | Novak, Volkov | Dollar-neutral 1:1 positions do not trade the tested spread | V (algebra) | legs now 1:β in dollars |
| crit | Novak | NaN comparisons silently freeze positions | V | forced exit, filled at the first real price after the gap (round 2, Asante) |
| crit | Asante | A fill at the signal bar's close fakes an edge | V | +7.5 bp/trade vs −2.2 bp at the next bar. One-bar delay; test |
| crit | Halvorsen | A Sharpe of 0.5–1 is what selection alone produces | A | trials ledger + DSR; config frozen |
| maj | Halvorsen | Years needed to tell a given Sharpe from zero | V | table reproduced exactly (SR 0.5: 12.2 years at 50% power, 27.8 at 80%) |
| maj | Volkov, Asante | A Kalman beta absorbs the spread | A | Kalman cut from v1 |
| crit (r2) | Zhao | The frozen formation z fires about 3× too often on genuine pairs | V | 12.5% of days vs nominal 4.6%; 4.1% after calibration, 3.7% on held-out seeds |
| crit (r2) | Volkov | CUSUM uncalibrated (14–31% false alarms per window) | V | 12.7% → 2.5% (3.8% on held-out seeds), from a Huber clip plus a per-pair bootstrap h |
| maj (r2) | Volkov | Unit-root breaks take about 100 days to detect | A | close to the information limit; the time stop and position caps are the protection |
| maj (r2) | Volkov | Simulator `beta_step` was a disguised level jump | A | fixed (slope change only) |
| maj (r2) | Novak | A crash between the log write and the state write loses orders | A (reported) | atomic state, queue-then-log; test `test_crash_between_state_and_log_loses_nothing` |
| maj (r2) | Novak | One missing held ticker freezes the whole book | A | pair frozen, healthy leg flattened; test |
| maj (r2) | Novak, Asante | Paper ≠ backtest (borrow, rounding, timing) | V | after the rebuild, paper P&L == backtest P&L over a full window (`test_paper_equals_backtest_over_a_full_window`) |
| maj (r2) | Asante | Dividend back-adjustment shifts paper z-scores | A | split-adjusted closes + dividends as cash; refit from the file every run |
| maj (r2) | Halvorsen | DSR collapses to PSR (ledger keyed on the file hash; variance → 0) | A | ledger keyed on universe + dates; distinct configs; variance floor of 1/years |
| min (r2) | Halvorsen | Placebo gate was 91%, not 95% | V (arithmetic) | 39 draws, no ties allowed: exactly 2.5% |
| maj (r2) | Zhao | Placebo too weak (random walks with a clipped half-life) | A | matched placebo (0.05 < p < 0.5, same screens and calibration) |
| min (r2) | Asante | Round trips filled and flattened on the same last bar | A | entries may not fill on the last bar; test |
| min (r2) | Asante | Delay 2 > delay 1 in the demo | A | small-sample noise (26 runs; delay 1 wins 13 of 16 when trades ≥ 25) |
| min (r2) | Novak | 20–32% of true 20-day pairs exceed the half-life cap of 30 | A | cap widened to 45 |
| judge | — | BH over *all* pairs finds nothing; within-sector BH finds true pairs | V | 4005 → 0 discoveries; 405 → 4, all true. **Round 2 overturned the strong form** (below) |

### 2. Overturned verdicts
- **"Sectors make selection work"** (judge, round 1) → **overturned** by Halvorsen and Novak in round 2.
  - The simulator gives every planted pair a perfect sector label.
  - Real sectors hold 50–100 names, and at m ≈ 1,225 BH finds only 0.36 of 5 true pairs.
  - Response in round 3: GICS *sub-industry* groups and per-group BH at q = 0.10.
- **"The Kendall correction fixes the half-life"** (judge, round 1) → **overturned** by Zhao in round 2.
  - On selected pairs the corrected half-life is still about 33% too fast.
  - Response: the selection-aware bootstrap.

### 3. Fixed / pending
- **Fixed**: everything marked V or A in the table, with regression tests (41 tests).
- **Pending**:
  - No point-in-time constituents or delisting returns (free data cannot provide them). Stock results are labelled
    a survivorship-biased upper bound.
  - Factor-neutral attribution (is the edge just short-term reversal?).
  - A graded power map and a 50-seed false-pass map of the gates (Halvorsen's top round-2 item; ~2.5 h offline).
  - Probability of backtest overfitting (PBO, Halvorsen).
  - An earnings blackout (Zhao).
  - A realistic jump model with 4 earnings jumps a year and correlated components (Volkov).
  - Opening-auction fills, borrow availability, an exchange calendar (Asante).
  - Re-hedging drift for β far from 1 (Volkov).

### 4. Round 3 decisions (asked by the user to be made by the board)
| decision | board recommendation | adopted |
|---|---|---|
| universe | ETFs primary; current S&P 500 secondary, labelled survivorship-biased (Halvorsen) | yes |
| grouping | GICS sub-industry for stocks; frozen asset-class groups for ETFs (Halvorsen) | yes (`data.ETF_GROUPS`) |
| selection power | BH per group at q = 0.10; K cap kept but not forced to fill (Halvorsen) | yes |
| prices | split-adjusted closes + dividends as cash (Asante) | yes |
| z calibration | predictive sd is insufficient → selection-aware bootstrap for both half-life and entry z (Zhao) | yes |
| holdout | develop to 2023-10-31; open 2023-11-01 onward once; pass rule written beforehand (Halvorsen) | yes, see DESIGN pre-registration |
| ML ("train the models") | **not run in v1** (Halvorsen, Zhao: few trades, survivorship, no information beyond the prices). If ever: the sector ETF 5-day return as the only external feature | yes: not trained |
| paper trial | replay the holdout through the paper code first, then forward (Asante) | yes |
| interface | read results, run the paper step, set the kill switch; nothing else (Asante) | yes (`app.py`) |

### 5. Positions
- **Halvorsen**: freeze everything before data, gate on DSR with an honest N, build the power map; a 3-year holdout
  can only falsify.
- **Zhao**: the entry signal must be calibrated on selected pairs, and the edge must beat matched placebos;
  ML adds nothing in v1.
- **Volkov**: the break detector is now calibrated, but unit-root breaks are slow to see; realistic earnings jumps
  may leave no cointegrated stock pairs at all.
- **Asante**: paper must be the backtest, run daily; raw-plus-dividend prices; the UI must not become a tuning knob.
- **Novak**: crash-safe and unblockable paper runs, and tests that prove what they claim.

### 6. Judge's plan
1. Run the pre-registered development backtests (ETF, S&P 500). Commit the reports before opening the holdout.
2. Open the holdout once and apply the pass rule.
3. Paper trial: replay the holdout day by day through `run_paper`; then the user may run it forward daily from the
   dashboard.
4. No real money unless all gates and the holdout pass, and a broker integration has been reviewed separately.

### 7. Round 4 (after the pre-registered real-data run), 2026-11-18
**Results**
- **ETF, development period 2012-01 to 2023-10:**
  - Net Sharpe −0.19, 95% CI −0.78 to 0.33. 73 trades; 38% made money net.
  - All four gates fail.
  - Matched placebos: median about 0, maximum 0.25.
  - Sharpe at 2× costs: −0.48.
- **ETF, holdout 2023-11 to 2026-11 (opened once):** Sharpe −0.36, CI −1.16 to 0.44, on 5 trades. The pass rule fails.
- **Paper trial (no money):** a replay of all 765 holdout days through `run_paper`. It matched the backtest to the
  cent in every window: −128.26 and −18.61 in the two windows that traded. The paper code path is the backtest.

**Members' positions**

| member | position | verdict |
|---|---|---|
| Halvorsen | E1: formally reject this config for real money; record it before the S&P 500 result is seen. E2: forward paper trading only as a plumbing test. E3: any follow-up is a new study with one pre-registered change; N counts every config hash; the holdout is spent. E4: S&P 500 results are exploratory and trigger nothing | adopted (`results/VERDICT.json`, DESIGN.md, dashboard banner) |
| Zhao | F1: the Engle-Granger screen selects the wrong thing. It is scale-free, so it favours near-clone ETFs (RWR/VNQ, IYR/VNQ, HYG/JNK) whose tracking-error spreads (about 13 bp) are about the size of the costs (about 10 bp a round trip). F2: stop. A cost-scaled screen would be a new trial, judged only on forward data | A (the cost arithmetic matches `strategy.py`: 5 bp × $10k gross per fill) |

**Decision: the config is REJECTED for real money.** Reported as "no evidence of an edge after costs".
A forward paper book (`runs/paper_etf`) started on 2026-11-18 as a plumbing test. It holds nothing yet: no ETF
pair passed selection in the current window, and the next selection is around 2027-01-14.
