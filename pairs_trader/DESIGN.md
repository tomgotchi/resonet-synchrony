# pairs_trader — design v3 (after review-board rounds 1-3, 2026-11-18)

Pairs trading / statistical arbitrage: find two stocks whose log prices are tied (cointegrated), hold a
hedge-ratio-weighted long/short position when the spread is unusually wide, close it when it reverts.
v0 (binary event contracts) was a misreading of the request and is dropped. v1 is in git history; the board
record is `docs/REVIEW_BOARD.md`. Every change below cites the board finding that caused it.

## Frozen v1 rules (`pairs_trader/config.py`; any change is a new trial and goes in the ledger)
| step | rule | why (board) |
|---|---|---|
| universe | user CSV of adjusted closes; pairs only **within a sector** | multiplicity: all-pairs screening has no power after FDR control (judge) |
| test | Engle-Granger, ADF on OLS residual with **MacKinnon N=2** p-values, **both directions**, p = max | naive ADF p rejects 14% of null pairs; direction disagreement ~7% (Halvorsen, Zhao, Novak, Volkov) |
| multiplicity | Benjamini-Hochberg q = 0.05 over all candidate pairs | ~5% of hundreds of null pairs pass otherwise; top-K fills with noise (Halvorsen) |
| formation | **504** days (not 252) | 252 days cannot tell a half-life of 20+ from noise (Volkov) |
| half-life | Kendall bias-corrected AR(1), keep 2–30 days | raw estimate ~30% too fast; the filter alone screens nothing (Volkov, Halvorsen) |
| hedge | static OLS beta in [0.2, 5]; positions **1 : beta in dollars** | dollar-neutral 1:1 leaves (beta − 1) of unhedged trend (Novak, Volkov) |
| signal | z against the **formation** mean/sd (frozen), entry 2, exit 0.5, stop 4, time stop 3 half-lives | rolling-mean z "reverts" 96% of the time on a random walk (Zhao) |
| breaks | two-sided CUSUM on standardised AR(1) innovations (k 0.5, h 8); a break closes and blacklists the pair | a z-stop adapts to a break and fires late (Volkov) |
| execution | decide at close t, fill at close t+1 | filling at the signal bar's close fakes ~8 bp/trade from bounce (Asante) |
| costs | 5 bp per leg per fill (4 fills per round trip) + 1%/yr borrow on the short leg | (Novak, Asante) |
| missing data | NaN price → forced exit at the last valid price, logged | NaN comparisons are False → frozen positions (Novak) |
| ML filter | **off by default**; label = net P&L sign; trained only on trades closed before the window | the features repeat the price information; the planned label was meaningless (Zhao, Novak) |
| Kalman beta, live broker, data downloader | **cut from v1** | a fast Kalman absorbs the spread (Volkov, Asante); keep v1 small (Novak) |

## Go / no-go gates (printed by `backtest`; all must pass before paper trading means anything)
1. Block-bootstrap (63-day blocks) 95% CI of the Sharpe excludes 0.
2. Deflated Sharpe > 0.95, with N = every trial in the ledger for that data file.
3. Beats ≥ 95% of random same-sector pairs run through identical rules (placebo).
4. Sharpe still > 0 at 2× costs plus one extra bar of delay.
Then ≥ 3 months of paper trading, which checks *plumbing* (fills, costs, tracking), not edge: six months of
returns have a Sharpe standard error of ~1.4 (Halvorsen).

## Known gaps (not fixed in v1)
Point-in-time universe/sector tables and delisting returns (user's data must provide them); earnings blackout;
factor-neutral attribution (is the edge just short-term reversal?); GARCH volatility in the simulator;
opens/auction fills; short-borrow availability. See the board record.

## Round 2-3 changes (code)
- Selection-aware parametric bootstrap per pair (`calibrate.py`): bias-corrected half-life, a z scale so that
  |z| > 2 fires on ~4.6% of days on genuine pairs (was 12.5%), and a per-pair CUSUM threshold (healthy pairs
  alarmed in 12.7% of windows, now ~2.5-3.8%). BH per group at q = 0.10; half-life cap 45.
- Halted exits fill at the first real price after the gap; a leg with no price until the window end is marked
  30% against us. Dividends are cash flows. No entry may fill on a window's last bar.
- Paper mode: crash-safe state, refit from the current file, calendar anchor, borrow/dividends, frozen pairs on
  missing data, split detection, history-revision and stale-bar checks; paper == backtest is a test.
- Placebo: matched (0.05 < p < 0.5, same screens and calibration), 39 draws, gate = no draw ties or beats.
  Deflated Sharpe: N = distinct configs per universe, variance floored at 1/years. Bootstrap blocks 126 days.

## PRE-REGISTRATION for the real-data run (frozen 2026-11-18, before any real-data backtest was run)
Data: Yahoo Finance daily, split-adjusted closes + cash dividends, 2010-01-04 .. 2026-11-18 (`data.py`).
Universes (board round 3, Halvorsen D1/D2):
1. **ETF (primary):** 72 ETFs in 8 frozen asset-class groups (`data.ETF_GROUPS`).
2. **S&P 500 (secondary):** current constituents by GICS sub-industry. Survivorship-biased upper bound.
Config: `StrategyConfig()` defaults at this commit; ML filter OFF (board D4: not run in v1). No other config
will be run on these data before the holdout is opened; any later config is a new ledger trial.
Development: walk-forward on 2010-01-04 .. 2023-10-31 only. Report the summary, the four gates, the dose-response
and the stress table.
Holdout (opened once, after the development report is committed): walk-forward on the full file; holdout returns
are the daily P&L dated 2023-11-01 onward (positions opened before that date and still open count).
**Holdout pass rule:** holdout net Sharpe > 0 AND inside the development 95% block-bootstrap CI. Three years give
SE(SR) ~ 0.58, so the holdout can falsify, not confirm.
Paper trial (no money): replay every holdout trading day through `paper.run_paper` (the exact daily code path),
then compare its P&L with the backtest of the same days (board round 3, Asante D2). Forward paper trading can
then run daily via the interface or cron.
