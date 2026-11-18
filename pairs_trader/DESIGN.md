# pairs_trader — design v2 (after the review board of 2026-11-18)

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
