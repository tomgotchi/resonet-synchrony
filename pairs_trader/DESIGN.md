# pairs_trader — design draft v1 (for the review board, 2026-11-18)

The user wants an ML trading app that "connects two different things and uses them proportionally, like two
stocks", run passively. That is **pairs trading / statistical arbitrage**: find two assets whose prices are tied
(cointegrated), hold a hedge-ratio-weighted long/short position when their spread is unusually wide, and close it
when the spread reverts. (v0 of this file was about binary event contracts; that was a misreading and is dropped.)

## Proposed architecture
1. **Data.** Daily split- and dividend-adjusted closes (and opens) for a universe of liquid US stocks/ETFs, from a
   CSV the user provides, or an optional downloader. Sector labels. Survivorship-bias-free data if available.
2. **Pair selection (formation window, e.g. 252 trading days).** Candidate pairs within the same sector.
   Engle-Granger test: regress log P_A on log P_B, run an ADF test on the residual. Keep pairs with p < 0.05,
   then estimate an Ornstein-Uhlenbeck half-life of the spread and keep 2 <= half-life <= 60 days. Rank by
   p-value; trade the top K (e.g. 10).
3. **Hedge ratio.** Static OLS beta from the formation window; option: a Kalman filter that updates beta daily.
4. **Signal (trading window, e.g. next 126 days).** spread_t = log P_A - beta log P_B; z_t from a rolling mean
   and std (window ~ 2x half-life). Enter when |z| > 2 (short the rich leg, long the cheap leg, dollar-neutral),
   exit when |z| < 0.5, stop-loss when |z| > 4, time stop after 3x half-life.
5. **ML layer.** A classifier (logistic regression / gradient boosting) predicts, at each entry signal, whether the
   trade will revert profitably before its stops. Features: z, half-life, spread volatility, ADF statistic,
   rolling correlation, recent beta drift, market volatility. Trades are taken only if P(success) > threshold.
   Baseline: the same rules without the ML filter.
6. **Execution & costs.** Signal at close t, trade at the next open (t+1). Costs: commission, half-spread
   (e.g. 5 bp per leg per side), short-borrow fee (e.g. 1%/yr), slippage.
7. **Evaluation.** Rolling formation/trading windows walk-forward over many years. Report annualised return,
   Sharpe, max drawdown, turnover, hit rate, vs. the no-ML baseline and vs. buy-and-hold SPY.
8. **Risk.** Max gross exposure, per-pair cap, max concurrent pairs, drawdown kill switch, no trade if a leg is
   hard to borrow.
9. **Broker.** Paper mode (default). Live mode through a broker API (e.g. Alpaca), only with an explicit flag and
   keys from the environment, passing the same risk checks. Scheduled daily run for "passive".
10. **Testing.** Synthetic simulator: cointegrated pairs (common random-walk trend + OU spread with known
    half-life), non-cointegrated random-walk pairs (null), and pairs whose cointegration breaks after formation.
    The pipeline must find edge on the first, none on the second, and lose gracefully on the third.

## Open questions for the board
- Multiple testing in pair selection (hundreds of pairs, p < 0.05). Out-of-sample decay.
- Does the ML filter have any information the rules don't? How to evaluate it without leakage?
- Causality: rolling z, Kalman, next-open fills, adjusted prices that are revised later.
- Is the OU / cointegration model the right dynamical description, and how to detect breaks?
- What will make the code silently wrong? What is the go/no-go for real money?
