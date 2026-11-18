# event_trader — design draft v0 (for the review board, 2026-11-18)

Goal stated by the user: an ML trading app for **two-outcome (binary) event contracts** — "Will X happen by
date D? YES/NO" — as on regulated prediction-market exchanges (e.g. Kalshi; Polymarket is similar). A YES
contract pays $1 if the event happens, $0 otherwise; its price in [0, 1] is the market's implied probability.
The user wants it to run "passively".

## Proposed architecture
1. **Data layer.** Snapshots of each market over time: timestamp, market id, event id (markets that share an
   underlying event, e.g. bracket strikes), category, best bid / best ask (YES side), last price, volume, open
   interest, close time, and the final resolution (0/1) once settled. Stored as one parquet/CSV table.
2. **Features at decision time t (strictly from data with timestamp <= t):** mid price, spread, log time to
   close, price change over the last 1h / 24h, volume over 24h, category one-hot, the market's own
   favourite-longshot position (|mid - 0.5|).
3. **Model.** Predict P(YES resolves) with a regularised logistic regression on the features above, plus the
   logit of mid price as a feature (so the model learns a *correction* to the market). Optional gradient
   boosting. Isotonic calibration on a held-out time block.
4. **Baseline.** The market itself: p = mid. A model only has value if it beats this in log loss / Brier
   *out of sample*.
5. **Decision rule.** Buy YES at ask if p_model - ask - fee > margin; buy NO at (1 - bid) if
   (1 - p_model) - (1 - bid) - fee > margin. One position per market, held to resolution.
6. **Sizing.** Fractional Kelly (0.25x), capped at 2% of bankroll per market and 10% per event; daily loss
   limit; global kill switch.
7. **Evaluation.** Walk-forward by time: train on markets closed before T, test on markets opened after T.
   Report log loss and Brier vs the market baseline, plus simulated PnL after fees and crossing the spread.
8. **Execution.** Exchange adapter with a **paper** (default) and **live** mode. Live needs an explicit flag,
   an API key from the environment, and passes the same risk checks. Runs on a schedule (cron) for "passive".
9. **Testing.** A synthetic market simulator with a known true probability, a configurable market bias (e.g.
   favourite-longshot) and fees, so we can check that the pipeline finds edge when it exists and finds **none**
   when the market is efficient.

## Open questions for the board
- Is the evaluation leak-free? What is the right unit of resampling for confidence intervals?
- How many resolved markets do we need before a measured edge is distinguishable from zero?
- Is "beats the market price" even identifiable given fees and spread?
- Latency/causality: snapshots, stale quotes, fill assumptions.
- What would make the code silently wrong?
