# pairs_trader

A paper-trading **pairs-trading (statistical arbitrage)** tool: it looks for two stocks whose prices move together
in a fixed proportion, and when the gap between them is unusually wide it goes long the cheap one and short the
rich one (in the right proportion), closing when the gap closes. It was designed and checked by the project's
review board (`docs/REVIEW_BOARD.md`, 2026-11-18).

## Read this first
- **This is not a money machine.** Pairs trading is decades old and heavily competed. Most published edges
  shrank after costs once they became known. Assume your result will be zero or negative until the gates
  below say otherwise.
- **The tool has no live trading.** `live` refuses on purpose. `paper` writes simulated orders to a CSV.
- **"Passive" ≠ unattended.** Data errors, splits, delistings and news jumps break pairs. Someone has to read
  the daily output.
- **Proving an edge takes years.** Distinguishing a true Sharpe of 0.5 from zero needs ~12 years of daily
  returns at 50% power (~28 years at 80%). A good-looking 6-month paper run means almost nothing.
- Not financial advice. Only ever risk money you can lose entirely.

## Install & try
```bash
cd pairs_trader
pip install -r requirements.txt
python -m pytest -q tests                      # null calibration, causality, accounting, paper == backtest
python -m pairs_trader.cli demo                # synthetic market WITH real pairs: shows the report and gates
python -m pairs_trader.cli demo --null         # synthetic market with NO real pairs: everything must fail
```

## Real data (Yahoo Finance, free)
```bash
python -m pairs_trader.cli download etf        # 72 ETFs in 8 frozen asset-class groups -> data/etf/
python -m pairs_trader.cli download sp500      # current S&P 500 by GICS sub-industry -> data/sp500/
python -m pairs_trader.cli backtest --data data/etf --period dev --out results/etf_dev.json
```
- Prices are split-adjusted closes with dividends booked as cash (history never shifts).
- `--period dev` is 2010 to 2023-10. `--period holdout` (2023-11 onward) was opened **once**, as pre-registered in
  DESIGN.md. Running more configurations on the same data makes every result less trustworthy; each run is
  counted in `runs/ledger.jsonl`.
- **S&P 500 results are an upper bound.** The list is today's members, so every stock in it survived. Free data
  cannot fix that; ETF results are the primary ones.

## Paper trial (no money)
```bash
python -m pairs_trader.cli replay --data data/etf --state runs/replay_etf --from 2023-11-01   # dry run over the holdout
python -m pairs_trader.cli paper  --data data/etf --state runs/paper_etf --refresh            # one real day, after the close
```
`paper` downloads fresh prices, fills yesterday's paper orders at today's close, and writes today's orders to
`runs/paper_etf/orders.csv`. Re-running the same day does nothing. It refuses on stale or revised data or on a
>25% move (unless `--accept-jump TICKER`). Kill switch: `touch runs/paper_etf/KILL`.
For a daily schedule: `cron` at about 18:00 New York time on weekdays.

## Dashboard
```bash
streamlit run app.py
```
Tabs: backtest research (gates, equity, placebo, stress, pairs), paper trial (P&L, book, orders, "run today",
kill switch) and settings (read-only). By design the dashboard cannot change settings or trade live.

## Before real money (if ever)
1. All four backtest gates pass on survivorship-free data, with every trial logged.
2. ≥ 3 months of paper trading whose fills and P&L track a backtest of the same days.
3. A broker integration that sends both legs together, reconciles positions before every run, uses
   idempotent order ids, and flattens a one-legged position. That is not built; it should be reviewed on its own.
4. Start with 1–5% of what you would eventually commit; stop if drawdown exceeds 2× the backtest's.

## Layout
`pairs_trader/coint.py` (Engle-Granger, BH, half-life) · `strategy.py` (selection, trading state machine) ·
`backtest.py` (walk-forward, bootstrap, deflated Sharpe, placebo, stress) · `ml.py` (optional filter, off) ·
`paper.py` · `ledger.py` · `simulate.py` (synthetic markets with known truth) · `cli.py`.
