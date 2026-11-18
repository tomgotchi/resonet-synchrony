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
python -m pytest -q tests                      # 21 tests: null calibration, causality, accounting, paper mode
python -m pairs_trader.cli demo                # synthetic market WITH real pairs: shows the report and gates
python -m pairs_trader.cli demo --null         # synthetic market with NO real pairs: everything must fail
```

## Your own data
`prices.csv`: first column a date, then one column per ticker of **split- and dividend-adjusted closes**
(survivorship-free if possible, i.e. including stocks that were later delisted). `sectors.csv`: `ticker,sector`.
At least ~3 years of history (2 years formation + trading windows); 10+ years for the statistics to mean much.

```bash
python -m pairs_trader.cli backtest --prices prices.csv --sectors sectors.csv --note "first look"
```
Every backtest appends a line to `runs/ledger.jsonl`. **Never delete lines**: the deflated Sharpe uses the
number of configurations you tried on that data file. If you change any setting in `config.py`, that is another
trial, and the bar for "significant" rises.

The report gives the summary, the bootstrap CI, the deflated Sharpe, random-pair placebo Sharpes, the cost/delay
stress table and four **gates**. If `all_gates_pass` is false, stop.

## Paper trading (daily, e.g. by cron after the close)
```bash
python -m pairs_trader.cli paper --prices prices.csv --sectors sectors.csv --state runs/paper
```
- Fills yesterday's paper orders at today's close, then writes today's orders to `runs/paper/orders.csv`.
- Re-running on the same day does nothing (no duplicate orders).
- Refuses to run on suspicious data (missing latest prices, >40% one-day moves) or if settings changed.
- Kill switch: `touch runs/paper/KILL` (or `PAIRS_KILL=1`) → only flattening orders from then on.

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
