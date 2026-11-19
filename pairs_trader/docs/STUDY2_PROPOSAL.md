# Study 2 proposal (DRAFT for review-board, 2026-11-19). Not yet frozen. No data has been run for it.

## Why
Study 1 (config f201d54b4912) was REJECTED (board round 4). Zhao F1: the Engle-Granger screen is scale-free,
so it selects near-clone ETFs (RWR/VNQ, IYR/VNQ, HYG/JNK). Their tracking-error spreads (sd ~13 bp) are about
the size of a round-trip cost (~10 bp on each leg). Zhao F2: a cost-scaled screen is a new trial, judged only on
data the study has not seen.

## The ONE strategy change (Halvorsen E3: one pre-registered change)
A pair is tradable only if its expected gross capture per round trip covers costs by a fixed margin:

    sd_eff   = formation spread sd × z_scale            (log units; the same scale the z-score uses)
    capture  = (entry_z − exit_z) × sd_eff × n_A        (n_A = gross / (1+β), the dollars in leg A)
    cost_rt  = 2 × cost_bps × gross                     (4 fills: both legs, in and out)
             + borrow_bps × short_notional × E[hold]/252, E[hold] = half_life × log2(entry_z / exit_z)
    edge_ratio = capture / cost_rt;   keep the pair iff edge_ratio >= 3      (proposed R_min = 3, fixed a priori)

Equivalently: (entry−exit)·sd_eff/(1+β) ≥ 3·(2·5 bp + borrow term). With β ≈ 1 this needs sd_eff ≳ 40 bp.
R_min = 3 is a prior, not tuned: the Study 1 net win rate was 38–49%, so a pair needs a margin well above 1×.
Proposed position in the pipeline: after BH and the β screen, alongside the half-life filter (a screen on the
selected set, like the existing ones). Everything else in `StrategyConfig` is unchanged.

## Validation plan (proposed)
1. Synthetic: on the simulator, the screen must keep planted wide-spread pairs, drop planted narrow ones, and
   must not raise the null false-pass rate of the gates.
2. Development backtest on data already seen (ETF and S&P 500 2010–2023-10). Informative only; N = 2 configs in
   the DSR.
3. Confirmation — options for the board:
   (a) **S&P 500 holdout 2023-11 → 2026-11, never opened** (Study 1 opened only the ETF holdout). Survivorship-
       biased, so a pass is an upper bound.
   (b) A universe never touched: e.g. non-US country/sector ETFs, or another market's large caps.
   (c) Forward paper trading only (months to years before it can say anything).
4. Pass rule (proposed): same four gates on development, plus holdout Sharpe > 0 AND inside the dev 95% CI.

## Engineering for a "fully functional" trader (user request: "a fully functional algo trader that actually works")
- Broker adapter interface: `SimBroker` (existing paper fills) and an **Alpaca paper-trading** adapter (free
  paper accounts, REST, needs API keys). Live endpoint behind a separate, reviewed switch — not built now.
- Order reconciliation before each run, both legs sent together, one-legged-fill flattening, idempotent client
  order ids (README "Before real money" item 3).
- Scheduler: a daily job after the close (cron / GitHub Actions / scheduled task) that runs `paper`, plus an
  alert if it fails or the data checks refuse.
- Dashboard: show the Study 2 screen (edge_ratio per pair) and the broker reconciliation status.

## Questions the board must answer
- Is the edge_ratio formula right (units, β, borrow, which sd)? Is R_min = 3 defensible a priori?
- Does screening after BH (or before) break FDR control or the calibration?
- Which confirmation option (a/b/c)? Is opening the S&P holdout legitimate now that the ETF one is spent?
- What would "actually works" have to mean, given SE(Sharpe) ≈ 1/√years?
- What must the broker integration do before it can be trusted even on a paper account?
