# Study 2: pre-registration (frozen 2026-11-19, after board rounds 1–2, before any Study 2 run)

The board record is in `docs/REVIEW_BOARD.md` §9. The draft (`STUDY2_PROPOSAL.md`) and the round-1 judge notes
(`study2_round1_judge.md`) are superseded by this file.

## What is being tested, said plainly (Halvorsen r2)
- **The one strategy change.** A cost filter applied *before* Benjamini-Hochberg removes candidate pairs whose spread
  cannot pay for its own trading.
  - **Effect on ETFs:** it removes the near-clone pairs (RWR/VNQ, HYG/JNK …) that Study 1 traded.
  - **Effect on stocks:** it keeps almost everything.
- **Where the evidence for the filter comes from.** Its own claim (that it removes untradable pairs without breaking
  FDR control) is tested **only on the simulator** (gates P1–P5).
- **What the confirmation universes test.** They are both stock universes, so they test **the stock pairs rule**: the
  Study 1 rules, plus the filter (inert on stocks), plus whole-share sizing. A pass there is not credited to the filter.
- **What can honestly be promised.** This study can show that the strategy has no detectable after-cost edge, or that
  an edge is not ruled out. It cannot prove that the strategy makes money. The software is built to trade correctly on
  a paper account; whether it earns money is only settled by years of forward results. No real capital is committed
  on the strength of this study.

## Config (`StrategyConfig`, Study 1 defaults plus the following)
| field | value | why |
|---|---|---|
| `study` | 2 | Enables the full-width calibration seed (Novak C2) and salted order ids (M5). |
| `cost_filter_m` | **1.0** (decision) | Keep a pair iff sd(Δla − b_r·Δlb)/(1+\|b_r\|) ≥ m·2·c, with b_r the **returns-space** β. In words: one typical day's spread move per gross dollar must be at least the round-trip cost per gross dollar. Fixed a priori; it is loosest for slow pairs (Zhao r2), and that is stated here. |
| `screen_cost_bps` | 5.0 | The filter's own cost. The ×2/×3 stress test does not scale it (Novak M1). |
| `integer_shares` | True | Whole shares, sized at the decision close. The search is q_A ∈ round(n_A/p_A) ± 3, minimising the hedge-ratio error; pairs are **never skipped** for rounding (Halvorsen r2, Asante r2). |

Other properties of this config:
- Paper entry orders expire at t + delay; exits still wait for the first real price (Novak r2).
- The filter is applied identically in `select_pairs`, in every calibration bootstrap copy, and in the placebo picker.
- **Sensitivity runs:** m = 0.5 and m = 2.0, on development data only. They are counted in N and never run on a holdout.
- **Known limitation (declined on proportion grounds):** backtest sizing uses split-adjusted prices. That understates
  rounding error for names that split later (Asante r2). Paper sizing uses current, nominal prices.

## Gate 0: simulator (`scripts/validate_study2_sim.py`, `simulate2.py`; 200 panels × 5 sectors; must pass first)
- **Setup.** Selection only: calibration off and BH per sector at q = 0.10. FDR counts null, clone_null and cross pairs
  as nulls; coint, clone and level_step count as true; beta_drift counts as neither.

| gate | requirement |
|---|---|
| P1 | Planted clones removed by the filter: the 95% Wilson lower bound is ≥ 0.90. |
| P2 | Planted coint pairs kept by the filter: the Wilson lower bound is ≥ 0.95. |
| P3 | FDR among BH discoveries in the filtered family (mean over panels of V/max(R,1)) is ≤ q + 2·MC SE. |
| P4 | Zero selected pairs have a filter statistic below the threshold. |
| P5 | Null rejection rate (raw p < 0.05) is equal for nulls that pass and nulls that fail the filter, within 2 SE. Needs ≥ 1000 failing nulls. |
| P6 | Reported, not gated: the null share among selected pairs, with and without the filter. |

- If P1–P5 fail, Study 2 stops here and is recorded as failed. There is no retuning.

## Data (hashed in `results/study2_freeze.json` before any real-data run)
- **ETF (development only):** `data/etf`. Its holdout was spent by Study 1.
- **S&P 500:** `data/sp500`. The holdout is 2023-11-01 → last bar, **never opened**.
- **S&P MidCap 400 (never touched):** constituents from the public Wikipedia list (GICS sub-industry), prices from
  Yahoo. Evaluated returns run 2019-11-01 → 2026-10-31, with formation from 2017-11. No midcap file existed before
  this freeze.
- **Survivorship:** both stock universes are today's members, so a pass is an upper bound.

## Order of runs
1. Gate 0.
2. Freeze the data hashes and commit.
3. Development runs (report the four gates and the DSR with global N):
   - ETF dev and S&P 500 dev at m = 1;
   - sensitivity runs at m = 0.5 and 2 on S&P dev (summary only).
   - **Development can only veto:** VETO if the S&P dev net Sharpe is ≤ 0. INCONCLUSIVE if S&P dev has < 150 trades.
4. Holdouts, opened once: the S&P 500 holdout and the MidCap 400 evaluation period, together.

## Holdout rule (Halvorsen r2), evaluated by `pairs_trader/study2.py`
- **Unit.** Each pair-window i in universe u: x_i = net P&L / gross notional per pair ($10k).
- **Primary test.** T = mean of x_i over all pooled holdout units.
  - Sign-flip test: B = 99,999 flips with rng(20261119); p = (1 + #{T* ≥ T}) / (1 + B).
  - Robustness, reported only: flips by universe × window cluster.
- **Pooled Sharpe.** S_u is the annualised daily Sharpe of universe u over Y_u years. S_p = Σ Y_u S_u / Σ Y_u;
  SE = 1/√ΣY_u; UB90 = S_p + 1.2816·SE. ret_p is the pooled net return per year on $100k gross capital.
- **Outcomes, evaluated in order:**
  1. **INCONCLUSIVE:** pooled holdout round trips < 60, or S&P dev trades < 150.
  2. **FALSIFIED:** S_p ≤ 0, or UB90 < 0.5.
  3. **CONFIRMED:** p < 0.05 AND ret_p ≥ 2%/yr AND T_u > 0 in each universe.
  4. **NOT FALSIFIED:** anything else. This permits forward paper trading only.
- **N for the deflated Sharpe.** N counts distinct config hashes ever run on real data, across all universes:
  f201d54b4912 plus the Study 2 configs at m = 0.5, 1 and 2, so N = 4. The DSR is reported on development data only.

## Real money
**No outcome on its own permits real money.** It would need all of the following:
- CONFIRMED;
- at least 6 months of a forward paper record on the broker paper account that tracks the backtest;
- a separate board review of the live path;
- a level-shift detector, or half_life_max ≤ 22 run as a separately logged trial. Volkov r2: slow pairs are blind to
  level steps.
