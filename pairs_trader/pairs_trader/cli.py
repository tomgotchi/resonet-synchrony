"""Command line.

  python -m pairs_trader.cli demo                         # synthetic data: shows what the tool does and its gates
  python -m pairs_trader.cli backtest --prices px.csv [--sectors sectors.csv] [--ledger runs/ledger.jsonl]
  python -m pairs_trader.cli paper --prices px.csv --state runs/paper [--sectors sectors.csv]
  python -m pairs_trader.cli live ...                     # refuses: not implemented (see README)

prices.csv: first column = date, one column per ticker, ADJUSTED closes, survivorship-free if you can get it.
sectors.csv: columns ticker,sector (pairs are only formed inside a sector).
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .backtest import deflated_sharpe, placebo_distribution, stress_table, walk_forward
from .config import StrategyConfig


def load_prices(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.index.has_duplicates:
        raise SystemExit("duplicate dates in price file")
    return df


def load_sectors(path):
    if not path:
        return None
    s = pd.read_csv(path)
    return dict(zip(s["ticker"].astype(str), s["sector"].astype(str)))


def report(prices, cfg, sectors, ledger=None, data_hash="synthetic", placebo_draws=20, note=""):
    from . import ledger as L
    res = walk_forward(prices, cfg, sectors)
    s = res.summary()
    if ledger:
        L.append(ledger, cfg, data_hash, s, note)
        trials = L.trial_sharpes(ledger, data_hash)
    else:
        trials = [s["sharpe"]]
    dsr = deflated_sharpe(res.daily_ret, trials)
    placebo = placebo_distribution(prices, cfg, sectors, draws=placebo_draws)
    stress = stress_table(prices, cfg, sectors)
    beat = sum(p >= s["sharpe"] for p in placebo)
    gates = {
        "bootstrap_ci_excludes_0": bool(s["sharpe_ci95"][0] > 0),
        "deflated_sharpe_gt_0.95": bool(dsr["dsr"] > 0.95),
        "beats_random_pairs_95pct": bool(beat <= 0.05 * len(placebo)),
        "positive_at_2x_cost_plus_delay": bool(
            stress.query("cost_mult == 2 and delay_bars == @cfg.exec_delay_bars + 1")["sharpe"].iloc[0] > 0),
    }
    return {"summary": s, "deflated": dsr, "placebo_sharpes": sorted(round(x, 2) for x in placebo),
            "stress": stress.round(3).to_dict("records"), "gates": gates, "all_gates_pass": all(gates.values()),
            "windows": res.windows}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pairs_trader")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--null", action="store_true", help="no real pairs at all (everything should fail)")
    b = sub.add_parser("backtest")
    b.add_argument("--prices", required=True)
    b.add_argument("--sectors")
    b.add_argument("--ledger", default="runs/ledger.jsonl")
    b.add_argument("--note", default="")
    p = sub.add_parser("paper")
    p.add_argument("--prices", required=True)
    p.add_argument("--sectors")
    p.add_argument("--state", required=True)
    sub.add_parser("live")
    a = ap.parse_args(argv)
    cfg = StrategyConfig()

    if a.cmd == "live":
        print("Live trading is deliberately not implemented. Paper-trade first and read README.md "
              "('Before real money').", file=sys.stderr)
        return 2
    if a.cmd == "demo":
        from .simulate import PanelConfig, simulate_panel
        kinds = (("null", 45),) if a.null else (("coint", 10), ("near_null", 5), ("null", 25), ("break", 5))
        panel = simulate_panel(PanelConfig(n_days=252 * 10, kinds=kinds, jumps_per_year=0.5, bounce_sd=0.002))
        out = report(panel.prices, cfg, panel.sectors, placebo_draws=10)
        out.pop("windows")
        print(json.dumps(out, indent=1, default=str))
        return 0
    if a.cmd == "backtest":
        from .ledger import file_sha256
        out = report(load_prices(a.prices), cfg, load_sectors(a.sectors), a.ledger, file_sha256(a.prices),
                     note=a.note)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if a.cmd == "paper":
        from .paper import run_paper
        print(json.dumps(run_paper(load_prices(a.prices), cfg, a.state, load_sectors(a.sectors)), indent=1,
                         default=str))
        return 0


if __name__ == "__main__":
    sys.exit(main())
