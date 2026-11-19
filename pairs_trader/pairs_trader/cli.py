"""Command line (run from the pairs_trader folder).

  python -m pairs_trader.cli download etf|sp500           # Yahoo Finance -> data/<universe>/
  python -m pairs_trader.cli backtest --data data/etf --period dev|holdout|full
  python -m pairs_trader.cli replay   --data data/etf --state runs/replay_etf --from 2023-11-01
  python -m pairs_trader.cli paper    --data data/etf --state runs/paper_etf [--refresh] [--accept-jump T]
  python -m pairs_trader.cli trade    --data data/sp500 --state runs/trade_sp500 --broker sim|alpaca-paper --refresh
  python -m pairs_trader.cli demo [--null]                # synthetic data with known answers
  python -m pairs_trader.cli live                         # refuses: not implemented
Settings are frozen in config.py. There is deliberately no command-line switch for any strategy parameter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import (deflated_sharpe, dose_response, placebo_distribution, sharpe, stress_table,
                       walk_forward)
from .config import StrategyConfig


def report(prices, cfg, sectors, dividends=None, ledger=None, universe="synthetic", placebo_draws=39,
           note="", returns_from=None):
    """Full development report. `returns_from` restricts the evaluated returns (holdout) without changing
    what the strategy sees (it still forms pairs on earlier data)."""
    from . import ledger as L
    res = walk_forward(prices, cfg, sectors, dividends=dividends)
    if returns_from is not None:
        res.daily_pnl = res.daily_pnl[res.daily_pnl.index >= pd.Timestamp(returns_from)]
        res.trades = [t for t in res.trades if t.exit_fill >= pd.Timestamp(returns_from)]
    s = res.summary()
    if ledger:
        L.append(ledger, cfg, universe, s, note, universe=universe)
        trials = L.trial_sharpes(ledger, universe)
    else:
        trials = [s["sharpe"]]
    dsr = deflated_sharpe(res.daily_ret, trials)
    out = {"summary": s, "deflated": dsr, "dose_response": dose_response(res).astype(str).to_dict("records"),
           "windows": [{**w, "start": str(w["start"].date()), "end": str(w["end"].date())} for w in res.windows],
           "equity": res.daily_ret.cumsum().round(6).to_dict()}
    if placebo_draws:
        placebo = placebo_distribution(prices, cfg, sectors, draws=placebo_draws, dividends=dividends)
        stress = stress_table(prices, cfg, sectors, dividends=dividends)
        beat = sum(p >= s["sharpe"] for p in placebo)
        out.update({
            "placebo_sharpes": sorted(round(x, 2) for x in placebo),
            "stress": stress.round(3).to_dict("records"),
            "gates": {
                "bootstrap_ci_excludes_0": bool(s["sharpe_ci95"][0] > 0),
                "deflated_sharpe_gt_0.95": bool(dsr["n_trials"] >= 2 and dsr["dsr"] > 0.95),
                "beats_all_39_matched_placebos": bool(beat == 0 and s["n_trades"] > 0),
                "positive_at_2x_cost_plus_delay": bool(stress.query(
                    "cost_mult == 2 and delay_bars == @cfg.exec_delay_bars + 1")["sharpe"].iloc[0] > 0),
            }})
        out["all_gates_pass"] = all(out["gates"].values())
    out["equity"] = {str(k.date()): v for k, v in out["equity"].items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pairs_trader")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("universe", choices=["etf", "sp500", "sp400"])
    d.add_argument("--out", default=None)
    b = sub.add_parser("backtest")
    b.add_argument("--data", required=True)
    b.add_argument("--period", choices=["dev", "holdout", "full"], default="dev")
    b.add_argument("--ledger", default="runs/ledger.jsonl")
    b.add_argument("--out", default=None, help="write the JSON report here")
    b.add_argument("--note", default="")
    r = sub.add_parser("replay")
    r.add_argument("--data", required=True)
    r.add_argument("--state", required=True)
    r.add_argument("--from", dest="start", default="2023-11-01")
    p = sub.add_parser("paper")
    p.add_argument("--data", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--refresh", action="store_true", help="re-download from Yahoo before running")
    p.add_argument("--accept-jump", action="append", default=[])
    tr = sub.add_parser("trade", help="daily broker loop (Study 3 config; paper accounts only)")
    tr.add_argument("--data", required=True)
    tr.add_argument("--state", required=True)
    tr.add_argument("--broker", choices=["sim", "alpaca-paper"], default="sim")
    tr.add_argument("--refresh", action="store_true")
    tr.add_argument("--accept-jump", action="append", default=[])
    dm = sub.add_parser("demo")
    dm.add_argument("--null", action="store_true")
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
        for k in ("windows", "equity"):
            out.pop(k)
        print(json.dumps(out, indent=1, default=str))
        return 0

    from . import data as D
    if a.cmd == "download":
        cache = Path(f"data/{a.universe}/constituents.csv")
        groups = (D.etf_universe() if a.universe == "etf" else
                  D.sp500_universe(cache) if a.universe == "sp500" else D.sp400_universe(cache))
        px, dv = D.download(list(groups))
        out = Path(a.out or f"data/{a.universe}")
        D.save(px, dv, {t: g for t, g in groups.items() if t in px.columns}, out)
        print(f"saved {px.shape[1]} tickers, {px.index[0].date()}..{px.index[-1].date()} to {out}")
        return 0

    px, dv, groups = D.load(Path(a.data))
    universe = Path(a.data).name
    if a.cmd == "backtest":
        if a.period == "dev":
            sl = px.index <= pd.Timestamp(D.DEV_END)
            out = report(px[sl], cfg, groups, dv[sl], a.ledger, f"{universe}-dev", note=a.note)
        elif a.period == "holdout":
            out = report(px, cfg, groups, dv, a.ledger, f"{universe}-holdout", placebo_draws=0, note=a.note,
                         returns_from=D.HOLDOUT_START)
        else:
            out = report(px, cfg, groups, dv, a.ledger, f"{universe}-full", note=a.note)
        text = json.dumps(out, indent=1, default=str)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text)
        brief = {k: out[k] for k in ("summary", "deflated", "gates", "all_gates_pass") if k in out}
        print(json.dumps(brief, indent=1, default=str))
        return 0

    from .paper import run_paper
    if a.cmd == "replay":
        days = px.index[px.index >= pd.Timestamp(a.start)]
        last = None
        for dday in days:
            sl = px.index <= dday
            last = run_paper(px[sl], cfg, a.state, groups, dividends=dv[sl])
        eq = json.loads((Path(a.state) / "state.json").read_text())["equity"]
        print(json.dumps({"days": len(days), "final_paper_pnl": last["paper_pnl"], "first": eq[0], "last": eq[-1]},
                         indent=1, default=str))
        return 0
    if a.cmd == "trade":
        from .broker import AlpacaPaperBroker, SimBroker
        from .study2 import S3
        from .trader import Halt, run_trader
        now = pd.Timestamp.now(tz="America/New_York")
        if a.refresh:
            px, dv = D.download(list(groups))
            D.save(px, dv, groups, Path(a.data))
        today = str(px.index[-1].date())
        if a.broker == "alpaca-paper":
            br = AlpacaPaperBroker()
            if now.hour < 19:
                print("run after 19:00 ET: market-on-close orders sent 15:50-19:00 ET are rejected", file=sys.stderr)
                return 2
            cal = br.calendar(str((now + pd.Timedelta(days=1)).date()), str((now + pd.Timedelta(days=10)).date()))
            nxt = cal[0]["date"]
        else:
            br = SimBroker.load(Path(a.state) / "simbroker.json")
            br.settle(today, px.iloc[-1].to_dict())          # today's closing auction fills yesterday's orders
            nxt = str((px.index[-1] + pd.offsets.BDay(1)).date())
        try:
            out = run_trader(px, S3, a.state, br, groups, dividends=dv, next_session=nxt,
                             accept_jumps=tuple(a.accept_jump), expected_date=now.normalize().tz_localize(None)
                             - pd.offsets.BDay(0 if now.hour >= 17 else 1))
        except Halt as e:
            print(f"HALT: {e}", file=sys.stderr)
            return 3
        print(json.dumps(out["broker"], indent=1, default=str))
        return 0
    if a.cmd == "paper":
        if a.refresh:
            px, dv = D.download(list(groups))
            D.save(px, dv, groups, Path(a.data))
        out = run_paper(px, cfg, a.state, groups, dividends=dv, accept_jumps=tuple(a.accept_jump),
                        expected_date=pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
                        - pd.offsets.BDay(0 if pd.Timestamp.now(tz="America/New_York").hour >= 17 else 1))
        print(json.dumps(out, indent=1, default=str))
        return 0


if __name__ == "__main__":
    sys.exit(main())
