"""pairs_trader dashboard (local).   Run:  cd pairs_trader && streamlit run app.py

Board round 3 (Asante D4): the interface may READ results, RUN the same paper step as the CLI, and SET the kill
switch. It cannot change settings, edit the paper book, or trade live. Settings are frozen in config.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from pairs_trader.config import StrategyConfig

ROOT = Path(__file__).resolve().parent
RESULTS, RUNS, DATA = ROOT / "results", ROOT / "runs", ROOT / "data"

st.set_page_config(page_title="Pairs trader (paper)", layout="wide")
st.title("Pairs trader: paper trading only")
st.caption("No real money is ever used. Live trading is not implemented. Not financial advice.")
_verdict = (ROOT / "results" / "VERDICT.json")
if _verdict.exists():
    _v = json.loads(_verdict.read_text())
    st.error(f"Board verdict: this configuration ({_v['config_hash']}) is {_v['verdict']} for real money. "
             f"Development Sharpe {_v['dev_sharpe']:.2f}, holdout Sharpe {_v['holdout_sharpe']:.2f}. "
             "Paper trading below is a plumbing test, not evidence of profit.")


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


tab_bt, tab_paper, tab_broker, tab_study, tab_cfg = st.tabs(
    ["Backtest research", "Paper trial", "Broker (paper account)", "Study 3 status", "Settings (read-only)"])

with tab_bt:
    reports = sorted((p for p in RESULTS.glob("*.json") if p.name != "VERDICT.json"
                      and "summary" in (load_json(p) or {})),
                     key=lambda p: (not p.stem.startswith("etf_dev"), p.stem))
    if not reports:
        st.info("No reports yet. Run: python -m pairs_trader.cli backtest --data data/etf --period dev "
                "--out results/etf_dev.json")
    else:
        choice = st.selectbox("Report", reports, format_func=lambda p: p.stem)
        rep = load_json(choice) or {}
        s = rep.get("summary", {})
        if "all_gates_pass" in rep:
            if rep["all_gates_pass"]:
                st.success("All four go/no-go gates pass. Next step: paper trading (it tests plumbing, not edge).")
            else:
                st.error("At least one go/no-go gate FAILS. Do not risk money on this strategy.")
            st.table(pd.Series(rep["gates"], name="pass").to_frame())
        c = st.columns(5)
        c[0].metric("Sharpe (net)", f"{s.get('sharpe', float('nan')):.2f}")
        ci = s.get("sharpe_ci95", [None, None])
        c[1].metric("95% CI", f"{ci[0]:.2f} .. {ci[1]:.2f}" if ci[0] is not None else "n/a")
        c[2].metric("Annual return", f"{100 * s.get('ann_return', 0):.2f}%")
        c[3].metric("Max drawdown", f"{100 * s.get('max_drawdown', 0):.2f}%")
        c[4].metric("Trades", s.get("n_trades", 0))
        if rep.get("equity"):
            eq = pd.Series(rep["equity"])
            eq.index = pd.to_datetime(eq.index)
            st.subheader("Cumulative return on capital (simple, net of costs)")
            st.line_chart(eq * 100, y_label="%")
        if rep.get("deflated"):
            st.write("Deflated Sharpe:", rep["deflated"])
        if rep.get("placebo_sharpes"):
            st.subheader("Matched random pairs (placebo) vs the strategy")
            st.bar_chart(pd.Series(rep["placebo_sharpes"], name="placebo Sharpe"))
            st.caption(f"Strategy Sharpe {s.get('sharpe', 0):.2f}. It must beat all 39 placebo draws.")
        if rep.get("stress"):
            st.subheader("Cost and delay stress")
            st.dataframe(pd.DataFrame(rep["stress"]), hide_index=True)
        if rep.get("dose_response"):
            st.subheader("Net P&L per trade by selection p-value (should fall as p rises)")
            st.dataframe(pd.DataFrame(rep["dose_response"]), hide_index=True)
        if rep.get("windows"):
            st.subheader("Trading windows and selected pairs")
            st.dataframe(pd.DataFrame(rep["windows"]), hide_index=True)

with tab_paper:
    states = sorted(p for p in RUNS.glob("*") if (p / "state.json").exists()) if RUNS.exists() else []
    datasets = sorted(p.name for p in DATA.glob("*") if (p / "prices.csv").exists()) if DATA.exists() else []
    colA, colB = st.columns([2, 1])
    with colB:
        st.subheader("Actions")
        ds = st.selectbox("Data", datasets or ["etf"])
        state_name = st.text_input("Paper book folder", value=f"paper_{ds}")
        if st.button("Run today's paper step (downloads fresh prices)"):
            cmd = [sys.executable, "-m", "pairs_trader.cli", "paper", "--data", f"data/{ds}", "--state",
                   f"runs/{state_name}", "--refresh"]
            res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            (st.success if res.returncode == 0 else st.error)(res.stdout[-3000:] or res.stderr[-3000:])
        kill = RUNS / state_name / "KILL"
        if kill.exists():
            st.warning("Kill switch is ON: the book is only being flattened. Remove the KILL file by hand to undo.")
        elif st.checkbox("I want to stop this paper book") and st.button("Activate kill switch"):
            kill.parent.mkdir(parents=True, exist_ok=True)
            kill.write_text("set from dashboard")
            st.warning("Kill switch activated.")
    with colA:
        if not states:
            st.info("No paper book yet. Replay the holdout or press 'Run today's paper step'.")
        else:
            sp = st.selectbox("Paper book", states, format_func=lambda p: p.name)
            stt = load_json(sp / "state.json") or {}
            last = stt.get("last_summary", {})
            m = st.columns(3)
            m[0].metric("Paper P&L ($, on $100k)", f"{last.get('paper_pnl', 0):,.0f}")
            m[1].metric("Last run", stt.get("last_run_date", "-"))
            m[2].metric("Open positions", len(stt.get("book", {})))
            if stt.get("equity"):
                eq = pd.DataFrame(stt["equity"]).set_index("date")["pnl"]
                eq.index = pd.to_datetime(eq.index)
                st.line_chart(eq, y_label="$")
            if stt.get("frozen"):
                st.error(f"Frozen pairs needing a human: {stt['frozen']}")
            if last.get("pairs"):
                st.subheader("Selected pairs this window")
                st.dataframe(pd.DataFrame(last["pairs"])[["a", "b", "group", "beta", "half_life", "z_scale", "z",
                                                          "p"]], hide_index=True)
            st.subheader("Book")
            st.json(stt.get("book", {}))
            log = sp / "orders.csv"
            if log.exists():
                st.subheader("Order log (paper)")
                st.dataframe(pd.read_csv(log).tail(200), hide_index=True)

with tab_broker:
    st.caption("Daily loop: python -m pairs_trader.cli trade --broker sim|alpaca-paper (after 19:00 ET). "
               "Alpaca *paper* only; live trading does not exist in this code.")
    books = sorted(p for p in RUNS.glob("trade_*") if (p / "state.json").exists()) if RUNS.exists() else []
    if not books:
        st.info("No broker book yet. Run scripts/daily.sh (BROKER=sim or BROKER=alpaca-paper).")
    else:
        bk = st.selectbox("Broker book", books, format_func=lambda p: p.name)
        if (bk / "HALT").exists():
            st.error(f"HALTED: {(bk / 'HALT').read_text()} Fix the cause, then delete the HALT file by hand.")
        hb = load_json(bk / "heartbeat.json") or {}
        sstate = load_json(bk / "state.json") or {}
        from pairs_trader.trader import shadow_positions
        shadow = shadow_positions(sstate)
        real = hb.get("positions", {})
        c = st.columns(4)
        c[0].metric("Last run (UTC)", hb.get("utc", "-"))
        c[1].metric("Session / next", f"{hb.get('session', '-')} / {hb.get('next_session', '-')}")
        c[2].metric("Orders submitted", hb.get("submitted", 0))
        c[3].metric("Rejected", hb.get("rejected", 0))
        rows = [{"ticker": t, "shadow": shadow.get(t, 0), "broker": real.get(t, 0),
                 "match": shadow.get(t, 0) == real.get(t, 0)} for t in sorted(set(shadow) | set(real))]
        st.subheader("Positions: shadow book vs broker (must match share for share)")
        st.dataframe(pd.DataFrame(rows), hide_index=True) if rows else st.write("flat")
        if hb.get("events"):
            st.subheader("Events")
            st.write(hb["events"])
        if (bk / "slippage.csv").exists():
            sl = pd.read_csv(bk / "slippage.csv", names=["date", "ticker", "order_id", "broker_px", "close_px", "bp"])
            st.subheader("Fill vs close (bp). Paper fills are simulated: plumbing, not cost evidence")
            st.dataframe(sl.tail(200), hide_index=True)
            st.metric("Mean |slippage| bp", f"{sl.bp.abs().mean():.1f}")
        if (bk / "alerts.log").exists():
            st.subheader("Alerts")
            st.code((bk / "alerts.log").read_text()[-3000:])

with tab_study:
    st.write("Pre-registration: docs/STUDY2.md + docs/STUDY3.md. Every step below was frozen before it ran.")
    for name, label in [("study2_gate0", "Study 2 simulator gate (FAILED: P5 mis-specified)"),
                        ("study3_gate0", "Study 3 simulator gate"), ("study3_freeze", "Data freeze"),
                        ("study3_holdout", "Holdout (opened once)")]:
        d = load_json(RESULTS / f"{name}.json")
        with st.expander(label + ("" if d else " (not run yet)")):
            if d:
                st.json(d.get("pass", d.get("outcome", d)) if name != "study3_freeze" else
                        {"config_hash": d["config_hash"], "git": d["git"], "files": len(d["data_sha256"])})
    devs = sorted(RESULTS.glob("study3_dev_*.json"))
    if devs:
        st.subheader("Development runs (can only veto)")
        st.dataframe(pd.DataFrame([{"run": p.stem.replace("study3_dev_", ""), **{k: (load_json(p) or {})
                     .get("summary", {}).get(k) for k in ("sharpe", "ann_return", "n_trades", "max_drawdown")},
                     "gates_pass": (load_json(p) or {}).get("all_gates_pass")} for p in devs]), hide_index=True)

with tab_cfg:
    cfg = StrategyConfig()
    st.write(f"Config hash **{cfg.hash()}**. Changing any value is a new trial: log it, and expect the bar for "
             "'significant' to rise. Edit config.py by hand; the dashboard cannot.")
    st.json(json.loads(cfg.to_json()))
