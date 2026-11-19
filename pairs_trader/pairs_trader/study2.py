"""Studies 2/3 runner. The rules are pre-registered in docs/STUDY2.md and docs/STUDY3.md; this file only executes
them.

  python -m pairs_trader.study2 freeze      # hash data + code + config -> results/study3_freeze.json
  python -m pairs_trader.study2 dev NAME    # NAME in etf, sp500, sp500_m0.5, sp500_m2 -> results/study3_dev_NAME.json
  python -m pairs_trader.study2 holdout     # opens BOTH holdouts once -> results/study3_holdout.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .backtest import deflated_sharpe, walk_forward
from .cli import report
from .config import StrategyConfig
from .ledger import append, file_sha256

S3 = StrategyConfig().with_(study=2, cost_filter_m=1.0, integer_shares=True)
assert S3.hash() == "6ce6d3e4de60", "Study 3 config drifted from the pre-registration"
SENS = {"sp500_m0.5": 0.5, "sp500_m2": 2.0}
RES = Path("results")
LEDGER = RES / "ledger_real_data.jsonl"          # tracked in git (Novak M3)
SP500_HOLDOUT = "2023-11-01"
SP400_FROM, SP400_EVAL, SP400_END = "2017-11-01", "2019-11-01", "2026-10-31"


def _git():
    return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()


def freeze():
    files = {str(p): file_sha256(p) for u in ("etf", "sp500", "sp400") for p in sorted(Path(f"data/{u}").glob("*.csv"))}
    out = {"config_hash": S3.hash(), "config": json.loads(S3.to_json()), "git": _git(), "data_sha256": files,
           "sensitivity_m": SENS, "note": "frozen before any Study 2/3 real-data run"}
    (RES / "study3_freeze.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _check_frozen(universe):
    fz = json.loads((RES / "study3_freeze.json").read_text())["data_sha256"]
    for p in sorted(Path(f"data/{universe}").glob("*.csv")):
        if fz.get(str(p)) != file_sha256(p):
            raise SystemExit(f"{p} changed since the freeze")


def dev(name):
    uni = "etf" if name == "etf" else "sp500"
    _check_frozen(uni)
    px, dv, groups = D.load(Path(f"data/{uni}"))
    sl = px.index <= pd.Timestamp(D.DEV_END)
    cfg = S3.with_(cost_filter_m=SENS[name]) if name in SENS else S3
    out = report(px[sl], cfg, groups, dv[sl], None, f"{uni}-dev", placebo_draws=0 if name in SENS else 39)
    s = out["summary"]
    append(LEDGER, cfg, file_sha256(Path(f"data/{uni}/prices.csv")), s, note=f"study3 dev {name}", universe=f"{uni}-dev")
    out["config_hash"] = cfg.hash()
    (RES / f"study3_dev_{name}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("summary", "gates", "all_gates_pass") if k in out}, indent=1, default=str))


def global_dsr():
    """N = distinct config hashes ever run on real data (Halvorsen r1 C2): Study 1 plus the three Study 3 configs.
    Trial Sharpes are each config's S&P 500 dev Sharpe; the DSR is evaluated on the decision config's dev returns."""
    s1 = json.loads((RES / "sp500_dev.json").read_text())["summary"]["sharpe"]
    trials = [s1] + [json.loads((RES / f"study3_dev_{n}.json").read_text())["summary"]["sharpe"]
                     for n in ("sp500", "sp500_m0.5", "sp500_m2")]
    return trials


def dev_verdict():
    sp = json.loads((RES / "study3_dev_sp500.json").read_text())["summary"]
    if sp["n_trades"] < 150:
        return "INCONCLUSIVE", sp
    if sp["sharpe"] <= 0:
        return "VETO", sp
    return "PASS_TO_HOLDOUT", sp


def _units(res, start, end, cfg):
    """x_i = net P&L per (pair, trading window) / gross per pair, for trades exiting inside [start, end]."""
    rows = []
    wins = [(pd.Timestamp(w["start"]), pd.Timestamp(w["end"])) for w in res.windows]
    for t in res.trades:
        if not (pd.Timestamp(start) <= t.exit_fill <= pd.Timestamp(end)):
            continue
        w = next((i for i, (a, b) in enumerate(wins) if a <= t.entry_fill <= b), -1)
        rows.append({"pair": t.pair, "window": w, "pnl": t.pnl})
    df = pd.DataFrame(rows, columns=["pair", "window", "pnl"])
    g = df.groupby(["pair", "window"], as_index=False).pnl.sum()
    g["x"] = g.pnl / (cfg.capital / cfg.max_pairs)
    return g, len(df)


def holdout():
    verdict, _ = dev_verdict()
    if verdict != "PASS_TO_HOLDOUT":
        raise SystemExit(f"development verdict is {verdict}: the holdouts stay closed")
    if (RES / "study3_holdout.json").exists():
        raise SystemExit("the holdouts were already opened")
    cfg = S3
    per = {}
    for uni in ("sp500", "sp400"):
        _check_frozen(uni)
        px, dv, groups = D.load(Path(f"data/{uni}"))
        if uni == "sp500":
            ev_start, ev_end = SP500_HOLDOUT, str(px.index[-1].date())
        else:
            keep = (px.index >= pd.Timestamp(SP400_FROM)) & (px.index <= pd.Timestamp(SP400_END))
            px, dv = px[keep], dv[keep]
            ev_start, ev_end = SP400_EVAL, SP400_END
        res = walk_forward(px, cfg, groups, dividends=dv)
        daily = res.daily_ret[(res.daily_ret.index >= pd.Timestamp(ev_start)) & (res.daily_ret.index <= pd.Timestamp(ev_end))]
        units, n_rt = _units(res, ev_start, ev_end, cfg)
        yrs = len(daily) / 252.0
        sr = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 else 0.0
        per[uni] = {"eval": [ev_start, ev_end], "years": yrs, "sharpe": sr, "ann_return": float(daily.mean() * 252),
                    "round_trips": n_rt, "units": len(units), "T_u": float(units.x.mean()) if len(units) else float("nan"),
                    "_x": units.x.to_numpy(), "_cluster": (uni + ":" + units.window.astype(str)).to_numpy()}
        append(LEDGER, cfg, file_sha256(Path(f"data/{uni}/prices.csv")), {"sharpe": sr, "n_trades": n_rt},
               note=f"study3 HOLDOUT {uni} {ev_start}..{ev_end}", universe=f"{uni}-holdout")
    x = np.concatenate([per[u]["_x"] for u in per])
    cl = np.concatenate([per[u]["_cluster"] for u in per])
    T = float(x.mean()) if len(x) else float("nan")
    rng = np.random.default_rng(20261119)
    B = 99_999
    if len(x):
        flips = rng.choice([-1.0, 1.0], size=(B, len(x)))
        p = float((1 + np.sum((flips * x).mean(axis=1) >= T)) / (1 + B))
        uc, inv = np.unique(cl, return_inverse=True)
        cf = rng.choice([-1.0, 1.0], size=(B, len(uc)))[:, inv]
        p_cluster = float((1 + np.sum((cf * x).mean(axis=1) >= T)) / (1 + B))
    else:
        p = p_cluster = float("nan")
    Y = sum(per[u]["years"] for u in per)
    S_p = sum(per[u]["years"] * per[u]["sharpe"] for u in per) / Y
    SE = 1 / np.sqrt(Y)
    UB90 = S_p + 1.2816 * SE
    ret_p = sum(per[u]["years"] * per[u]["ann_return"] for u in per) / Y
    n_rt = sum(per[u]["round_trips"] for u in per)
    dev_trades = json.loads((RES / "study3_dev_sp500.json").read_text())["summary"]["n_trades"]
    if n_rt < 60 or dev_trades < 150:
        outcome = "INCONCLUSIVE"
    elif S_p <= 0 or UB90 < 0.5:
        outcome = "FALSIFIED"
    elif p < 0.05 and ret_p >= 0.02 and all(per[u]["T_u"] > 0 for u in per):
        outcome = "CONFIRMED"
    else:
        outcome = "NOT FALSIFIED"
    out = {"outcome": outcome, "config_hash": cfg.hash(), "git": _git(),
           "pooled": {"T": T, "p_signflip": p, "p_cluster_robustness": p_cluster, "S_p": S_p, "SE": SE, "UB90": UB90,
                      "ret_p": ret_p, "round_trips": n_rt, "units": int(len(x)), "years": Y},
           "per_universe": {u: {k: v for k, v in d.items() if not k.startswith("_")} for u, d in per.items()}}
    (RES / "study3_holdout.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"freeze": freeze, "holdout": holdout}.get(cmd, lambda: dev(sys.argv[2]))()
