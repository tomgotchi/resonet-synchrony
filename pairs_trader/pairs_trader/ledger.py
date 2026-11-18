"""Append-only trials ledger. Every backtest on real data is one line; the deflated Sharpe uses all of them.

Board (statistician): with ~50-1000 configurations tried, a Sharpe of 0.6-1.0 is what pure selection produces.
The only defence is to count every trial honestly. Never delete lines from this file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def universe_key(prices) -> str:
    """Trials are counted per universe + date range, not per file hash: re-downloading the same data must not
    reset the trial count (board round 2)."""
    cols = ",".join(sorted(map(str, prices.columns)))
    return hashlib.sha256(f"{cols}|{prices.index[0].date()}|{prices.index[-1].date()}".encode()).hexdigest()[:16]


def append(ledger_path, cfg, data_hash: str, summary: dict, note: str = "", universe: str = "") -> None:
    rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git": git_sha(),
           "config_hash": cfg.hash(), "config": json.loads(cfg.to_json()), "data_sha256": data_hash,
           "universe": universe, "sharpe": summary.get("sharpe"), "n_trades": summary.get("n_trades"),
           "note": note}
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def trial_sharpes(ledger_path, universe: str | None = None) -> list[float]:
    """One Sharpe per DISTINCT configuration tried on this universe (the latest run of each config)."""
    p = Path(ledger_path)
    if not p.exists():
        return []
    by_cfg = {}
    for line in p.read_text().splitlines():
        rec = json.loads(line)
        if universe is None or rec.get("universe") == universe:
            if rec.get("sharpe") is not None:
                by_cfg[rec["config_hash"]] = float(rec["sharpe"])
    return list(by_cfg.values())
