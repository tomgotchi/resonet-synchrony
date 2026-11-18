"""Real data: frozen universes and a Yahoo Finance downloader.

Board round 3 decisions (frozen here BEFORE any data was downloaded, 2026-11-18):
- Universe: ETFs are the primary result (little survivorship bias). Current S&P 500 stocks are reported separately
  and labelled a survivorship-biased upper bound (today's constituents all survived to 2026).
- Groups: ETFs by the asset-class list below; stocks by GICS sub-industry (current labels, not point-in-time).
- Prices: split-adjusted closes, NOT dividend-adjusted, with dividends as separate cash per share (Asante D1).
  Yahoo's "Close" with auto_adjust=False is split-adjusted only; dividends come from the actions table.
- Development period 2010-01-01 .. 2023-10-31; holdout 2023-11-01 .. data end, opened once (Halvorsen D3).
- Same-index clones (SPY/IVV/VOO, GLD/IAU, ...) are excluded: they are one asset, not a pair.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

DEV_START, DEV_END, HOLDOUT_START = "2010-01-01", "2023-10-31", "2023-11-01"

ETF_GROUPS: dict[str, list[str]] = {
    "us_sector":      ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    "us_size_style":  ["SPY", "QQQ", "IWM", "MDY", "DIA", "IWD", "IWF", "IWN", "IWO"],
    "us_industry":    ["XOP", "OIH", "KRE", "KBE", "SMH", "XBI", "IBB", "ITB", "XHB", "XRT"],
    "dev_country":    ["EWA", "EWC", "EWG", "EWJ", "EWU", "EWQ", "EWL", "EWP", "EWI", "EWN", "EWD", "EWH", "EWS"],
    "em_country":     ["EWZ", "EWW", "EWY", "EWT", "FXI", "EZA", "EEM", "ECH"],
    "commodity":      ["GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "DBA", "DBC", "XME"],
    "rates_credit":   ["TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "AGG", "TIP", "EMB", "MUB"],
    "real_estate":    ["VNQ", "IYR", "ICF", "RWR"],
}

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def etf_universe() -> dict[str, str]:
    return {t: g for g, ts in ETF_GROUPS.items() for t in ts}


def sp500_universe(cache: Path | None = None) -> dict[str, str]:
    """Current S&P 500 -> GICS sub-industry. SURVIVORSHIP-BIASED: only today's survivors."""
    import urllib.request
    if cache is not None and cache.exists():
        df = pd.read_csv(cache)
    else:
        with urllib.request.urlopen(SP500_URL, timeout=60) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache, index=False)
    df["yahoo"] = df["Symbol"].str.replace(".", "-", regex=False)
    return dict(zip(df["yahoo"], df["GICS Sub-Industry"]))


def download(tickers: list[str], start: str = DEV_START, end: str | None = None, batch: int = 100):
    """Return (closes, dividends): split-adjusted closes and cash dividends per share on ex-dates."""
    import yfinance as yf
    closes, divs = [], []
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        raw = yf.download(chunk, start=start, end=end, auto_adjust=False, actions=True, progress=False,
                          group_by="column", threads=True)
        if raw.empty:
            continue
        c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
        d = raw["Dividends"] if "Dividends" in raw.columns.get_level_values(0) else c * 0
        closes.append(c)
        divs.append(d.reindex(columns=c.columns))
    px = pd.concat(closes, axis=1).sort_index()
    dv = pd.concat(divs, axis=1).sort_index().reindex(px.index).fillna(0.0)
    px.index = pd.to_datetime(px.index).tz_localize(None)
    dv.index = px.index
    px = px.loc[:, px.notna().sum() > 0]
    return px, dv.reindex(columns=px.columns).fillna(0.0)


def save(px: pd.DataFrame, dv: pd.DataFrame, groups: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    px.to_csv(folder / "prices.csv")
    dv.to_csv(folder / "dividends.csv")
    pd.DataFrame({"ticker": list(groups), "sector": list(groups.values())}).to_csv(folder / "sectors.csv",
                                                                                   index=False)


def load(folder: Path):
    px = pd.read_csv(folder / "prices.csv", index_col=0, parse_dates=True)
    dv = pd.read_csv(folder / "dividends.csv", index_col=0, parse_dates=True)
    s = pd.read_csv(folder / "sectors.csv")
    return px, dv, dict(zip(s.ticker.astype(str), s.sector.astype(str)))


def quality_report(px: pd.DataFrame) -> pd.DataFrame:
    r = np.log(px).diff()
    return pd.DataFrame({"first": px.apply(pd.Series.first_valid_index), "last": px.apply(pd.Series.last_valid_index),
                         "missing_inside": px.apply(lambda s: int(s.loc[s.first_valid_index():s.last_valid_index()].isna().sum())),
                         "max_abs_logret": r.abs().max().round(3)})
