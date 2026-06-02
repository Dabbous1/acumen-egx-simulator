"""
Market data — 100% REAL historic data. No synthetic / calibrated models.

Sources:
  - Yahoo Finance v8 chart API (requests + Mozilla UA), TICKER.CA:
      EFIH, TALM, EXPA, ADIB, BTFH, MOED  → daily OHLCV
  - EGX30 benchmark → the genuine EGX 30 index "since inception" daily series,
    exported from the Egyptian Exchange itself (egx.com.eg, Indices Data →
    EGX 30 → Index Since Inception (EGP)) and stored locally in
    data/egx30_real.csv. Loaded under key "egx30" (the engine's benchmark
    slot). Index points; the engine rebases it for comparison, so the raw
    level is used directly.
  - Yahoo USDEGP=X → daily USD/EGP rate.
  - World Bank API (Egypt, annual, real), monthly-interpolated:
      FP.CPI.TOTL → consumer price index (key "cpi")
      FR.INR.DPST → deposit interest rate, used as the risk-free / T-bill
                    proxy (key "tbill", column "tbill_annual").

The stock/FX/macro series are fetched live; the EGX30 benchmark is read from
the committed CSV (the EGX site is WAF-protected and not directly scriptable).
There is no synthetic fallback — any failure raises so that no fabricated data
silently enters results.
"""

import datetime as dt
import json
import os
import urllib.request
import numpy as np
import pandas as pd
from data.ipo_universe import EGX_IPO_UNIVERSE

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_EGX30_CSV = os.path.join(_DATA_DIR, "egx30_real.csv")

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Local ticker -> Yahoo symbol (all real, all have .CA history)
_YAHOO_TICKERS: dict[str, str] = {
    "EFIH": "EFIH.CA",
    "TALM": "TALM.CA",
    "EXPA": "EXPA.CA",
    "ADIB": "ADIB.CA",
    "BTFH": "BTFH.CA",
    "MOED": "MOED.CA",
}

_START = dt.date(2021, 1, 1)
_END = dt.date(2025, 12, 31)


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Yahoo v8 chart fetcher
# ---------------------------------------------------------------------------

def _fetch_yahoo_chart(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Pull real daily OHLCV from the Yahoo v8 chart endpoint."""
    p1 = int(dt.datetime(start.year, start.month, start.day).timestamp())
    p2 = int(dt.datetime(end.year, end.month, end.day).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    raw = _http_get(url)
    d = json.loads(raw)
    result = d["chart"]["result"]
    if not result:
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {d['chart'].get('error')}")
    r = result[0]
    ts = r.get("timestamp")
    if not ts:
        raise RuntimeError(f"Yahoo returned no timestamps for {symbol}")
    q = r["indicators"]["quote"][0]
    dates = [dt.datetime.utcfromtimestamp(t).date() for t in ts]

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": q.get("open"),
        "high": q.get("high"),
        "low": q.get("low"),
        "close": q.get("close"),
        "volume": q.get("volume"),
    })
    # Drop rows with no close (market holidays returned as null)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _fetch_stock(local_ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    sym = _YAHOO_TICKERS[local_ticker]
    df = _fetch_yahoo_chart(sym, start, end)
    out = pd.DataFrame({
        "date": df["date"],
        "ticker": local_ticker,
        "open": df["open"].round(2),
        "high": df["high"].round(2),
        "low": df["low"].round(2),
        "close": df["close"].round(2),
        "volume": df["volume"].fillna(0).astype("int64"),
    })
    print(f"  {local_ticker}: {len(out)} real days from Yahoo ({sym})")
    return out.reset_index(drop=True)


def _fetch_usdegp(start: dt.date, end: dt.date) -> pd.DataFrame:
    df = _fetch_yahoo_chart("USDEGP=X", start, end)
    out = pd.DataFrame({"date": df["date"], "usdegp": df["close"].round(4)})
    print(f"  USD/EGP: {len(out)} real days from Yahoo (USDEGP=X)")
    return out.reset_index(drop=True)


def _load_egx30_real(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Genuine EGX 30 index daily closes, sourced from the Egyptian Exchange
    (egx.com.eg) and committed to data/egx30_real.csv. Stored as the benchmark
    ('egx30' slot) in raw index points; the engine rebases it for comparison."""
    if not os.path.exists(_EGX30_CSV):
        raise RuntimeError(f"Real EGX30 file missing: {_EGX30_CSV}")
    df = pd.read_csv(_EGX30_CSV, parse_dates=["date"])
    df = df.rename(columns={"close": "egx30"})[["date", "egx30"]]
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    out = df.loc[mask].dropna().reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError("Real EGX30 file produced no rows in range")
    print(f"  Benchmark (real EGX 30 index): {len(out)} days from EGX (egx.com.eg)")
    return out


# ---------------------------------------------------------------------------
# World Bank macro (annual real, monthly-interpolated)
# ---------------------------------------------------------------------------

def _fetch_worldbank(indicator: str) -> dict[int, float]:
    """Return {year: value} for an Egypt World Bank indicator."""
    url = (
        f"https://api.worldbank.org/v2/country/EGY/indicator/{indicator}"
        f"?format=json&date=2020:2025&per_page=100"
    )
    raw = _http_get(url)
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        raise RuntimeError(f"World Bank returned no data for {indicator}")
    out: dict[int, float] = {}
    for row in data[1]:
        if row.get("value") is not None:
            out[int(row["date"])] = float(row["value"])
    if not out:
        raise RuntimeError(f"World Bank indicator {indicator} empty")
    return out


def _annual_to_monthly(annual: dict[int, float], start: dt.date, end: dt.date,
                       col: str, log_interp: bool) -> pd.DataFrame:
    """Interpolate annual year-end values to a monthly series.

    Annual value is anchored at mid-year (Jul 1) of each year; months between
    anchors are linearly interpolated (log-linear for index levels like CPI so
    growth compounds smoothly).
    """
    anchors_x, anchors_y = [], []
    for yr in sorted(annual):
        anchors_x.append(pd.Timestamp(yr, 7, 1))
        v = annual[yr]
        anchors_y.append(np.log(v) if log_interp else v)

    months = pd.date_range(start, end, freq="MS")
    x_num = np.array([a.value for a in anchors_x], dtype=float)
    m_num = np.array([m.value for m in months], dtype=float)
    interp = np.interp(m_num, x_num, anchors_y)
    if log_interp:
        interp = np.exp(interp)

    return pd.DataFrame({"date": months, col: np.round(interp, 4)})


def _fetch_cpi(start: dt.date, end: dt.date) -> pd.DataFrame:
    annual = _fetch_worldbank("FP.CPI.TOTL")
    df = _annual_to_monthly(annual, start, end, "cpi", log_interp=True)
    print(f"  CPI: {len(df)} months from World Bank (FP.CPI.TOTL, {min(annual)}-{max(annual)})")
    return df


def _fetch_tbill(start: dt.date, end: dt.date) -> pd.DataFrame:
    annual = _fetch_worldbank("FR.INR.DPST")  # deposit rate, % per annum
    df = _annual_to_monthly(annual, start, end, "tbill_annual", log_interp=False)
    df["tbill_annual"] = (df["tbill_annual"] / 100.0).round(4)  # % -> fraction
    print(f"  Risk-free (deposit rate): {len(df)} months from World Bank (FR.INR.DPST)")
    return df


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_full_dataset() -> dict[str, pd.DataFrame]:
    """Build the complete REAL dataset. Raises on any fetch failure."""
    start, end = _START, _END

    print("Loading REAL stock prices from Yahoo Finance...")
    stock_frames = []
    for co in EGX_IPO_UNIVERSE:
        df = _fetch_stock(co.ticker, co.listing_date, end)
        if len(df) == 0:
            raise RuntimeError(f"No real price data returned for {co.ticker}")
        stock_frames.append(df)

    print("Loading REAL macro / FX / benchmark data...")
    usdegp = _fetch_usdegp(start, end)
    egx30 = _load_egx30_real(start, end)
    cpi = _fetch_cpi(start, end)
    tbill = _fetch_tbill(start, end)

    return {
        "prices": pd.concat(stock_frames, ignore_index=True),
        "egx30": egx30,
        "usdegp": usdegp,
        "cpi": cpi,
        "tbill": tbill,
    }
