"""
Market data — real historic OHLCV from Yahoo Finance where available,
synthetic fallback for tickers/series not covered.

Real data (Yahoo Finance):
  - EFIH.CA, TALM.CA, MCRO.CA  → daily OHLCV
  - USDEGP=X                    → daily USD/EGP rate

Synthetic (calibrated models):
  - NKHC, GHAZ, GOUR            → no Yahoo coverage
  - EGX30 benchmark              → no reliable Yahoo symbol
  - CPI, T-bill yields           → macro data, not on Yahoo
"""

import datetime as dt
import numpy as np
import pandas as pd
from data.ipo_universe import EGX_IPO_UNIVERSE, IPOCompany

# Yahoo Finance ticker mapping for stocks with real data
_YAHOO_TICKERS: dict[str, str] = {
    "EFIH": "EFIH.CA",
    "TALM": "TALM.CA",
    "MCRO": "MCRO.CA",
}


def _business_days(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


# ---------------------------------------------------------------------------
# Real data fetchers
# ---------------------------------------------------------------------------

def _fetch_yahoo_ohlcv(
    yahoo_ticker: str,
    local_ticker: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame | None:
    """Pull real OHLCV from Yahoo Finance. Returns None on failure."""
    try:
        import yfinance as yf
        df = yf.download(
            yahoo_ticker,
            start=str(start),
            end=str(end + dt.timedelta(days=1)),
            progress=False,
            auto_adjust=True,
        )
        if df is None or len(df) == 0:
            return None

        # yfinance returns MultiIndex columns when downloading single ticker too
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        out = pd.DataFrame({
            "date": df.index,
            "ticker": local_ticker,
            "open": df["Open"].round(2).values,
            "high": df["High"].round(2).values,
            "low": df["Low"].round(2).values,
            "close": df["Close"].round(2).values,
            "volume": df["Volume"].astype(int).values,
        })
        return out.reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Yahoo fetch failed for {yahoo_ticker}: {e}")
        return None


def _fetch_yahoo_usdegp(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame | None:
    """Pull real USD/EGP daily rate from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download(
            "USDEGP=X",
            start=str(start),
            end=str(end + dt.timedelta(days=1)),
            progress=False,
            auto_adjust=True,
        )
        if df is None or len(df) == 0:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        out = pd.DataFrame({
            "date": df.index,
            "usdegp": df["Close"].round(4).values,
        })
        return out.reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Yahoo fetch failed for USDEGP=X: {e}")
        return None


# ---------------------------------------------------------------------------
# Synthetic generators (fallback)
# ---------------------------------------------------------------------------

def generate_stock_prices_synthetic(
    company: IPOCompany,
    end_date: dt.date = dt.date(2025, 12, 31),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV — used only when real data is unavailable."""
    if seed is None:
        seed = hash(company.ticker) % (2**31)
    rng = np.random.RandomState(seed)

    dates = _business_days(company.listing_date, end_date)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)
    price = company.offer_price

    first_day_pop = 1.0 + rng.uniform(0.05, 0.35)
    daily_vol_base = rng.uniform(0.018, 0.035)

    prices = np.zeros(n)
    volumes = np.zeros(n)

    prices[0] = price * first_day_pop
    volumes[0] = company.shares_offered * rng.uniform(0.08, 0.20)

    for i in range(1, n):
        d = dates[i].date()
        days_since_ipo = (d - company.listing_date).days

        vol_decay = max(0.6, 1.0 - days_since_ipo / 500)
        vol = daily_vol_base * vol_decay

        drift = 0.0
        if dt.date(2022, 10, 1) <= d <= dt.date(2023, 3, 31):
            drift = -0.0012
            vol *= 1.4
        elif dt.date(2024, 3, 1) <= d <= dt.date(2024, 6, 30):
            drift = -0.0008
            vol *= 1.3
        elif dt.date(2023, 6, 1) <= d <= dt.date(2023, 12, 31):
            drift = 0.0004
        elif dt.date(2025, 1, 1) <= d:
            drift = 0.0003

        if company.sector in ("Technology / Financial Infrastructure",):
            drift += 0.0002
        elif company.sector in ("Sports / Entertainment",):
            drift -= 0.0001

        ret = drift + vol * rng.randn()
        prices[i] = prices[i - 1] * np.exp(ret)
        prices[i] = max(prices[i], 0.50)

        avg_vol = company.shares_offered * 0.005
        vol_mult = max(0.3, 1.0 - days_since_ipo / 800)
        volumes[i] = max(1000, avg_vol * vol_mult * rng.lognormal(0, 0.5))

    highs = prices * (1 + rng.uniform(0.005, 0.025, n))
    lows = prices * (1 - rng.uniform(0.005, 0.025, n))
    opens = np.roll(prices, 1)
    opens[0] = company.offer_price

    return pd.DataFrame({
        "date": dates,
        "ticker": company.ticker,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(prices, 2),
        "volume": volumes.astype(int),
    })


def generate_stock_prices(
    company: IPOCompany,
    end_date: dt.date = dt.date(2025, 12, 31),
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Get daily OHLCV for a stock.
    Tries real Yahoo Finance data first; falls back to synthetic if unavailable.
    """
    yahoo_sym = _YAHOO_TICKERS.get(company.ticker)
    if yahoo_sym:
        real = _fetch_yahoo_ohlcv(yahoo_sym, company.ticker, company.listing_date, end_date)
        if real is not None and len(real) > 0:
            print(f"  {company.ticker}: loaded {len(real)} days of real data from Yahoo ({yahoo_sym})")
            return real

    print(f"  {company.ticker}: using synthetic data (no Yahoo coverage)")
    return generate_stock_prices_synthetic(company, end_date, seed)


def generate_egx30(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """EGX30 benchmark — synthetic, calibrated to actual index trajectory."""
    rng = np.random.RandomState(42)
    dates = _business_days(start, end)
    n = len(dates)

    level = 10800.0
    levels = np.zeros(n)

    for i in range(n):
        d = dates[i].date()
        drift = 0.0002
        vol = 0.012

        if dt.date(2021, 6, 1) <= d <= dt.date(2022, 4, 30):
            drift = 0.0005
        elif dt.date(2022, 10, 1) <= d <= dt.date(2023, 1, 31):
            drift = -0.0006
            vol = 0.018
        elif dt.date(2023, 2, 1) <= d <= dt.date(2024, 2, 28):
            drift = 0.0008
            vol = 0.014
        elif dt.date(2024, 3, 1) <= d <= dt.date(2024, 6, 30):
            drift = -0.0003
            vol = 0.016
        elif dt.date(2024, 7, 1) <= d:
            drift = 0.0003
            vol = 0.011

        ret = drift + vol * rng.randn()
        level = level * np.exp(ret)
        levels[i] = level

    return pd.DataFrame({"date": dates, "egx30": np.round(levels, 2)})


def generate_usdegp(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """
    USD/EGP daily rate.
    Tries real Yahoo Finance data first; falls back to synthetic if unavailable.
    """
    real = _fetch_yahoo_usdegp(start, end)
    if real is not None and len(real) > 0:
        print(f"  USD/EGP: loaded {len(real)} days of real data from Yahoo (USDEGP=X)")
        return real

    print("  USD/EGP: using synthetic data (Yahoo unavailable)")
    return _generate_usdegp_synthetic(start, end)


def _generate_usdegp_synthetic(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """Synthetic USD/EGP — fallback when Yahoo is unavailable."""
    rng = np.random.RandomState(99)
    dates = _business_days(start, end)
    n = len(dates)

    rate = 15.70
    rates = np.zeros(n)

    for i in range(n):
        d = dates[i].date()
        noise = rng.randn() * 0.001

        if d < dt.date(2022, 3, 1):
            rate *= np.exp(0.0001 + noise)
        elif d < dt.date(2022, 10, 27):
            rate *= np.exp(0.0015 + noise)
        elif d < dt.date(2023, 1, 12):
            rate *= np.exp(0.004 + noise)
        elif d < dt.date(2024, 3, 6):
            rate *= np.exp(0.0002 + noise)
        elif d < dt.date(2024, 4, 15):
            rate *= np.exp(0.006 + noise)
        else:
            rate *= np.exp(0.0001 + noise * 0.5)

        rates[i] = rate

    return pd.DataFrame({"date": dates, "usdegp": np.round(rates, 4)})


def generate_cpi(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """Monthly CPI index — synthetic, calibrated to Egypt's inflation trajectory."""
    months = pd.date_range(start, end, freq="MS")
    cpi = 100.0
    values = []

    for m in months:
        d = m.date()
        if d < dt.date(2022, 3, 1):
            monthly_inf = 0.005
        elif d < dt.date(2022, 10, 1):
            monthly_inf = 0.012
        elif d < dt.date(2023, 9, 1):
            monthly_inf = 0.025
        elif d < dt.date(2024, 6, 1):
            monthly_inf = 0.028
        else:
            monthly_inf = 0.010

        cpi *= (1 + monthly_inf)
        values.append({"date": m, "cpi": round(cpi, 2)})

    return pd.DataFrame(values)


def generate_tbill_rate(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """Egyptian T-bill annualized yield — synthetic, tracks CBE policy rate."""
    months = pd.date_range(start, end, freq="MS")
    values = []

    for m in months:
        d = m.date()
        if d < dt.date(2022, 3, 1):
            rate = 0.125
        elif d < dt.date(2022, 10, 1):
            rate = 0.145
        elif d < dt.date(2023, 3, 1):
            rate = 0.175
        elif d < dt.date(2023, 8, 1):
            rate = 0.195
        elif d < dt.date(2024, 3, 1):
            rate = 0.225
        elif d < dt.date(2024, 9, 1):
            rate = 0.265
        else:
            rate = 0.225

        values.append({"date": m, "tbill_annual": rate})

    return pd.DataFrame(values)


def build_full_dataset() -> dict[str, pd.DataFrame]:
    """
    Build the complete dataset for the simulation.
    Uses real Yahoo Finance data where available, synthetic fallback otherwise.
    """
    end = dt.date(2025, 12, 31)

    print("Loading market data (real + synthetic hybrid)...")
    stock_frames = []
    for co in EGX_IPO_UNIVERSE:
        if co.listing_date <= end:
            df = generate_stock_prices(co, end)
            if len(df) > 0:
                stock_frames.append(df)

    print("Loading macro data...")
    return {
        "prices": pd.concat(stock_frames, ignore_index=True),
        "egx30": generate_egx30(end=end),
        "usdegp": generate_usdegp(end=end),
        "cpi": generate_cpi(end=end),
        "tbill": generate_tbill_rate(end=end),
    }
