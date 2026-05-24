"""
Market data generator — produces realistic synthetic price series, FX, CPI, T-bill rates,
and EGX30 benchmark calibrated to actual EGX market behavior (2021-2025).
"""

import datetime as dt
import numpy as np
import pandas as pd
from data.ipo_universe import EGX_IPO_UNIVERSE, IPOCompany


def _business_days(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def generate_stock_prices(
    company: IPOCompany,
    end_date: dt.date = dt.date(2025, 12, 31),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate realistic daily OHLCV for a single stock from its listing date."""
    if seed is None:
        seed = hash(company.ticker) % (2**31)
    rng = np.random.RandomState(seed)

    dates = _business_days(company.listing_date, end_date)
    if len(dates) == 0:
        return pd.DataFrame()

    n = len(dates)
    price = company.offer_price

    # IPO-specific dynamics: first-day pop, early volatility decay, regime shifts
    first_day_pop = 1.0 + rng.uniform(0.05, 0.35)
    daily_vol_base = rng.uniform(0.018, 0.035)

    prices = np.zeros(n)
    volumes = np.zeros(n)

    # Day 0: listing day pop
    prices[0] = price * first_day_pop
    volumes[0] = company.shares_offered * rng.uniform(0.08, 0.20)

    # Regime parameters calibrated to EGX reality
    # 2022 H2: EGP devaluation shock → sell-off
    # 2023: recovery in some names, continued weakness in others
    # 2024: mixed, low volume environment
    for i in range(1, n):
        d = dates[i].date()
        days_since_ipo = (d - company.listing_date).days

        # Volatility decays from IPO excitement
        vol_decay = max(0.6, 1.0 - days_since_ipo / 500)
        vol = daily_vol_base * vol_decay

        # Regime drift adjustments
        drift = 0.0
        if dt.date(2022, 10, 1) <= d <= dt.date(2023, 3, 31):
            drift = -0.0012  # devaluation sell-off
            vol *= 1.4
        elif dt.date(2024, 3, 1) <= d <= dt.date(2024, 6, 30):
            drift = -0.0008  # second devaluation wave
            vol *= 1.3
        elif dt.date(2023, 6, 1) <= d <= dt.date(2023, 12, 31):
            drift = 0.0004  # partial recovery
        elif dt.date(2025, 1, 1) <= d:
            drift = 0.0003  # stabilization

        # Sector tilt
        if company.sector in ("Technology / Financial Infrastructure",):
            drift += 0.0002
        elif company.sector in ("Sports / Entertainment",):
            drift -= 0.0001

        ret = drift + vol * rng.randn()
        prices[i] = prices[i - 1] * np.exp(ret)
        prices[i] = max(prices[i], 0.50)  # floor

        # Volume with decay and randomness
        avg_vol = company.shares_offered * 0.005
        vol_mult = max(0.3, 1.0 - days_since_ipo / 800)
        volumes[i] = max(1000, avg_vol * vol_mult * rng.lognormal(0, 0.5))

    # Build OHLCV
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


def generate_egx30(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """EGX30 benchmark — calibrated to actual index trajectory."""
    rng = np.random.RandomState(42)
    dates = _business_days(start, end)
    n = len(dates)

    level = 10800.0  # approximate EGX30 start of 2021
    levels = np.zeros(n)

    for i in range(n):
        d = dates[i].date()
        drift = 0.0002
        vol = 0.012

        if dt.date(2021, 6, 1) <= d <= dt.date(2022, 4, 30):
            drift = 0.0005  # bull run to ~12k
        elif dt.date(2022, 10, 1) <= d <= dt.date(2023, 1, 31):
            drift = -0.0006  # devaluation shock
            vol = 0.018
        elif dt.date(2023, 2, 1) <= d <= dt.date(2024, 2, 28):
            drift = 0.0008  # strong rally to ~30k
            vol = 0.014
        elif dt.date(2024, 3, 1) <= d <= dt.date(2024, 6, 30):
            drift = -0.0003  # correction
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
    """USD/EGP rate series — captures the two major devaluations."""
    rng = np.random.RandomState(99)
    dates = _business_days(start, end)
    n = len(dates)

    rate = 15.70  # start of 2021
    rates = np.zeros(n)

    for i in range(n):
        d = dates[i].date()
        noise = rng.randn() * 0.001

        if d < dt.date(2022, 3, 1):
            rate *= np.exp(0.0001 + noise)  # stable ~15.7
        elif d < dt.date(2022, 10, 27):
            rate *= np.exp(0.0015 + noise)  # gradual to ~19.7
        elif d < dt.date(2023, 1, 12):
            rate *= np.exp(0.004 + noise)  # first deval shock → ~30
        elif d < dt.date(2024, 3, 6):
            rate *= np.exp(0.0002 + noise)  # controlled crawl
        elif d < dt.date(2024, 4, 15):
            rate *= np.exp(0.006 + noise)  # March 2024 float → ~49
        else:
            rate *= np.exp(0.0001 + noise * 0.5)  # stabilization ~49-51

        rates[i] = rate

    return pd.DataFrame({"date": dates, "usdegp": np.round(rates, 4)})


def generate_cpi(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """Monthly CPI index — captures Egypt's inflation surge."""
    months = pd.date_range(start, end, freq="MS")
    cpi = 100.0
    values = []

    for m in months:
        d = m.date()
        if d < dt.date(2022, 3, 1):
            monthly_inf = 0.005  # ~6% annualized
        elif d < dt.date(2022, 10, 1):
            monthly_inf = 0.012  # ~15%
        elif d < dt.date(2023, 9, 1):
            monthly_inf = 0.025  # ~35% peak
        elif d < dt.date(2024, 6, 1):
            monthly_inf = 0.028  # ~40% peak
        else:
            monthly_inf = 0.010  # cooling to ~12%

        cpi *= (1 + monthly_inf)
        values.append({"date": m, "cpi": round(cpi, 2)})

    return pd.DataFrame(values)


def generate_tbill_rate(
    start: dt.date = dt.date(2021, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> pd.DataFrame:
    """Egyptian T-bill annualized yield — tracks CBE policy rate."""
    months = pd.date_range(start, end, freq="MS")
    values = []

    for m in months:
        d = m.date()
        if d < dt.date(2022, 3, 1):
            rate = 0.125  # 12.5%
        elif d < dt.date(2022, 10, 1):
            rate = 0.145
        elif d < dt.date(2023, 3, 1):
            rate = 0.175
        elif d < dt.date(2023, 8, 1):
            rate = 0.195
        elif d < dt.date(2024, 3, 1):
            rate = 0.225
        elif d < dt.date(2024, 9, 1):
            rate = 0.265  # peak ~26.5%
        else:
            rate = 0.225  # easing

        values.append({"date": m, "tbill_annual": rate})

    return pd.DataFrame(values)


def build_full_dataset() -> dict[str, pd.DataFrame]:
    """Build the complete dataset for the simulation."""
    end = dt.date(2025, 12, 31)

    stock_frames = []
    for co in EGX_IPO_UNIVERSE:
        if co.listing_date <= end:
            df = generate_stock_prices(co, end)
            if len(df) > 0:
                stock_frames.append(df)

    return {
        "prices": pd.concat(stock_frames, ignore_index=True),
        "egx30": generate_egx30(end=end),
        "usdegp": generate_usdegp(end=end),
        "cpi": generate_cpi(end=end),
        "tbill": generate_tbill_rate(end=end),
    }
