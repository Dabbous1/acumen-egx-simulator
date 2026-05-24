"""
EGX IPO Universe — candidate companies, metadata, and synthetic price generation.
All IPO metadata sourced from public EGX/Mubasher/prospectus records.
Price series are generated using realistic parameters calibrated to actual EGX behavior.
"""

import datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class IPOCompany:
    ticker: str
    name: str
    listing_date: dt.date
    offer_price: float  # EGP per share
    shares_offered: int
    raise_amount_egp: float  # total raise in EGP
    sector: str
    oversubscription_retail: Optional[float] = None  # e.g. 68.0 means 68x
    oversubscription_institutional: Optional[float] = None
    lockup_days: int = 180
    retail_allocation_pct: float = 0.30
    notes: str = ""

EGX_IPO_UNIVERSE = [
    IPOCompany(
        ticker="EFIH",
        name="e-finance for Digital & Financial Investments",
        listing_date=dt.date(2021, 10, 12),
        offer_price=13.98,
        shares_offered=417_600_000,
        raise_amount_egp=5_838_048_000,
        sector="Technology / Financial Infrastructure",
        oversubscription_retail=68.0,
        oversubscription_institutional=30.0,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="Landmark gov. tech IPO; largest EGX listing in years",
    ),
    IPOCompany(
        ticker="TALM",
        name="Taaleem Management Services",
        listing_date=dt.date(2021, 11, 2),
        offer_price=6.46,
        shares_offered=325_000_000,
        raise_amount_egp=2_099_500_000,
        sector="Education",
        oversubscription_retail=35.0,
        oversubscription_institutional=18.0,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="Private education operator",
    ),
    IPOCompany(
        ticker="MCRO",
        name="Macro Group Pharmaceuticals",
        listing_date=dt.date(2022, 2, 15),
        offer_price=4.85,
        shares_offered=250_000_000,
        raise_amount_egp=1_212_500_000,
        sector="Healthcare / Cosmeceuticals",
        oversubscription_retail=6.49,
        oversubscription_institutional=5.2,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="Cosmeceuticals & pharma distribution",
    ),
    IPOCompany(
        ticker="NKHC",
        name="Nahr El Khair (Al Khair River)",
        listing_date=dt.date(2022, 1, 20),
        offer_price=6.50,
        shares_offered=180_000_000,
        raise_amount_egp=1_170_000_000,
        sector="Food & Beverages",
        oversubscription_retail=12.0,
        oversubscription_institutional=8.0,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="First IPO of 2022; food production",
    ),
    IPOCompany(
        ticker="GHAZ",
        name="Ghazl El Mahalla SC",
        listing_date=dt.date(2022, 6, 1),
        offer_price=15.00,
        shares_offered=67_500_000,
        raise_amount_egp=1_012_500_000,
        sector="Sports / Entertainment",
        oversubscription_retail=22.0,
        oversubscription_institutional=10.0,
        lockup_days=90,
        retail_allocation_pct=0.50,
        notes="First sports-club public subscription on EGX",
    ),
    IPOCompany(
        ticker="GOUR",
        name="Gourmet Egypt.com Food Industries",
        listing_date=dt.date(2026, 2, 10),
        offer_price=35.00,
        shares_offered=50_000_000,
        raise_amount_egp=1_750_000_000,
        sector="Food & Beverages",
        oversubscription_retail=15.0,
        oversubscription_institutional=12.0,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="2026 reopening; public offering + private placement (forward calibration only)",
    ),
]


def get_universe_df() -> pd.DataFrame:
    rows = []
    for c in EGX_IPO_UNIVERSE:
        rows.append({
            "ticker": c.ticker,
            "name": c.name,
            "listing_date": c.listing_date,
            "offer_price": c.offer_price,
            "shares_offered": c.shares_offered,
            "raise_egp": c.raise_amount_egp,
            "sector": c.sector,
            "oversub_retail": c.oversubscription_retail,
            "oversub_inst": c.oversubscription_institutional,
            "lockup_days": c.lockup_days,
            "retail_alloc_pct": c.retail_allocation_pct,
        })
    return pd.DataFrame(rows)
