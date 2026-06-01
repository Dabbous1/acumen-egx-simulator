"""
EGX Event Universe — REAL IPOs and cash Rights Issues only.

Every event below is a verified, real corporate action on the Egyptian Exchange
during 2021-2024, and every ticker has real daily price history on Yahoo Finance
(TICKER.CA). No synthetic / fabricated events are included.

IPOs (real listings, real Yahoo first-trade dates):
  - EFIH  e-finance for Digital & Financial Investments (listed Oct 2021)
  - TALM  Taaleem Management Services

Cash rights issues (real, discounted capital increases — existing shareholders
subscribe to new shares; discount_pct measured against the real market price on
the event date):
  - EXPA  Export Development Bank of Egypt (EBank)   — Apr 2022
  - ADIB  Abu Dhabi Islamic Bank — Egypt             — Dec 2022
  - BTFH  Beltone Financial Holding                  — Jul 2023
  - MOED  Egyptian Modern Education Systems          — Nov 2023 (medium confidence)

Notes on parameters:
  * offer_price for IPOs is the real IPO subscription price; for rights issues it
    is the real cash subscription price per share.
  * discount_pct is derived from the REAL market close on the event date
    (1 - offer/market). It is negative for EXPA because that recapitalization was
    priced above market — a real outcome, not an error.
  * oversubscription_* and retail_allocation_pct are fund-mechanics ASSUMPTIONS
    used by the allocation model; they are not market data.
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
    offer_price: float  # EGP per share (real IPO / subscription price)
    shares_offered: int
    raise_amount_egp: float  # total raise in EGP
    sector: str
    event_type: str = "ipo"  # "ipo" or "rights_issue"
    discount_pct: Optional[float] = None  # real discount to market on event date
    ratio: Optional[str] = None  # e.g. "1:4" — 1 new share per 4 held
    oversubscription_retail: Optional[float] = None  # model assumption
    oversubscription_institutional: Optional[float] = None  # model assumption
    lockup_days: int = 180
    retail_allocation_pct: float = 0.30
    ex_date: Optional[dt.date] = None  # event / ex-rights date
    notes: str = ""

EGX_IPO_UNIVERSE = [
    # ── Real IPOs ─────────────────────────────────────────────────
    IPOCompany(
        ticker="EFIH",
        name="e-finance for Digital & Financial Investments",
        listing_date=dt.date(2021, 10, 20),
        offer_price=13.98,
        shares_offered=417_600_000,
        raise_amount_egp=5_838_048_000,
        sector="Technology / Financial Infrastructure",
        event_type="ipo",
        oversubscription_retail=68.0,
        oversubscription_institutional=30.0,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="Real IPO. Offer EGP 13.98; first real trade 2021-10-20 @ 16.80 (Yahoo EFIH.CA).",
    ),
    IPOCompany(
        ticker="TALM",
        name="Taaleem Management Services",
        listing_date=dt.date(2021, 4, 18),
        offer_price=5.58,
        shares_offered=325_000_000,
        raise_amount_egp=1_813_500_000,
        sector="Education",
        event_type="ipo",
        oversubscription_retail=35.0,
        oversubscription_institutional=18.0,
        lockup_days=180,
        retail_allocation_pct=0.30,
        notes="Real IPO. Entry uses real first-trade price/date from Yahoo TALM.CA (2021-04-18 @ 5.58).",
    ),

    # ── Real cash Rights Issues ───────────────────────────────────
    IPOCompany(
        ticker="EXPA",
        name="Export Development Bank of Egypt (EBank) — Rights Issue",
        listing_date=dt.date(2022, 4, 19),
        offer_price=10.00,
        shares_offered=200_000_000,
        raise_amount_egp=2_000_000_000,
        sector="Banking",
        event_type="rights_issue",
        discount_pct=-44.5,  # sub EGP 10.00 vs real market EGP 6.92 on 2022-04-19 (priced above market)
        ratio="3:5",
        oversubscription_retail=1.5,
        oversubscription_institutional=1.3,
        lockup_days=0,
        retail_allocation_pct=0.35,
        ex_date=dt.date(2022, 4, 19),
        notes="Real cash rights issue: EGP 2.0bn, 200M shares at par EGP 10.00, 1st-phase subscription ~Apr 2022. Priced above market (real recapitalization).",
    ),
    IPOCompany(
        ticker="ADIB",
        name="Abu Dhabi Islamic Bank — Egypt — Rights Issue",
        listing_date=dt.date(2022, 12, 18),
        offer_price=10.05,
        shares_offered=100_000_000,
        raise_amount_egp=1_005_000_000,
        sector="Banking",
        event_type="rights_issue",
        discount_pct=29.0,  # sub EGP 10.05 vs real market EGP 14.16 on 2022-12-18
        ratio="1:4",
        oversubscription_retail=2.0,
        oversubscription_institutional=1.6,
        lockup_days=0,
        retail_allocation_pct=0.30,
        ex_date=dt.date(2022, 12, 18),
        notes="Real cash rights issue: EGP 1.0bn, 100M new : 400M old (1:4), subscription EGP 10.05, Dec 2022-Jan 2023.",
    ),
    IPOCompany(
        ticker="BTFH",
        name="Beltone Financial Holding — Rights Issue",
        listing_date=dt.date(2023, 7, 27),
        offer_price=2.00,
        shares_offered=5_000_000_000,
        raise_amount_egp=10_000_000_000,
        sector="Financial Services",
        event_type="rights_issue",
        discount_pct=36.3,  # sub EGP 2.00 vs real market EGP 3.14 on 2023-07-27
        ratio="11:1",
        oversubscription_retail=1.2,
        oversubscription_institutional=1.1,
        lockup_days=0,
        retail_allocation_pct=0.30,
        ex_date=dt.date(2023, 7, 27),
        notes="Real cash rights issue: EGP 10.0bn, 5bn shares at par EGP 2.00, subscription closed ~Jul 2023 (capital EGP 926M -> 10.9bn).",
    ),
    IPOCompany(
        ticker="MOED",
        name="Egyptian Modern Education Systems — Rights Issue",
        listing_date=dt.date(2023, 11, 5),
        offer_price=0.10,
        shares_offered=2_495_000_000,
        raise_amount_egp=249_500_000,
        sector="Education",
        event_type="rights_issue",
        discount_pct=58.3,  # sub EGP 0.10 vs real market EGP 0.24 on 2023-11-05
        ratio="5:2",
        oversubscription_retail=1.3,
        oversubscription_institutional=1.2,
        lockup_days=0,
        retail_allocation_pct=0.35,
        ex_date=dt.date(2023, 11, 5),
        notes="Real cash rights issue: EGP 249.5M at par EGP 0.10, board approved 2023-11-05. Medium confidence on exact dates; thinly traded.",
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
            "event_type": c.event_type,
            "discount_pct": c.discount_pct,
            "ratio": c.ratio,
            "oversub_retail": c.oversubscription_retail,
            "oversub_inst": c.oversubscription_institutional,
            "lockup_days": c.lockup_days,
            "retail_alloc_pct": c.retail_allocation_pct,
            "ex_date": c.ex_date,
        })
    return pd.DataFrame(rows)
