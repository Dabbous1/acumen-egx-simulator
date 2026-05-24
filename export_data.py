"""Export raw market data as JSON for the browser-based simulation engine."""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from data.ipo_universe import EGX_IPO_UNIVERSE
from data.market_data import build_full_dataset

dataset = build_full_dataset()

# Stock prices: group by ticker, only keep date+close+volume
prices_by_ticker = {}
for _, row in dataset["prices"].iterrows():
    t = row["ticker"]
    if t not in prices_by_ticker:
        prices_by_ticker[t] = []
    prices_by_ticker[t].append({
        "d": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
        "c": round(float(row["close"]), 2),
        "v": int(row["volume"]),
    })

# EGX30
egx30 = [{"d": str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
           "v": round(float(r["egx30"]), 2)}
          for _, r in dataset["egx30"].iterrows()]

# USD/EGP
usdegp = [{"d": str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
            "v": round(float(r["usdegp"]), 4)}
           for _, r in dataset["usdegp"].iterrows()]

# CPI
cpi = [{"d": str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
        "v": round(float(r["cpi"]), 2)}
       for _, r in dataset["cpi"].iterrows()]

# T-bill
tbill = [{"d": str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
          "v": float(r["tbill_annual"])}
         for _, r in dataset["tbill"].iterrows()]

# Universe
universe = []
for co in EGX_IPO_UNIVERSE:
    universe.append({
        "ticker": co.ticker,
        "name": co.name,
        "listing_date": str(co.listing_date),
        "offer_price": co.offer_price,
        "shares_offered": co.shares_offered,
        "raise_egp": co.raise_amount_egp,
        "sector": co.sector,
        "oversub_retail": co.oversubscription_retail,
        "oversub_inst": co.oversubscription_institutional,
        "lockup_days": co.lockup_days,
        "retail_alloc_pct": co.retail_allocation_pct,
    })

output = {
    "prices": prices_by_ticker,
    "egx30": egx30,
    "usdegp": usdegp,
    "cpi": cpi,
    "tbill": tbill,
    "universe": universe,
}

out_path = os.path.join(os.path.dirname(__file__), "output", "market_data.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(",", ":"))

size = os.path.getsize(out_path)
print(f"Exported to {out_path} ({size/1024:.0f} KB)")
for t, prices in prices_by_ticker.items():
    print(f"  {t}: {len(prices)} days")
