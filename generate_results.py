"""
Run the full simulation grid across all lever combinations and export results as JSON
for the web simulator to consume.
"""

import sys
import os
import json
import datetime as dt
import itertools

sys.path.insert(0, os.path.dirname(__file__))

from data.ipo_universe import EGX_IPO_UNIVERSE, get_universe_df
from data.market_data import build_full_dataset
from engine.simulator import SimulationEngine, SimConfig
from engine.metrics import compute_metrics, generate_verdict


def build_scenario_grid() -> list[dict]:
    """Define the lever combinations to test."""
    models = ["B", "C"]
    inception_dates = [dt.date(2022, 1, 1), dt.date(2022, 7, 1), dt.date(2023, 1, 1)]

    # Model B lever options
    b_levers = {
        "entry_timing": ["day1", "day5", "day30"],
        "holding_period_months": [None, 12, 24],
        "weighting": ["equal", "offersize"],
        "position_cap": [None, 0.15, 0.20],
        "downside_protection": ["none", "stop15", "stop20"],
        "tbill_buffer_pct": [0.0, 0.10, 0.20],
        "rebalance_freq": ["quarterly", "on_new_listing"],
        "sector_cap": [None, 0.30],
        "lockup_avoidance": [False, True],
    }

    # Model C lever options
    c_levers = {
        "flip_strategy": ["flip_day1", "flip_week1", "hold"],
        "holding_period_months": [None, 6, 12],
        "tbill_buffer_pct": [0.0, 0.10, 0.20],
        "downside_protection": ["none", "stop15"],
        "allocation_track": ["retail", "institutional"],
    }

    scenarios = []

    # Curated Model B scenarios (testing key lever interactions, not full cartesian)
    b_key_combos = [
        # Baseline
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # Later entry
        {"entry_timing": "day5", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        {"entry_timing": "day30", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # Stop-loss variants
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "stop15", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "stop20", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # T-bill buffer variants
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.0,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.20,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # Offer-size weighting
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "offersize",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # No position cap
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": None, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # Sector cap
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": 0.30, "lockup_avoidance": False},
        # Lockup avoidance
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": True},
        # Holding period variants
        {"entry_timing": "day1", "holding_period_months": 12, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        {"entry_timing": "day1", "holding_period_months": 24, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "quarterly", "sector_cap": None, "lockup_avoidance": False},
        # Rebalance on new listing
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "none", "tbill_buffer_pct": 0.10,
         "rebalance_freq": "on_new_listing", "sector_cap": None, "lockup_avoidance": False},
        # Combined best: later entry + stop + T-bill buffer
        {"entry_timing": "day5", "holding_period_months": None, "weighting": "equal",
         "position_cap": 0.15, "downside_protection": "stop20", "tbill_buffer_pct": 0.20,
         "rebalance_freq": "quarterly", "sector_cap": 0.30, "lockup_avoidance": True},
        # Aggressive: no caps, no protection
        {"entry_timing": "day1", "holding_period_months": None, "weighting": "offersize",
         "position_cap": None, "downside_protection": "none", "tbill_buffer_pct": 0.0,
         "rebalance_freq": "on_new_listing", "sector_cap": None, "lockup_avoidance": False},
    ]

    c_key_combos = [
        # Baseline flip day1
        {"flip_strategy": "flip_day1", "holding_period_months": None, "tbill_buffer_pct": 0.0,
         "downside_protection": "none", "allocation_track": "retail"},
        # Flip week1
        {"flip_strategy": "flip_week1", "holding_period_months": None, "tbill_buffer_pct": 0.0,
         "downside_protection": "none", "allocation_track": "retail"},
        # Hold 6m
        {"flip_strategy": "hold", "holding_period_months": 6, "tbill_buffer_pct": 0.0,
         "downside_protection": "none", "allocation_track": "retail"},
        # Hold 12m
        {"flip_strategy": "hold", "holding_period_months": 12, "tbill_buffer_pct": 0.0,
         "downside_protection": "none", "allocation_track": "retail"},
        # T-bill buffer variants
        {"flip_strategy": "flip_day1", "holding_period_months": None, "tbill_buffer_pct": 0.10,
         "downside_protection": "none", "allocation_track": "retail"},
        {"flip_strategy": "flip_day1", "holding_period_months": None, "tbill_buffer_pct": 0.20,
         "downside_protection": "none", "allocation_track": "retail"},
        # Institutional track
        {"flip_strategy": "flip_day1", "holding_period_months": None, "tbill_buffer_pct": 0.0,
         "downside_protection": "none", "allocation_track": "institutional"},
        # With stop-loss
        {"flip_strategy": "hold", "holding_period_months": 12, "tbill_buffer_pct": 0.10,
         "downside_protection": "stop15", "allocation_track": "retail"},
        # Best combo
        {"flip_strategy": "hold", "holding_period_months": 6, "tbill_buffer_pct": 0.20,
         "downside_protection": "stop15", "allocation_track": "retail"},
    ]

    for inception in inception_dates:
        for combo in b_key_combos:
            scenarios.append({"model": "B", "inception_date": inception, **combo})
        for combo in c_key_combos:
            scenarios.append({"model": "C", "inception_date": inception, **combo})

    return scenarios


def _config_label(s: dict) -> str:
    model = s["model"]
    parts = [f"Model {model}"]
    parts.append(f"inception={s['inception_date']}")

    if model == "B":
        parts.append(f"entry={s['entry_timing']}")
        if s.get("holding_period_months"):
            parts.append(f"hold={s['holding_period_months']}m")
        parts.append(f"wt={s['weighting']}")
        if s.get("position_cap"):
            parts.append(f"cap={int(s['position_cap']*100)}%")
        if s["downside_protection"] != "none":
            parts.append(f"prot={s['downside_protection']}")
        parts.append(f"tbill={int(s['tbill_buffer_pct']*100)}%")
        parts.append(f"rebal={s['rebalance_freq']}")
        if s.get("sector_cap"):
            parts.append(f"secCap={int(s['sector_cap']*100)}%")
        if s.get("lockup_avoidance"):
            parts.append("lockupAvoid")
    else:
        parts.append(f"flip={s['flip_strategy']}")
        if s.get("holding_period_months"):
            parts.append(f"hold={s['holding_period_months']}m")
        parts.append(f"tbill={int(s['tbill_buffer_pct']*100)}%")
        parts.append(f"track={s['allocation_track']}")
        if s["downside_protection"] != "none":
            parts.append(f"prot={s['downside_protection']}")

    return " | ".join(parts)


def generate_recommendations(all_results: list[dict]) -> list[dict]:
    """Analyze all results and produce ranked feature recommendations."""
    recs = []

    # Group by model
    for model in ["B", "C"]:
        model_results = [r for r in all_results if r["config"]["model"] == model]
        if not model_results:
            continue

        # Find best by Sharpe
        best_sharpe = max(model_results, key=lambda r: r["metrics"]["sharpe"])
        # Find best by real return
        best_real = max(model_results, key=lambda r: r["metrics"]["cagr_real"])
        # Find lowest drawdown among positive-return configs
        positive = [r for r in model_results if r["metrics"]["total_return_real"] > 0]
        best_dd = min(positive, key=lambda r: abs(r["metrics"]["max_drawdown"])) if positive else None

        recs.append({
            "model": model,
            "best_sharpe": {
                "label": best_sharpe["label"],
                "sharpe": best_sharpe["metrics"]["sharpe"],
                "cagr_real": best_sharpe["metrics"]["cagr_real"],
                "max_dd": best_sharpe["metrics"]["max_drawdown"],
            },
            "best_real_return": {
                "label": best_real["label"],
                "cagr_real": best_real["metrics"]["cagr_real"],
                "sharpe": best_real["metrics"]["sharpe"],
            },
            "lowest_drawdown": {
                "label": best_dd["label"],
                "max_dd": best_dd["metrics"]["max_drawdown"],
                "cagr_real": best_dd["metrics"]["cagr_real"],
            } if best_dd else None,
        })

    # Feature-level analysis: measure each lever's average impact
    feature_impacts = []

    # Entry timing impact (Model B)
    b_results = [r for r in all_results if r["config"]["model"] == "B"]
    for lever, options in [
        ("entry_timing", ["day1", "day5", "day30"]),
        ("downside_protection", ["none", "stop15", "stop20"]),
        ("tbill_buffer_pct", [0.0, 0.10, 0.20]),
        ("weighting", ["equal", "offersize"]),
        ("lockup_avoidance", [False, True]),
    ]:
        for opt in options:
            matching = [r for r in b_results if r["config"].get(lever) == opt]
            if matching:
                avg_sharpe = sum(r["metrics"]["sharpe"] for r in matching) / len(matching)
                avg_real = sum(r["metrics"]["cagr_real"] for r in matching) / len(matching)
                avg_dd = sum(abs(r["metrics"]["max_drawdown"]) for r in matching) / len(matching)
                feature_impacts.append({
                    "model": "B", "lever": lever, "option": str(opt),
                    "avg_sharpe": round(avg_sharpe, 3),
                    "avg_cagr_real": round(avg_real, 2),
                    "avg_max_dd": round(avg_dd, 2),
                    "n_scenarios": len(matching),
                })

    # Model C feature impacts
    c_results = [r for r in all_results if r["config"]["model"] == "C"]
    for lever, options in [
        ("flip_strategy", ["flip_day1", "flip_week1", "hold"]),
        ("tbill_buffer_pct", [0.0, 0.10, 0.20]),
        ("allocation_track", ["retail", "institutional"]),
    ]:
        for opt in options:
            matching = [r for r in c_results if r["config"].get(lever) == opt]
            if matching:
                avg_sharpe = sum(r["metrics"]["sharpe"] for r in matching) / len(matching)
                avg_real = sum(r["metrics"]["cagr_real"] for r in matching) / len(matching)
                avg_dd = sum(abs(r["metrics"]["max_drawdown"]) for r in matching) / len(matching)
                feature_impacts.append({
                    "model": "C", "lever": lever, "option": str(opt),
                    "avg_sharpe": round(avg_sharpe, 3),
                    "avg_cagr_real": round(avg_real, 2),
                    "avg_max_dd": round(avg_dd, 2),
                    "n_scenarios": len(matching),
                })

    return recs, feature_impacts


def main():
    print("Building dataset...")
    dataset = build_full_dataset()
    prices = dataset["prices"]
    egx30 = dataset["egx30"]
    usdegp = dataset["usdegp"]
    cpi = dataset["cpi"]
    tbill = dataset["tbill"]

    print("Generating scenario grid...")
    scenarios = build_scenario_grid()
    print(f"Running {len(scenarios)} scenarios...")

    all_results = []
    for i, s in enumerate(scenarios):
        config = SimConfig(
            model=s["model"],
            inception_date=s["inception_date"],
            entry_timing=s.get("entry_timing", "day1"),
            holding_period_months=s.get("holding_period_months"),
            weighting=s.get("weighting", "equal"),
            position_cap=s.get("position_cap"),
            downside_protection=s.get("downside_protection", "none"),
            tbill_buffer_pct=s.get("tbill_buffer_pct", 0.10),
            lockup_avoidance=s.get("lockup_avoidance", False),
            rebalance_freq=s.get("rebalance_freq", "quarterly"),
            sector_cap=s.get("sector_cap"),
            flip_strategy=s.get("flip_strategy", "hold"),
            allocation_track=s.get("allocation_track", "retail"),
        )

        engine = SimulationEngine(config, prices, egx30, tbill, EGX_IPO_UNIVERSE)
        result = engine.run()
        metrics = compute_metrics(result, egx30, usdegp, cpi, tbill)
        label = _config_label(s)
        verdict = generate_verdict(metrics, label)

        config_dict = {
            "model": s["model"],
            "inception_date": str(s["inception_date"]),
        }
        for k, v in s.items():
            if k not in ("model", "inception_date"):
                config_dict[k] = v

        all_results.append({
            "id": i,
            "label": label,
            "config": config_dict,
            "metrics": metrics,
            "verdict": verdict,
            "allocation_details": _extract_allocation_details(result) if s["model"] == "C" else None,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(scenarios)} done")

    print("Generating recommendations...")
    recs, feature_impacts = generate_recommendations(all_results)

    # Universe info for the UI
    universe_info = []
    for co in EGX_IPO_UNIVERSE:
        universe_info.append({
            "ticker": co.ticker,
            "name": co.name,
            "listing_date": str(co.listing_date),
            "offer_price": co.offer_price,
            "raise_egp": co.raise_amount_egp,
            "sector": co.sector,
            "oversub_retail": co.oversubscription_retail,
            "oversub_inst": co.oversubscription_institutional,
            "lockup_days": co.lockup_days,
        })

    output = {
        "generated_at": str(dt.datetime.now()),
        "universe": universe_info,
        "scenarios": all_results,
        "recommendations": recs,
        "feature_impacts": feature_impacts,
        "lever_options": {
            "models": ["B", "C"],
            "inception_dates": ["2022-01-01", "2022-07-01", "2023-01-01"],
            "entry_timing": ["day1", "day5", "day30"],
            "holding_period_months": [null_safe("None"), "6", "12", "24"],
            "weighting": ["equal", "offersize"],
            "position_cap": [null_safe("None"), "0.15", "0.20"],
            "downside_protection": ["none", "stop15", "stop20"],
            "tbill_buffer_pct": ["0.0", "0.10", "0.20"],
            "rebalance_freq": ["quarterly", "on_new_listing"],
            "sector_cap": [null_safe("None"), "0.30"],
            "lockup_avoidance": ["false", "true"],
            "flip_strategy": ["flip_day1", "flip_week1", "hold"],
            "allocation_track": ["retail", "institutional"],
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "output", "simulation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDone! {len(all_results)} scenarios written to {out_path}")
    print(f"Passing scenarios: {sum(1 for r in all_results if r['metrics']['passes_verdict'])}/{len(all_results)}")


def null_safe(val):
    return None if val == "None" else val


def _extract_allocation_details(result) -> list[dict]:
    details = []
    for t in result.trades:
        if t["action"] == "ipo_alloc":
            details.append({
                "date": t["date"],
                "ticker": t["ticker"],
                "offer_price": t["price"],
                "shares_allocated": round(t["shares"], 2),
                "subscription_amount": round(t.get("subscription", 0), 2),
                "fill_rate": round(t.get("fill_rate", 0) * 100, 2),
            })
    return details


if __name__ == "__main__":
    main()
