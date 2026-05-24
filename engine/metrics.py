"""
Performance metrics — computed three ways: nominal EGP, real (CPI-adjusted), and USD.
"""

import numpy as np
import pandas as pd
import datetime as dt
from engine.simulator import SimResult


def compute_metrics(
    result: SimResult,
    egx30_df: pd.DataFrame,
    usdegp_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    tbill_df: pd.DataFrame,
) -> dict:
    if not result.dates or len(result.nav_series) < 2:
        return _empty_metrics()

    nav = np.array(result.nav_series)
    dates = result.dates

    egx30_df = egx30_df.copy()
    egx30_df["date"] = pd.to_datetime(egx30_df["date"]).dt.date
    usdegp_df = usdegp_df.copy()
    usdegp_df["date"] = pd.to_datetime(usdegp_df["date"]).dt.date
    cpi_df = cpi_df.copy()
    cpi_df["date"] = pd.to_datetime(cpi_df["date"]).dt.date
    tbill_df = tbill_df.copy()
    tbill_df["date"] = pd.to_datetime(tbill_df["date"]).dt.date

    # Map FX, CPI, EGX30 to simulation dates
    fx_map = dict(zip(usdegp_df["date"], usdegp_df["usdegp"]))
    cpi_map = dict(zip(cpi_df["date"], cpi_df["cpi"]))
    egx_map = dict(zip(egx30_df["date"], egx30_df["egx30"]))
    tbill_map = dict(zip(tbill_df["date"], tbill_df["tbill_annual"]))

    def _nearest(lookup, target):
        best_d = None
        for d in lookup:
            if d <= target:
                if best_d is None or d > best_d:
                    best_d = d
        return lookup.get(best_d) if best_d else None

    start_date = dates[0]
    end_date = dates[-1]
    years = max((end_date - start_date).days / 365.25, 0.01)

    # Nominal EGP
    total_return_nominal = (nav[-1] / nav[0]) - 1
    cagr_nominal = (nav[-1] / nav[0]) ** (1 / years) - 1

    # Real (CPI-adjusted)
    cpi_start = _nearest(cpi_map, start_date) or 100
    cpi_end = _nearest(cpi_map, end_date) or cpi_start
    inflation_factor = cpi_end / cpi_start
    real_nav_end = nav[-1] / inflation_factor
    total_return_real = (real_nav_end / nav[0]) - 1
    cagr_real = (real_nav_end / nav[0]) ** (1 / years) - 1

    # USD
    fx_start = _nearest(fx_map, start_date) or 15.7
    fx_end = _nearest(fx_map, end_date) or fx_start
    nav_usd_start = nav[0] / fx_start
    nav_usd_end = nav[-1] / fx_end
    total_return_usd = (nav_usd_end / nav_usd_start) - 1
    cagr_usd = (nav_usd_end / nav_usd_start) ** (1 / years) - 1

    # Daily returns
    daily_returns = np.diff(nav) / nav[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # Volatility (annualized)
    vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0

    # Max drawdown
    peak = np.maximum.accumulate(nav)
    drawdowns = (nav - peak) / peak
    max_drawdown = float(np.min(drawdowns))

    # Sharpe (vs T-bill rate)
    avg_tbill = np.mean([r for r in tbill_map.values()]) if tbill_map else 0.20
    excess_return = cagr_nominal - avg_tbill
    sharpe = excess_return / vol if vol > 0 else 0

    # Sortino
    neg_returns = daily_returns[daily_returns < 0]
    downside_vol = np.std(neg_returns) * np.sqrt(252) if len(neg_returns) > 1 else vol
    sortino = excess_return / downside_vol if downside_vol > 0 else 0

    # Alpha vs EGX30
    egx_start = _nearest(egx_map, start_date) or 10800
    egx_end = _nearest(egx_map, end_date) or egx_start
    egx30_return = (egx_end / egx_start) - 1
    egx30_cagr = (egx_end / egx_start) ** (1 / years) - 1
    alpha = cagr_nominal - egx30_cagr

    # Per-holding contribution
    holding_contributions = _compute_holding_contributions(result)

    hit_rate = 0.0
    worst_name = ""
    worst_contribution = 0.0
    if holding_contributions:
        winners = sum(1 for v in holding_contributions.values() if v > 0)
        hit_rate = winners / len(holding_contributions)
        worst_name = min(holding_contributions, key=holding_contributions.get)
        worst_contribution = holding_contributions[worst_name]

    # Verdict
    tbill_only_return = (1 + avg_tbill) ** years - 1
    passes = (
        total_return_real > 0
        and max_drawdown >= -abs(_nearest(dict(zip(egx30_df["date"], egx30_df["egx30"])), end_date) or 0) * 0  # placeholder
        and sharpe > 0
    )

    # Build NAV timeseries for chart (sampled)
    sample_step = max(1, len(dates) // 500)
    nav_timeseries = []
    for i in range(0, len(dates), sample_step):
        d = dates[i]
        fx_d = _nearest(fx_map, d) or fx_start
        cpi_d = _nearest(cpi_map, d) or cpi_start
        nav_timeseries.append({
            "date": str(d),
            "nav_nominal": round(nav[i], 2),
            "nav_real": round(nav[i] / (cpi_d / cpi_start), 2),
            "nav_usd": round(nav[i] / fx_d, 2),
        })
    # Always include last point
    if len(dates) > 0 and nav_timeseries[-1]["date"] != str(dates[-1]):
        d = dates[-1]
        fx_d = _nearest(fx_map, d) or fx_start
        cpi_d = _nearest(cpi_map, d) or cpi_start
        nav_timeseries.append({
            "date": str(d),
            "nav_nominal": round(nav[-1], 2),
            "nav_real": round(nav[-1] / (cpi_d / cpi_start), 2),
            "nav_usd": round(nav[-1] / fx_d, 2),
        })

    # Drawdown timeseries
    dd_timeseries = []
    for i in range(0, len(dates), sample_step):
        dd_timeseries.append({"date": str(dates[i]), "drawdown": round(float(drawdowns[i]) * 100, 2)})

    # EGX30 timeseries (normalized)
    egx_norm_series = []
    egx_base = _nearest(egx_map, start_date) or 10800
    for i in range(0, len(dates), sample_step):
        d = dates[i]
        e = _nearest(egx_map, d) or egx_base
        egx_norm_series.append({"date": str(d), "value": round((e / egx_base) * nav[0], 2)})

    return {
        "total_return_nominal": round(total_return_nominal * 100, 2),
        "cagr_nominal": round(cagr_nominal * 100, 2),
        "total_return_real": round(total_return_real * 100, 2),
        "cagr_real": round(cagr_real * 100, 2),
        "total_return_usd": round(total_return_usd * 100, 2),
        "cagr_usd": round(cagr_usd * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "volatility": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "alpha_vs_egx30": round(alpha * 100, 2),
        "egx30_return": round(egx30_return * 100, 2),
        "hit_rate": round(hit_rate * 100, 1),
        "worst_name": worst_name,
        "worst_contribution": round(worst_contribution * 100, 2),
        "holding_contributions": {k: round(v * 100, 2) for k, v in holding_contributions.items()},
        "passes_verdict": passes,
        "years": round(years, 2),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "nav_timeseries": nav_timeseries,
        "drawdown_timeseries": dd_timeseries,
        "egx30_timeseries": egx_norm_series,
        "cash_pct_avg": round(np.mean(result.cash_series) / np.mean(result.nav_series) * 100, 1) if result.nav_series else 0,
        "num_trades": len(result.trades),
        "final_nav": round(nav[-1], 2),
        "initial_nav": round(nav[0], 2),
    }


def _compute_holding_contributions(result: SimResult) -> dict[str, float]:
    if not result.holdings_history:
        return {}

    contributions: dict[str, float] = {}
    initial_nav = result.nav_series[0] if result.nav_series else 1

    first_holdings = {}
    last_holdings = {}

    for record in result.holdings_history:
        for ticker, info in record["holdings"].items():
            if ticker not in first_holdings:
                first_holdings[ticker] = info["cost_basis"] * info["shares"]
            last_holdings[ticker] = info["price"] * info["shares"]

    for ticker in set(list(first_holdings.keys()) + list(last_holdings.keys())):
        start_val = first_holdings.get(ticker, 0)
        end_val = last_holdings.get(ticker, 0)
        contributions[ticker] = (end_val - start_val) / initial_nav

    return contributions


def _empty_metrics() -> dict:
    return {
        "total_return_nominal": 0, "cagr_nominal": 0, "total_return_real": 0,
        "cagr_real": 0, "total_return_usd": 0, "cagr_usd": 0,
        "max_drawdown": 0, "volatility": 0, "sharpe": 0, "sortino": 0,
        "alpha_vs_egx30": 0, "egx30_return": 0, "hit_rate": 0,
        "worst_name": "", "worst_contribution": 0, "holding_contributions": {},
        "passes_verdict": False, "years": 0, "start_date": "", "end_date": "",
        "nav_timeseries": [], "drawdown_timeseries": [], "egx30_timeseries": [],
        "cash_pct_avg": 0, "num_trades": 0, "final_nav": 0, "initial_nav": 0,
    }


def generate_verdict(metrics: dict, config_label: str) -> str:
    m = metrics
    if m["passes_verdict"]:
        verdict = (
            f"PASS — {config_label} returned {m['cagr_nominal']:.1f}% CAGR nominal "
            f"({m['cagr_real']:.1f}% real, {m['cagr_usd']:.1f}% USD) with a "
            f"{abs(m['max_drawdown']):.1f}% max drawdown and {m['sharpe']:.2f} Sharpe. "
        )
        if m["alpha_vs_egx30"] > 0:
            verdict += f"Outperformed EGX30 by {m['alpha_vs_egx30']:.1f}pp."
        else:
            verdict += f"Underperformed EGX30 by {abs(m['alpha_vs_egx30']):.1f}pp."
    else:
        reasons = []
        if m["total_return_real"] <= 0:
            reasons.append("negative real return")
        if m["sharpe"] <= 0:
            reasons.append("Sharpe below T-bills")
        verdict = (
            f"FAIL — {config_label} returned {m['cagr_nominal']:.1f}% CAGR nominal "
            f"({m['cagr_real']:.1f}% real) with {abs(m['max_drawdown']):.1f}% max drawdown. "
            f"Failed due to: {', '.join(reasons) if reasons else 'criteria not met'}."
        )
    return verdict
