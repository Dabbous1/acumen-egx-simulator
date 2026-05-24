"""
Core simulation engine — daily NAV calculation for Models B and C.
Parameterised by the feature levers defined in the execution plan.
Enforces no look-ahead bias: only information available on each simulated date drives decisions.
"""

import datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimConfig:
    model: Literal["B", "C"] = "B"
    inception_date: dt.date = dt.date(2022, 1, 1)
    initial_aum: float = 100_000_000  # EGP
    recently_listed_window_years: int = 3

    # Fee structure
    management_fee_annual: float = 0.020  # 2.0%
    subscription_load: float = 0.01
    redemption_load: float = 0.005

    # Transaction costs
    brokerage_commission: float = 0.002  # 0.2%
    stamp_tax: float = 0.001  # 0.1%

    # Feature levers
    entry_timing: Literal["day1", "day5", "day30", "first_earnings"] = "day1"
    holding_period_months: int | None = None  # None = hold until aged out
    weighting: Literal["equal", "freefloat", "offersize", "conviction"] = "equal"
    position_cap: float | None = 0.15  # max 15% per name, None = no cap
    sector_cap: float | None = None  # max sector weight, None = no cap
    downside_protection: Literal["none", "stop15", "stop20", "hedge"] = "none"
    tbill_buffer_pct: float = 0.10  # 10% in T-bills
    lockup_avoidance: bool = False
    liquidity_filter_adv: float | None = None  # min avg daily volume in EGP
    rebalance_freq: Literal["none", "quarterly", "on_new_listing"] = "quarterly"

    # Model C specific
    flip_strategy: Literal["flip_day1", "flip_week1", "hold"] = "hold"
    allocation_track: Literal["retail", "institutional"] = "retail"


@dataclass
class Position:
    ticker: str
    shares: float
    cost_basis: float
    entry_date: dt.date
    sector: str


@dataclass
class SimResult:
    dates: list[dt.date]
    nav_series: list[float]
    cash_series: list[float]
    holdings_value_series: list[float]
    trades: list[dict]
    holdings_history: list[dict]
    config: SimConfig


class SimulationEngine:
    def __init__(
        self,
        config: SimConfig,
        prices_df: pd.DataFrame,
        egx30_df: pd.DataFrame,
        tbill_df: pd.DataFrame,
        ipo_universe: list,
    ):
        self.config = config
        self.prices = prices_df.copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"]).dt.date
        self.egx30 = egx30_df.copy()
        self.egx30["date"] = pd.to_datetime(self.egx30["date"]).dt.date
        self.tbill = tbill_df.copy()
        self.tbill["date"] = pd.to_datetime(self.tbill["date"]).dt.date
        self.universe = ipo_universe

        self._price_lookup: dict[tuple[str, dt.date], float] = {}
        for _, row in self.prices.iterrows():
            self._price_lookup[(row["ticker"], row["date"])] = row["close"]

        self._volume_lookup: dict[tuple[str, dt.date], float] = {}
        for _, row in self.prices.iterrows():
            self._volume_lookup[(row["ticker"], row["date"])] = row["volume"]

        self._tbill_lookup: dict = {}
        for _, row in self.tbill.iterrows():
            self._tbill_lookup[row["date"]] = row["tbill_annual"]

    def _get_price(self, ticker: str, date: dt.date) -> float | None:
        return self._price_lookup.get((ticker, date))

    def _get_tbill_daily(self, date: dt.date) -> float:
        best = None
        for d, rate in self._tbill_lookup.items():
            if d <= date:
                if best is None or d > best:
                    best = d
        return self._tbill_lookup.get(best, 0.20) / 252 if best else 0.20 / 252

    def _eligible_companies(self, as_of: dt.date) -> list:
        cfg = self.config
        window = dt.timedelta(days=cfg.recently_listed_window_years * 365)
        eligible = []
        for co in self.universe:
            if co.listing_date > as_of:
                continue
            age = as_of - co.listing_date
            if age > window:
                continue

            # Entry timing filter
            entry_delay = {"day1": 0, "day5": 5, "day30": 30, "first_earnings": 90}
            min_days = entry_delay.get(cfg.entry_timing, 0)
            if age.days < min_days:
                continue

            # Must have a price
            if self._get_price(co.ticker, as_of) is None:
                continue

            eligible.append(co)
        return eligible

    def _apply_transaction_cost(self, amount: float) -> float:
        cost_rate = self.config.brokerage_commission + self.config.stamp_tax
        return amount * (1 - cost_rate)

    def _should_exit(self, pos: Position, current_date: dt.date, current_price: float) -> bool:
        cfg = self.config

        # Holding period check
        if cfg.holding_period_months is not None:
            max_hold = dt.timedelta(days=cfg.holding_period_months * 30)
            if (current_date - pos.entry_date) > max_hold:
                return True

        # Age-out: exceeded recently-listed window
        co = next((c for c in self.universe if c.ticker == pos.ticker), None)
        if co:
            age = current_date - co.listing_date
            if age.days > cfg.recently_listed_window_years * 365:
                return True

        # Downside protection — stop loss
        if cfg.downside_protection == "stop15":
            if current_price < pos.cost_basis * 0.85:
                return True
        elif cfg.downside_protection == "stop20":
            if current_price < pos.cost_basis * 0.80:
                return True

        # Lockup avoidance
        if cfg.lockup_avoidance and co:
            days_to_lockup_end = co.lockup_days - (current_date - co.listing_date).days
            if 0 < days_to_lockup_end <= 10:
                return True

        return False

    def _compute_weights(self, companies: list) -> dict[str, float]:
        cfg = self.config
        n = len(companies)
        if n == 0:
            return {}

        if cfg.weighting == "equal":
            w = {c.ticker: 1.0 / n for c in companies}
        elif cfg.weighting == "offersize":
            total = sum(c.raise_amount_egp for c in companies)
            w = {c.ticker: c.raise_amount_egp / total for c in companies}
        elif cfg.weighting == "freefloat":
            total = sum(c.shares_offered * c.offer_price for c in companies)
            w = {c.ticker: (c.shares_offered * c.offer_price) / total for c in companies}
        else:  # conviction — equal for simulation
            w = {c.ticker: 1.0 / n for c in companies}

        # Position cap
        if cfg.position_cap is not None:
            capped = {}
            excess = 0.0
            uncapped_count = 0
            for t, wt in w.items():
                if wt > cfg.position_cap:
                    capped[t] = cfg.position_cap
                    excess += wt - cfg.position_cap
                else:
                    capped[t] = wt
                    uncapped_count += 1
            if excess > 0 and uncapped_count > 0:
                per_share = excess / uncapped_count
                for t in capped:
                    if capped[t] < cfg.position_cap:
                        capped[t] += per_share
            w = capped

        # Sector cap
        if cfg.sector_cap is not None:
            sector_weights: dict[str, float] = {}
            for c in companies:
                sector_weights.setdefault(c.sector, 0.0)
                sector_weights[c.sector] += w[c.ticker]

            for sector, sw in sector_weights.items():
                if sw > cfg.sector_cap:
                    scale = cfg.sector_cap / sw
                    for c in companies:
                        if c.sector == sector:
                            w[c.ticker] *= scale

            total_w = sum(w.values())
            if total_w > 0:
                w = {t: v / total_w for t, v in w.items()}

        return w

    def run(self) -> SimResult:
        if self.config.model == "B":
            return self._run_model_b()
        else:
            return self._run_model_c()

    def _run_model_b(self) -> SimResult:
        cfg = self.config
        start = cfg.inception_date
        end = dt.date(2025, 12, 31)

        all_dates = sorted(set(d for d in self.prices["date"].unique() if start <= d <= end))
        if not all_dates:
            return SimResult([], [], [], [], [], [], cfg)

        cash = cfg.initial_aum * (1 - cfg.subscription_load)
        positions: dict[str, Position] = {}
        trades = []
        nav_series = []
        cash_series = []
        holdings_value_series = []
        dates_out = []
        last_rebalance = start
        holdings_history = []

        for day_idx, date in enumerate(all_dates):
            # T-bill earnings on cash
            tbill_buffer = cash * cfg.tbill_buffer_pct
            investable_cash = cash - tbill_buffer
            cash += tbill_buffer * self._get_tbill_daily(date)

            # Daily management fee
            total_value = cash
            for pos in positions.values():
                p = self._get_price(pos.ticker, date)
                if p:
                    total_value += pos.shares * p
            daily_fee = total_value * cfg.management_fee_annual / 252
            cash -= daily_fee

            # Check exits
            exits = []
            for ticker, pos in list(positions.items()):
                p = self._get_price(ticker, date)
                if p is None:
                    continue
                if self._should_exit(pos, date, p):
                    exits.append(ticker)

            for ticker in exits:
                pos = positions.pop(ticker)
                p = self._get_price(ticker, date)
                if p:
                    proceeds = self._apply_transaction_cost(pos.shares * p)
                    cash += proceeds
                    trades.append({"date": str(date), "ticker": ticker, "action": "sell",
                                   "price": p, "shares": pos.shares, "proceeds": proceeds})

            # Check for rebalance / new entries
            eligible = self._eligible_companies(date)
            need_rebalance = False

            if cfg.rebalance_freq == "quarterly":
                if (date - last_rebalance).days >= 90:
                    need_rebalance = True
            elif cfg.rebalance_freq == "on_new_listing":
                new_tickers = {c.ticker for c in eligible} - set(positions.keys())
                if new_tickers:
                    need_rebalance = True

            if day_idx == 0:
                need_rebalance = True

            if need_rebalance and eligible:
                target_weights = self._compute_weights(eligible)
                total_portfolio = cash
                for pos in positions.values():
                    p = self._get_price(pos.ticker, date)
                    if p:
                        total_portfolio += pos.shares * p

                equity_budget = total_portfolio * (1 - cfg.tbill_buffer_pct)

                for co in eligible:
                    target_value = equity_budget * target_weights.get(co.ticker, 0)
                    current_value = 0.0
                    p = self._get_price(co.ticker, date)
                    if not p:
                        continue
                    if co.ticker in positions:
                        current_value = positions[co.ticker].shares * p

                    diff = target_value - current_value
                    if abs(diff) < 1000:
                        continue

                    if diff > 0 and cash > diff * 1.01:
                        shares_to_buy = diff / p
                        cost = self._apply_transaction_cost(diff)
                        actual_cost = diff
                        cash -= actual_cost
                        if co.ticker in positions:
                            old = positions[co.ticker]
                            total_shares = old.shares + shares_to_buy
                            avg_cost = (old.cost_basis * old.shares + p * shares_to_buy) / total_shares
                            positions[co.ticker] = Position(co.ticker, total_shares, avg_cost, old.entry_date, co.sector)
                        else:
                            positions[co.ticker] = Position(co.ticker, shares_to_buy, p, date, co.sector)
                        trades.append({"date": str(date), "ticker": co.ticker, "action": "buy",
                                       "price": p, "shares": shares_to_buy, "cost": actual_cost})
                    elif diff < 0:
                        shares_to_sell = abs(diff) / p
                        if co.ticker in positions:
                            shares_to_sell = min(shares_to_sell, positions[co.ticker].shares)
                            proceeds = self._apply_transaction_cost(shares_to_sell * p)
                            cash += proceeds
                            positions[co.ticker].shares -= shares_to_sell
                            if positions[co.ticker].shares < 1:
                                del positions[co.ticker]
                            trades.append({"date": str(date), "ticker": co.ticker, "action": "sell",
                                           "price": p, "shares": shares_to_sell, "proceeds": proceeds})

                last_rebalance = date

            # End-of-day NAV
            holdings_val = 0.0
            day_holdings = {}
            for ticker, pos in positions.items():
                p = self._get_price(ticker, date)
                if p:
                    val = pos.shares * p
                    holdings_val += val
                    day_holdings[ticker] = {"shares": pos.shares, "price": p, "value": val,
                                            "cost_basis": pos.cost_basis}

            nav = cash + holdings_val
            nav_series.append(nav)
            cash_series.append(cash)
            holdings_value_series.append(holdings_val)
            dates_out.append(date)
            holdings_history.append({"date": str(date), "holdings": day_holdings, "cash": cash, "nav": nav})

        return SimResult(dates_out, nav_series, cash_series, holdings_value_series,
                         trades, holdings_history, cfg)

    def _run_model_c(self) -> SimResult:
        """Model C: IPO-Allocation fund — subscribe, get allocated, flip or hold."""
        cfg = self.config
        start = cfg.inception_date
        end = dt.date(2025, 12, 31)

        all_dates = sorted(set(d for d in self.prices["date"].unique() if start <= d <= end))
        if not all_dates:
            return SimResult([], [], [], [], [], [], cfg)

        cash = cfg.initial_aum * (1 - cfg.subscription_load)
        positions: dict[str, Position] = {}
        trades = []
        nav_series = []
        cash_series = []
        holdings_value_series = []
        dates_out = []
        holdings_history = []
        allocated_ipos: set[str] = set()

        for date in all_dates:
            # T-bill on idle cash (the core of Model C between deals)
            cash += cash * self._get_tbill_daily(date)

            # Daily management fee
            total_value = cash
            for pos in positions.values():
                p = self._get_price(pos.ticker, date)
                if p:
                    total_value += pos.shares * p
            daily_fee = total_value * cfg.management_fee_annual / 252
            cash -= daily_fee

            # Check for new IPO allocations on listing day
            for co in self.universe:
                if co.ticker in allocated_ipos:
                    continue
                if co.listing_date != date:
                    continue
                if co.listing_date < start:
                    continue

                # Compute allocation
                if cfg.allocation_track == "retail":
                    oversub = co.oversubscription_retail or 10.0
                    alloc_pct = co.retail_allocation_pct
                else:
                    oversub = co.oversubscription_institutional or 5.0
                    alloc_pct = 1.0 - co.retail_allocation_pct

                subscription_amount = min(cash * 0.80, co.raise_amount_egp * alloc_pct * 0.10)
                fill_rate = 1.0 / oversub
                allocated_value = subscription_amount * fill_rate
                allocated_shares = allocated_value / co.offer_price

                if allocated_shares < 1:
                    allocated_ipos.add(co.ticker)
                    continue

                cash -= allocated_value
                positions[co.ticker] = Position(co.ticker, allocated_shares, co.offer_price, date, co.sector)
                allocated_ipos.add(co.ticker)
                trades.append({"date": str(date), "ticker": co.ticker, "action": "ipo_alloc",
                               "price": co.offer_price, "shares": allocated_shares,
                               "subscription": subscription_amount, "fill_rate": fill_rate})

            # Flip strategy
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                p = self._get_price(ticker, date)
                if not p:
                    continue
                days_held = (date - pos.entry_date).days

                should_flip = False
                if cfg.flip_strategy == "flip_day1" and days_held >= 1:
                    should_flip = True
                elif cfg.flip_strategy == "flip_week1" and days_held >= 5:
                    should_flip = True
                elif cfg.flip_strategy == "hold":
                    if cfg.holding_period_months and days_held > cfg.holding_period_months * 30:
                        should_flip = True
                    elif self._should_exit(pos, date, p):
                        should_flip = True

                if should_flip:
                    proceeds = self._apply_transaction_cost(pos.shares * p)
                    cash += proceeds
                    trades.append({"date": str(date), "ticker": ticker, "action": "sell",
                                   "price": p, "shares": pos.shares, "proceeds": proceeds})
                    del positions[ticker]

            # End-of-day NAV
            holdings_val = 0.0
            day_holdings = {}
            for ticker, pos in positions.items():
                p = self._get_price(ticker, date)
                if p:
                    val = pos.shares * p
                    holdings_val += val
                    day_holdings[ticker] = {"shares": pos.shares, "price": p, "value": val,
                                            "cost_basis": pos.cost_basis}

            nav = cash + holdings_val
            nav_series.append(nav)
            cash_series.append(cash)
            holdings_value_series.append(holdings_val)
            dates_out.append(date)
            holdings_history.append({"date": str(date), "holdings": day_holdings, "cash": cash, "nav": nav})

        return SimResult(dates_out, nav_series, cash_series, holdings_value_series,
                         trades, holdings_history, cfg)
