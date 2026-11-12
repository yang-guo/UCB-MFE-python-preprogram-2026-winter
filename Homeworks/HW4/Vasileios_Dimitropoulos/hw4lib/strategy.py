"""Weekly top-N strategy with name and sector concentration limits.

The allocation idea (fee-inclusive equal budgets, reduce a purchase to fit
name/sector room) follows ``capped_budgets`` in
``Lectures/modeling/backtest.py``.  Differences: we buy every week instead of
monthly; during the first 13 weeks each weekly cohort is NAV/13 (13 weekly
cohorts ~ one 90-day holding period), afterwards all available cash (cash +
expected proceeds of lots maturing at the fill close) is reinvested; and if a
name hits a cap we move further down the ranking so the budget is not idle.
"""

import numpy as np
import pandas as pd


class WeeklyTopN:
    def __init__(self, scores, sectors, top_n=20, cohort_fraction=1 / 13,
                 name_cap=0.05, sector_cap=0.30, min_order=1_000.0):
        """``scores``: DataFrame with columns date (signal date), symbol, score.

        Scores must already be restricted to stocks that were eligible at their
        signal date; the strategy only ranks what it is given for that date.
        """
        if scores.duplicated(["date", "symbol"]).any() or not np.isfinite(scores.score).all():
            raise ValueError("Scores must be finite with one row per date and symbol")
        self.by_date = {d: g.set_index("symbol").score.sort_values(ascending=False, kind="stable")
                        for d, g in scores.sort_values(["date", "symbol"]).groupby("date")}
        self.sectors = sectors
        self.top_n = top_n
        self.cohort_fraction = cohort_fraction
        self.name_cap = name_cap
        self.sector_cap = sector_cap
        self.min_order = min_order

    @property
    def signal_dates(self):
        return pd.DatetimeIndex(sorted(self.by_date))

    def orders(self, signal_date, execution_date, snapshot):
        """Fee-inclusive dollar budgets by symbol, planned at the signal close."""
        ranked = self.by_date.get(signal_date)
        available = snapshot.available     # cash + expected proceeds of lots maturing tomorrow
        if ranked is None or available <= self.min_order:
            return pd.Series(dtype=float)
        nav, fee = snapshot.nav, snapshot.fee_rate
        cohort = nav * self.cohort_fraction
        # Ramp-up (lots of cash): invest one weekly cohort (NAV/13) so purchases are
        # spread over the first 13 weeks.  Steady state (cash <= two cohorts): invest
        # all available cash, so proceeds from extra maturities never sit idle.
        budget = available if available <= 2 * cohort else cohort
        per_name = budget / self.top_n
        names = snapshot.holdings.to_dict()
        sector = self.sectors.to_dict()
        by_sector = {}
        for sym, value in names.items():
            sec = sector.get(sym, "Unknown")
            by_sector[sec] = by_sector.get(sec, 0.0) + value
        orders = {}
        for sym in ranked.index:
            if len(orders) == self.top_n or budget < self.min_order:
                break
            sec = sector.get(sym, "Unknown")
            name_room = self.name_cap * nav - names.get(sym, 0.0)
            sector_room = self.sector_cap * nav - by_sector.get(sec, 0.0)
            amount = min(per_name, budget, name_room * (1 + fee), sector_room * (1 + fee))
            if amount < self.min_order:
                continue          # cap reached: try the next-best name
            orders[sym] = amount
            budget -= amount
            names[sym] = names.get(sym, 0.0) + amount / (1 + fee)
            by_sector[sec] = by_sector.get(sec, 0.0) + amount / (1 + fee)
        return pd.Series(orders, dtype=float)
