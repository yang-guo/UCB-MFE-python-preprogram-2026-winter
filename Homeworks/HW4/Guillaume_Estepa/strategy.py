"""Trading rule: staggered tranches of the top-N predicted stocks.

Capital is split into ``n_tranches`` sleeves that start 21 sessions apart.
Each sleeve buys the N best-ranked stocks, holds them for 90 calendar days,
and reinvests its proceeds the session after they are sold. Because the
90-day exit date is known at entry, the whole decision calendar can be built
from the market calendar alone (see ``build_schedule``).
"""

import math

import pandas as pd


def build_schedule(sessions, first_signal, end, n_tranches=3, spacing=21, holding_days=90):
    """Return {signal_date: tranche} for every planned rebalance.

    A sleeve decides at the close of ``signal``, buys at the next close,
    sells at the first session on/after entry + 90 days, and decides again at
    that same close (the sale proceeds are known by then).
    """
    sessions = pd.DatetimeIndex(sessions)
    end = pd.Timestamp(end)
    schedule = {}
    start = sessions.get_loc(pd.Timestamp(first_signal))
    for tranche in range(n_tranches):
        i = start + tranche * spacing
        while i + 1 < len(sessions) and sessions[i + 1] <= end:
            schedule.setdefault(sessions[i], tranche)
            exit_date = sessions[i + 1] + pd.Timedelta(days=holding_days)
            i = sessions.searchsorted(exit_date)
    return dict(sorted(schedule.items()))


class TopNTrancheStrategy:
    """Equal-weight the N highest predictions on each scheduled signal date.

    Parameters
    ----------
    predictions : DataFrame
        Columns ``date, symbol, prediction, sector, liquidity_rank``. A row
        dated ``t`` only uses information available at the close of ``t``.
    schedule : dict
        ``{signal_date: tranche}`` from ``build_schedule``.
    max_liquidity_rank : int
        Screen: only buy the most liquid names (rank by trailing dollar volume).
    sector_cap : float
        At most this share of a tranche's names may come from one sector.
    """

    def __init__(self, predictions, schedule, n_stocks=50, n_tranches=3,
                 max_liquidity_rank=1000, sector_cap=0.30):
        usable = predictions.dropna(subset=["prediction"])
        self.by_date = {date: group for date, group in usable.groupby("date")}
        self.schedule = schedule
        self.n_stocks = n_stocks
        self.n_tranches = n_tranches
        self.max_liquidity_rank = max_liquidity_rank
        self.max_per_sector = math.ceil(sector_cap * n_stocks)
        self.deployed = set()  # tranches that made their first purchase
        self.log = []

    def select(self, date):
        """Pick the names for one tranche: best predictions, sector-capped."""
        candidates = self.by_date.get(date)
        if candidates is None:
            return []
        candidates = candidates[candidates.liquidity_rank <= self.max_liquidity_rank]
        candidates = candidates.sort_values(["prediction", "symbol"], ascending=[False, True])
        chosen, per_sector = [], {}
        for row in candidates.itertuples():
            if per_sector.get(row.sector, 0) >= self.max_per_sector:
                continue
            chosen.append(row.symbol)
            per_sector[row.sector] = per_sector.get(row.sector, 0) + 1
            if len(chosen) == self.n_stocks:
                break
        return chosen

    def plan(self, snapshot):
        """Called by the backtester after each close; returns {symbol: dollars}."""
        tranche = self.schedule.get(snapshot.date)
        if tranche is None:
            return {}
        if tranche not in self.deployed:
            # Initial build-up: leave an equal share for sleeves not yet started.
            remaining = self.n_tranches - len(self.deployed)
            budget = snapshot.cash / remaining
            self.deployed.add(tranche)
        else:
            budget = snapshot.cash  # reinvest everything that has matured
        symbols = self.select(snapshot.date)
        if not symbols or budget < 1.0:
            return {}
        budget *= 1 - 1e-9  # guard against floating-point overspending
        self.log.append({"signal_date": snapshot.date, "tranche": tranche,
                         "budget": budget, "n_names": len(symbols)})
        return {symbol: budget / len(symbols) for symbol in symbols}
