"""Top-score tranche strategy for HW4.

On each weekly signal date the strategy may buy stocks for tomorrow's close:

- Ramp-in: the first purchase uses 1/3 of capital, 4 signals (~1 month) later
  up to 2/3 is invested, and after 8 signals everything. Because each purchase
  lives 90 days, this creates three staggered tranches whose maturities return
  cash roughly monthly; that cash is reinvested at the next signal date.
- Candidates are stocks with a score on that date (i.e. eligible universe
  members with a quote), optionally screened on trailing dollar volume and on
  the quarter's capitalisation rank (e.g. the 500 largest at selection).
- Buys the ``top_n`` highest scores, equal dollar budget each, with at most
  ``max_sector_share`` of the names from one sector and no stock above
  ``max_weight`` of NAV across all its open lots.
- Skips a date if the investable cash is below ``min_trade_frac`` of NAV.
"""

from __future__ import annotations

import math

import pandas as pd

from backtester import PortfolioView


class TopScoreStrategy:
    def __init__(self, scores: pd.DataFrame, top_n=30, tranches=3, ramp_signals=4,
                 max_weight=0.05, max_sector_share=0.3, min_dollar_vol=0.0,
                 max_universe_rank=None, min_trade_frac=0.01):
        self.by_date = {d: g.sort_values("score", ascending=False)
                        for d, g in scores.groupby("date")}
        self.top_n, self.tranches, self.ramp_signals = top_n, tranches, ramp_signals
        self.max_weight, self.max_sector_share = max_weight, max_sector_share
        self.min_dollar_vol, self.min_trade_frac = min_dollar_vol, min_trade_frac
        self.max_universe_rank = max_universe_rank
        self.signals_seen = 0
        self.log = []

    def __call__(self, view: PortfolioView) -> dict[str, float]:
        g = self.by_date.get(view.date)
        if g is None:
            return {}
        k = self.signals_seen
        self.signals_seen += 1
        invested_cap = view.nav * min(1.0, (k // self.ramp_signals + 1) / self.tranches)
        budget = min(view.cash, max(0.0, invested_cap - view.holdings.sum()))
        if budget < self.min_trade_frac * view.nav:
            return {}

        per_name = budget / self.top_n
        sector_limit = max(1, math.floor(self.max_sector_share * self.top_n))
        cand = g[g.dollar_vol_raw >= self.min_dollar_vol] if self.min_dollar_vol else g
        if self.max_universe_rank:
            cand = cand[cand.universe_rank <= self.max_universe_rank]
        picks, sectors = {}, {}
        for sym, sector in zip(cand.symbol, cand.sector):
            if len(picks) == self.top_n:
                break
            if sectors.get(sector, 0) >= sector_limit:
                continue
            if view.holdings.get(sym, 0.0) + per_name > self.max_weight * view.nav:
                continue
            picks[sym] = per_name
            sectors[sector] = sectors.get(sector, 0) + 1
        self.log.append(dict(date=view.date, budget=budget, names=len(picks),
                             candidates=len(cand)))
        return picks
