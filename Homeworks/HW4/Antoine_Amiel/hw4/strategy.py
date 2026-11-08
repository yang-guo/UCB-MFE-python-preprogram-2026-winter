"""Staggered-sleeve, top-N strategy.

Capital is split into ``n_sleeves`` equal sleeves started ``sleeve_gap``
sessions apart (initial deployment). Each purchase is held 90 calendar days;
when a sleeve's lots mature the cash returns and is reinvested at the next
session's close into the top-ranked stocks. Each sleeve therefore rolls every
~90 days and the portfolio stays almost fully invested with staggered entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SleeveTopN:
    def __init__(self, scorer, calendar, start, *, initial_cash=1_000_000.0, n_names=40,
                 n_sleeves=3, sleeve_gap=21, name_cap=0.05, min_deploy=0.01, weighting="equal"):
        self.scorer = scorer
        self.calendar = pd.DatetimeIndex(calendar)
        first = self.calendar.get_loc(self.calendar[self.calendar >= pd.Timestamp(start)][0])
        self.release_days = [self.calendar[first + k * sleeve_gap] for k in range(n_sleeves)]
        self.initial_cash, self.n_names, self.n_sleeves = initial_cash, n_names, n_sleeves
        self.name_cap, self.min_deploy = name_cap, min_deploy
        if weighting not in ("equal", "cap"):
            raise ValueError(weighting)
        self.weighting = weighting
        self.decisions: list[dict] = []

    def reserved_cash(self, exec_date) -> float:
        """Initial capital not yet released to the market."""
        released = sum(d <= exec_date for d in self.release_days)
        return self.initial_cash * (self.n_sleeves - released) / self.n_sleeves

    def orders(self, signal_date, exec_date, cash, nav, holdings) -> pd.Series:
        deployable = cash - self.reserved_cash(exec_date)
        if deployable < self.min_deploy * nav:
            return pd.Series(dtype=float)
        scores = self.scorer.scores(signal_date).sort_values(["score", "symbol"], ascending=[False, True])
        # Names already at the concentration cap are skipped; take the next ranked ones.
        room = self.name_cap * nav - scores.symbol.map(holdings).fillna(0.0).to_numpy()
        picks = scores.loc[room > 0.25 * deployable / self.n_names].head(self.n_names).copy()
        picks["room"] = room[room > 0.25 * deployable / self.n_names][: len(picks)]
        if self.weighting == "cap":     # weights proportional to capitalization at the signal close
            w = picks.mcap.where(np.isfinite(picks.mcap) & (picks.mcap > 0))
            w = w.fillna(w.median() if w.notna().any() else 1.0)
        else:
            w = pd.Series(1.0, index=picks.index)
        out = pd.Series(_capped_allocation(deployable, w.to_numpy(), picks.room.to_numpy()),
                        index=picks.symbol.to_numpy(), dtype=float)
        if out.sum() > deployable:                # never plan more than the free cash
            out *= deployable / out.sum()
        self.decisions.append({"signal_date": signal_date, "exec_date": exec_date, "cash": cash,
                               "deployable": deployable, "n_orders": len(out), "budget": out.sum(),
                               "fit_date": scores.fit_date.iloc[0] if len(scores) else pd.NaT})
        return out


def _capped_allocation(total, weights, caps, iterations=20):
    """Split ``total`` in proportion to ``weights`` without exceeding ``caps``.

    Amounts above a cap are redistributed to the uncapped names (water filling).
    """
    alloc = np.zeros(len(weights))
    free = np.ones(len(weights), dtype=bool)
    remaining = total
    for _ in range(iterations):
        if remaining <= 1e-6 or not free.any():
            break
        share = remaining * weights * free / (weights * free).sum()
        alloc += share
        over = alloc > caps
        remaining = (alloc[over] - caps[over]).sum()
        alloc[over] = caps[over]
        free &= ~over
    return alloc


class Ranked:
    """Adapter: a fixed score column (e.g. momentum) from the same eligible cross-section."""

    def __init__(self, store, column, max_rank=None, glitch=None, sign=1.0):
        self.store, self.column, self.max_rank, self.glitch, self.sign = store, column, max_rank, glitch, sign

    def scores(self, signal_date):
        xs = self.store.cross_section([signal_date], max_rank=self.max_rank)
        if self.glitch is not None:
            flagged = self.glitch.iloc[self.store.calendar.get_loc(signal_date)]
            xs = xs.loc[~xs.symbol.map(flagged).fillna(False).to_numpy()]
        xs = xs.loc[np.isfinite(xs[self.column])]
        xs["score"] = self.sign * xs[self.column]
        xs["fit_date"] = signal_date
        return xs[["date", "symbol", "score", "fit_date", "universe_rank", "mcap"]]
