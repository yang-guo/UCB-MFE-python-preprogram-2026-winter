"""Strategy = schedule + alpha + selection + sizing + budget + risk constraints.

Each concept is a small, swappable object, so a new idea changes one piece:

* ``Schedule``    when to open a new cohort of purchases
* ``ScoreBook``   point-in-time scores (alpha) with their timing metadata
* ``Selector``    which candidates to buy (TopN, TopFraction)
* ``Sizer``       relative weights among selected names (equal, rank,
                  score-proportional, inverse volatility)
* ``CohortBudget`` how much cash a cohort may use
* ``NameCap`` / ``SectorCap``  fee-inclusive exposure limits, aggregated over
                  all open lots (adapted from ``backtest.capped_budgets`` in Lecture 8)

A model score is never treated as a weight unless ``ScoreProportional`` is
chosen deliberately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import DecisionContext, Strategy


# ------------------------------------------------------------------ schedules
class Schedule(ABC):
    @abstractmethod
    def is_entry(self, ctx: DecisionContext) -> bool: ...


class Weekly(Schedule):
    """Open a cohort after the last session of each week (entry next session)."""

    def is_entry(self, ctx):
        return ctx.signal_date.to_period("W-FRI") != ctx.execution_date.to_period("W-FRI")


class Monthly(Schedule):
    """Open a cohort after the last session of each month (Lecture 8 convention)."""

    def is_entry(self, ctx):
        return ctx.signal_date.to_period("M") != ctx.execution_date.to_period("M")


SCHEDULES = {"weekly": Weekly, "monthly": Monthly}
COHORTS_PER_HOLDING = {"weekly": 13, "monthly": 3}  # 90 calendar days of overlapping cohorts


# ------------------------------------------------------------------ alpha
class ScoreBook:
    """Scores prepared outside the backtest (e.g. by walk-forward model fits).

    Required columns: date, symbol, score, fit_date, train_max_label_end.
    Timing rules enforced at construction: fit_date <= date (model existed at
    the signal close) and train_max_label_end < fit_date (every training
    target had completed before the fit).  Extra columns (sector, vol_63, ...)
    travel with the scores for sizers and constraints.
    """

    REQUIRED = ["date", "symbol", "score", "fit_date", "train_max_label_end"]

    def __init__(self, frame: pd.DataFrame):
        missing = set(self.REQUIRED) - set(frame.columns)
        if missing:
            raise ValueError(f"ScoreBook missing columns {sorted(missing)}")
        f = frame.copy()
        for c in ["date", "fit_date", "train_max_label_end"]:
            f[c] = pd.to_datetime(f[c])
        if f.duplicated(["date", "symbol"]).any():
            raise ValueError("One score per (date, symbol)")
        if not np.isfinite(f.score.to_numpy(float)).all():
            raise ValueError("Scores must be finite")
        if (f.fit_date > f.date).any():
            raise ValueError("A score was produced by a model fitted after its signal date")
        trained = f.train_max_label_end.notna()
        if (f.loc[trained, "train_max_label_end"] >= f.loc[trained, "fit_date"]).any():
            raise ValueError("A model trained on a target that completed at/after its fit date")
        self.frame = f
        self._by_date = {d: g.set_index("symbol") for d, g in f.groupby("date")}

    def info(self, date) -> pd.DataFrame:
        return self._by_date.get(pd.Timestamp(date), pd.DataFrame(columns=self.frame.columns).set_index("symbol"))

    @classmethod
    def from_column(cls, panel: pd.DataFrame, column: str, extra=("sector", "vol_63"), sign: float = 1.0):
        """Model-free alpha: an existing point-in-time panel column as the score."""
        rows = panel.loc[panel.eligible & panel[column].notna(), ["date", "symbol", column, *extra]]
        rows = rows.rename(columns={column: "score"})
        rows["score"] = sign * rows.score.astype(float)
        rows["fit_date"] = rows.date
        rows["train_max_label_end"] = pd.NaT
        return cls(rows)


# ------------------------------------------------------------------ selection
class Selector(ABC):
    @abstractmethod
    def select(self, info: pd.DataFrame, ctx: DecisionContext) -> pd.Index: ...


@dataclass
class TopN(Selector):
    n: int

    def select(self, info, ctx):
        ranked = info.reset_index().sort_values(["score", "symbol"], ascending=[False, True])
        return pd.Index(ranked.symbol.head(self.n))


@dataclass
class TopFraction(Selector):
    fraction: float
    max_n: int = 100

    def select(self, info, ctx):
        n = int(min(self.max_n, max(1, np.floor(self.fraction * len(info)))))
        return TopN(n).select(info, ctx)


# ------------------------------------------------------------------ sizing
class Sizer(ABC):
    @abstractmethod
    def weights(self, chosen: pd.DataFrame, ctx: DecisionContext) -> pd.Series:
        """Nonnegative weights summing to 1, indexed by symbol."""


class EqualWeight(Sizer):
    def weights(self, chosen, ctx):
        return pd.Series(1.0 / len(chosen), index=chosen.index)


class RankWeight(Sizer):
    """Linearly decreasing in score rank: best name gets weight n, worst 1."""

    def weights(self, chosen, ctx):
        r = chosen.score.rank(method="first", ascending=True)
        return r / r.sum()


class ScoreProportional(Sizer):
    """Weights proportional to score minus the lowest selected score (plus a floor)."""

    def weights(self, chosen, ctx):
        s = chosen.score - chosen.score.min()
        s = s + 0.1 * s.mean() if s.sum() > 0 else pd.Series(1.0, index=chosen.index)
        return s / s.sum()


@dataclass
class InverseVolatility(Sizer):
    column: str = "vol_63"

    def weights(self, chosen, ctx):
        v = chosen[self.column].astype(float)
        inv = 1.0 / v.where(v > 0)
        inv = inv.fillna(inv.median() if inv.notna().any() else 1.0)
        return inv / inv.sum()


SIZERS = {"equal": EqualWeight, "rank": RankWeight, "score": ScoreProportional, "inverse_vol": InverseVolatility}


# ------------------------------------------------------------------ budget & constraints
@dataclass
class CohortBudget:
    """A cohort may use min(cash, NAV / cohorts): with 13 weekly cohorts over a
    90-day hold the book is roughly fully invested without borrowing."""

    cohorts: int

    def __call__(self, portfolio) -> float:
        return max(0.0, min(portfolio.cash, portfolio.nav / self.cohorts))


@dataclass
class NameCap:
    cap: float  # max fraction of NAV in one symbol across all lots

    def apply(self, orders, ctx, info, fee):
        nav, held = ctx.portfolio.nav, ctx.portfolio.holdings
        out = {}
        for sym, budget in orders.items():
            room = max(0.0, self.cap * nav - held.get(sym, 0.0))
            out[sym] = min(budget, room * (1 + fee))
        return pd.Series(out, dtype=float)


@dataclass
class SectorCap:
    cap: float
    sectors: dict  # symbol -> sector (static snapshot; documented limitation)

    def apply(self, orders, ctx, info, fee):
        nav, held = ctx.portfolio.nav, ctx.portfolio.holdings
        used = held.groupby(held.index.map(lambda s: self.sectors.get(s, "Unknown"))).sum().to_dict()
        out = {}
        for sym, budget in orders.items():  # orders arrive in score order
            sec = self.sectors.get(sym, "Unknown")
            room = max(0.0, self.cap * nav - used.get(sec, 0.0))
            take = min(budget, room * (1 + fee))
            out[sym] = take
            used[sec] = used.get(sec, 0.0) + take / (1 + fee)
        return pd.Series(out, dtype=float)


# ------------------------------------------------------------------ composite
class CompositeStrategy(Strategy):
    def __init__(self, scores: ScoreBook, schedule: Schedule, selector: Selector, sizer: Sizer,
                 budget: CohortBudget, constraints=(), holding_days: int = 90, min_order: float = 100.0):
        self.scores, self.schedule, self.selector, self.sizer = scores, schedule, selector, sizer
        self.budget, self.constraints = budget, list(constraints)
        self.holding_days, self.min_order = holding_days, min_order
        self.decisions: list[dict] = []

    def generate_orders(self, ctx):
        if not self.schedule.is_entry(ctx):
            return pd.Series(dtype=float)
        info = self.scores.info(ctx.signal_date)
        budget = self.budget(ctx.portfolio)
        log = {"signal_date": ctx.signal_date, "candidates": len(info), "budget": budget}
        if not len(info) or budget <= self.min_order:
            self.decisions.append({**log, "selected": 0, "ordered": 0.0})
            return pd.Series(dtype=float)
        chosen = info.loc[self.selector.select(info, ctx)]
        w = self.sizer.weights(chosen, ctx)
        if (w < -1e-12).any() or not np.isclose(w.sum(), 1.0):
            raise ValueError("Sizer weights must be nonnegative and sum to one")
        orders = (w * budget).reindex(chosen.index)
        for c in self.constraints:
            orders = c.apply(orders, ctx, info, ctx.portfolio.fee_rate)
        orders = orders[orders > self.min_order]
        self.decisions.append({**log, "selected": len(chosen), "ordered": float(orders.sum())})
        return orders.rename_axis("symbol").astype(float)
