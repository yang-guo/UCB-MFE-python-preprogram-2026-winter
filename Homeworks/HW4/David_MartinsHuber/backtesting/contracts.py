"""Interfaces shared by research, strategies, and the simulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import numbers
from typing import Mapping

import pandas as pd


def plain_amount(value) -> float:
    """Return a plain finite nonnegative float for a proposed budget.

    Converting at the trust boundary prevents a float/int subclass from
    overriding arithmetic dunders and lying to risk policies while the order
    itself records an honest value.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("Budget must be a finite nonnegative number")
    try:
        amount = float(value)
    except Exception:
        raise ValueError("Budget must be a finite nonnegative number") from None
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Budget must be a finite nonnegative number")
    return 0.0 if amount == 0 else amount


class Phase(str, Enum):
    READY = "ready"
    REFITTING = "refitting"
    DECIDING = "deciding"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class Portfolio:
    cash: float
    available_cash: float
    nav: float
    holdings: Mapping[str, float]


@dataclass(frozen=True)
class DecisionContext:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    prices: pd.DataFrame
    membership: pd.DataFrame
    fundamentals: pd.DataFrame
    portfolio: Portfolio
    model_version: int
    # Added at the end so existing positional construction keeps working.
    volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class FitData:
    X: pd.DataFrame
    y: pd.Series
    audit: pd.DataFrame


class Strategy(ABC):
    """Consume supplied history only; never read future data independently.

    One strategy runs one account in v1. Keep trained models on this object.
    Refit implementations must fit preprocessing using FitData.X only.

    `required_financials` declares the point-in-time financial columns the
    strategy needs. The engine supplies exactly those in
    `DecisionContext.financials`; requesting a column the data source cannot
    provide fails at construction rather than silently returning NaN.

    Set ``self_fitting = True`` when the strategy fits itself from the
    point-in-time price/factor history in ``DecisionContext`` and therefore does
    not need the engine's 90-day ``training_panel``. The engine still owns the
    refit phase, model-version counter, and audit record.
    """

    required_financials: tuple[str, ...] = ()
    self_fitting: bool = False

    def __init__(self, strategy_id: str):
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise ValueError("A strategy needs a nonempty strategy_id")
        required = getattr(type(self), "required_financials", ())
        if isinstance(required, str) or any(not isinstance(c, str) or not c for c in required):
            raise ValueError("required_financials must be a sequence of column names")
        self.strategy_id = strategy_id
        self.required_financials = tuple(required)

    def needs_refit(self, context: DecisionContext) -> bool:
        return False

    def refit(self, data: FitData, context: DecisionContext) -> None:
        raise NotImplementedError("A refitting strategy must implement refit")

    def refit_settings(self) -> Mapping:
        """Override to record model settings, window policy, and random seeds."""
        return {}

    @abstractmethod
    def propose_buys(self, context: DecisionContext) -> Mapping[str, float]:
        """Return symbol -> fee-inclusive dollar budget; {} means no buys."""
