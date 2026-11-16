"""Explicit assumptions shared by experiments; no mutable global account state."""

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COURSE = PROJECT.parents[2]
DATABASE = COURSE / "Lectures/modeling/data/equities_top2000.duckdb"
CLASS_DATABASE = DATABASE.with_name("equities_class.duckdb")
PRICE_FEATURES = ["momentum_63", "momentum_252_21", "volatility_63"]


@dataclass(frozen=True)
class TradingRules:
    """Weights are fractions; costs are basis points PER SIDE; days are calendar days."""

    initial_cash: float = 1_000_000.0
    holding_days: int = 90
    top_n: int = 20
    cohort_fraction: float = 1 / 3
    name_cap: float = 0.05
    fee_bps: float = 5.0
    max_stale_sessions: int = 5
    annual_rf: float = 0.02

    def as_dict(self):
        return asdict(self)
