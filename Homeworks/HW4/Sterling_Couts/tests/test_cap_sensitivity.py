from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cap_sensitivity_allocation import plan_sector_budgets
from cap_sensitivity_engine import run_sector_backtest as sweep_backtest
from sector_allocation import plan_sector_budgets as original_plan
from sector_engine import run_sector_backtest
from hw4.settings import TradingRules


def test_different_sector_caps_and_zero_new_allocation():
    scores = pd.Series({"E": 3.0, "T": 2.0, "U": 1.0})
    sectors = {"E": "Energy", "OLD": "Energy", "T": "Technology", "U": "Utilities"}
    rules = replace(TradingRules(), top_n=3, name_cap=0.5, cohort_fraction=1.0)
    caps = {"Energy": 0.0, "Technology": 0.10, "Utilities": 0.20}
    orders = plan_sector_budgets(
        scores, 950.0, 1000.0, {"OLD": 50.0}, rules, sectors, caps
    )
    assert "E" not in orders  # Existing $50 is retained; a zero cap is not a sale.
    np.testing.assert_allclose([orders["T"], orders["U"]], [100.05, 200.10])
    assert sum(orders.values()) < 950.0


def test_uniform_caps_reproduce_original_and_missing_sector_cap_is_rejected():
    scores, rules = pd.Series({"E": 2.0, "T": 1.0}), TradingRules()
    labels = {"E": "Energy", "T": "Technology"}
    assert plan_sector_budgets(
        scores, 1000.0, 1000.0, {}, rules, labels, {"Energy": 0.20, "Technology": 0.20}
    ) == original_plan(scores, 1000.0, 1000.0, {}, rules, labels, 0.20)
    with pytest.raises(ValueError, match="missing"):
        plan_sector_budgets(scores, 1000.0, 1000.0, {}, rules, labels, {"Energy": 0.20})
    with pytest.raises(ValueError, match="finite"):
        plan_sector_budgets(
            scores,
            1000.0,
            1000.0,
            {},
            rules,
            labels,
            {"Energy": np.nan, "Technology": 0.20},
        )


@pytest.mark.parametrize("reinvest", [False, True])
def test_sensitivity_engine_matches_existing_engine_for_same_caps(reinvest):
    days = pd.bdate_range("2015-01-01", "2015-06-30")
    quotes = pd.DataFrame(
        {
            "X": np.linspace(100.0, 125.0, len(days)),
            "Y": np.linspace(100.0, 90.0, len(days)),
        },
        index=days,
    )
    benchmark = pd.Series(100.0, index=days)
    scores = pd.DataFrame(
        {"date": [days[0], days[0]], "symbol": ["X", "Y"], "score": [2.0, 1.0]}
    )
    eligible = pd.DataFrame(True, index=days, columns=quotes.columns)
    args = dict(
        sectors={"X": "Energy", "Y": "Technology"},
        reinvest_maturity_cash=reinvest,
        entry_eligibility=eligible,
    )
    old = run_sector_backtest(
        quotes, benchmark, scores, days[1], days[-1], sector_cap=0.20, **args
    )
    new = sweep_backtest(
        quotes,
        benchmark,
        scores,
        days[1],
        days[-1],
        sector_cap={"Energy": 0.20, "Technology": 0.20},
        **args,
    )
    for field in ["history", "fills", "open_lots", "incidents"]:
        pd.testing.assert_frame_equal(getattr(old, field), getattr(new, field))
    if reinvest:
        pd.testing.assert_frame_equal(old.decisions, new.decisions)


def test_zero_all_sectors_keeps_account_in_cash():
    days = pd.bdate_range("2015-01-01", "2015-01-09")
    quotes = pd.DataFrame(100.0, index=days, columns=["X"])
    scores = pd.DataFrame({"date": [days[0]], "symbol": ["X"], "score": [1.0]})
    result = sweep_backtest(
        quotes,
        pd.Series(100.0, index=days),
        scores,
        days[1],
        days[-1],
        sectors={"X": "Energy"},
        sector_cap={"Energy": 0.0},
    )
    assert result.fills.empty
    assert result.history.nav.eq(1_000_000.0).all()
