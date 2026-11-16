from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hw4.engine import plan_budgets, run_backtest
from hw4.settings import TradingRules
from sector_allocation import plan_sector_budgets
from sector_engine import run_sector_backtest


def test_sector_room_aggregates_existing_names_and_scales_orders_proportionally():
    rules = replace(TradingRules(), top_n=4, cohort_fraction=1)
    scores = pd.Series({"E1": 4.0, "E2": 3.0, "T1": 2.0, "T2": 1.0})
    sectors = {
        "E0": "Energy",
        "E1": "Energy",
        "E2": "Energy",
        "T0": "Technology",
        "T1": "Technology",
        "T2": "Technology",
    }
    budgets = plan_sector_budgets(
        scores, 700.0, 1000.0, {"E0": 180.0, "T0": 120.0}, rules, sectors, 0.2
    )
    # Energy has $20 room; Technology has $80, split equally between two orders.
    cost = 1 + rules.fee_bps / 10000
    np.testing.assert_allclose(
        [budgets[s] / cost for s in ["E1", "E2", "T1", "T2"]], [10.0, 10.0, 40.0, 40.0]
    )
    assert sum(budgets.values()) < 700


def test_overweight_sector_blocks_new_orders_without_forcing_a_sale():
    rules = TradingRules()
    orders = plan_sector_budgets(
        pd.Series({"X": 1.0}),
        700.0,
        1000.0,
        {"OLD": 300.0},
        rules,
        {"X": "Energy", "OLD": "Energy"},
        0.2,
    )
    assert orders == {}


def test_disabled_sector_cap_matches_original_planner_and_unknown_sector_fails():
    rules = TradingRules()
    scores = pd.Series({"X": 1.0})
    assert plan_sector_budgets(
        scores, 1000.0, 1000.0, {}, rules, {}, None
    ) == plan_budgets(scores, 1000.0, 1000.0, {}, rules)
    with pytest.raises(ValueError, match="sector label"):
        plan_sector_budgets(scores, 1000.0, 1000.0, {}, rules, {}, 0.2)


def example_account():
    dates = pd.bdate_range("2013-01-01", "2013-05-31")
    quotes = pd.DataFrame(
        {"X": np.linspace(100, 130, len(dates)), "Y": np.linspace(100, 90, len(dates))},
        index=dates,
    )
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[25], dates[25]],
            "symbol": ["X", "Y", "X", "Y"],
            "score": [2.0, 1.0, 2.0, 1.0],
        }
    )
    return dates, quotes, benchmark, scores


def test_sector_engine_disabled_cap_reproduces_original_ledgers_exactly():
    dates, quotes, benchmark, scores = example_account()
    old = run_backtest(quotes, benchmark, scores, dates[1], dates[-1])
    new = run_sector_backtest(
        quotes, benchmark, scores, dates[1], dates[-1], sector_cap=None
    )
    for field in ["history", "fills", "open_lots", "incidents"]:
        pd.testing.assert_frame_equal(getattr(new, field), getattr(old, field))
    assert new.status == old.status


def test_sector_entry_budget_is_decided_before_fill_price_and_maturity_is_preserved():
    dates, quotes, benchmark, scores = example_account()
    sectors = {"X": "Energy", "Y": "Energy"}
    rules = replace(TradingRules(), name_cap=0.05)
    original = run_sector_backtest(
        quotes,
        benchmark,
        scores,
        dates[1],
        dates[-1],
        rules,
        sectors=sectors,
        sector_cap=0.06,
    )
    shifted = quotes.copy()
    shifted.loc[dates[1], "X"] *= 1.4
    other = run_sector_backtest(
        shifted,
        benchmark,
        scores,
        dates[1],
        dates[-1],
        rules,
        sectors=sectors,
        sector_cap=0.06,
    )
    first = original.fills.loc[original.fills.date.eq(dates[1])]
    altered = other.fills.loc[other.fills.date.eq(dates[1])]
    np.testing.assert_allclose(first.cash_change, altered.cash_change)
    buys = original.fills.loc[original.fills.side.eq("BUY")].set_index("lot_id")
    sells = original.fills.loc[original.fills.side.eq("SELL")]
    assert len(sells) > 0
    for row in sells.itertuples():
        due = buys.loc[row.lot_id, "date"] + pd.Timedelta(days=90)
        assert row.date == dates[dates.searchsorted(due)]
