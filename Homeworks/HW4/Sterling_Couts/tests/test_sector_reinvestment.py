"""Small cash/timing examples independent of the historical model's results."""

from dataclasses import replace

import numpy as np
import pandas as pd

from hw4.settings import TradingRules
from sector_allocation import plan_sector_budgets
from sector_engine import run_sector_backtest
from sector_reinvestment import eligibility_at_entry


def setup_account():
    dates = pd.bdate_range("2013-01-01", "2013-05-01")
    quotes = pd.DataFrame(100.0, index=dates, columns=["X", "Y"])
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame(
        {"date": [dates[0]] * 2, "symbol": ["X", "Y"], "score": [2.0, 1.0]}
    )
    rules = replace(
        TradingRules(),
        initial_cash=1000.0,
        top_n=2,
        cohort_fraction=0.5,
        name_cap=0.5,
        fee_bps=10,
    )
    eligible = pd.DataFrame(True, index=dates, columns=quotes.columns)
    return quotes, benchmark, scores, rules, eligible


def execute(quotes, benchmark, scores, rules, eligible):
    return run_sector_backtest(
        quotes,
        benchmark,
        scores,
        quotes.index[1],
        quotes.index[-1],
        rules,
        sectors={"X": "Energy", "Y": "Technology"},
        sector_cap=1.0,
        reinvest_maturity_cash=True,
        entry_eligibility=eligible,
    )


def test_sales_fund_only_next_session_and_new_lots_get_new_maturity():
    q, b, s, rules, eligible = setup_account()
    result = execute(q, b, s, rules, eligible)
    first, sells = result.fills.iloc[:2], result.fills.query("side == 'SELL'")
    due = first.date.iloc[0] + pd.Timedelta(days=90)
    sale_day = q.index[q.index.searchsorted(due)]
    next_day = q.index[q.index.get_loc(sale_day) + 1]
    assert sells.date.eq(sale_day).all() and len(sells) == 2
    assert not result.fills.query("side == 'BUY'").date.eq(sale_day).any()
    new = result.fills.loc[result.fills.date.eq(next_day)]
    expected_cash = 500 * (1 - 0.001) / (1 + 0.001)
    np.testing.assert_allclose(sells.cash_change.sum(), expected_cash)
    np.testing.assert_allclose(-new.cash_change.sum(), expected_cash)
    assert new.maturity.eq(next_day + pd.Timedelta(days=90)).all()
    assert new.lot_id.min() > first.lot_id.max()
    assert result.history.cash.ge(0).all()
    assert len(result.decisions) == 2  # No repeated daily recycling of unused cash.


def test_current_eligibility_blocks_stale_ranking_and_unused_cash_is_not_retried():
    q, b, s, rules, eligible = setup_account()
    due = q.index[q.index.searchsorted(q.index[1] + pd.Timedelta(days=90))]
    next_day = q.index[q.index.get_loc(due) + 1]
    eligible.loc[next_day, "X"] = False
    result = execute(q, b, s, rules, eligible)
    buys = result.fills.query("side == 'BUY'")
    assert buys.loc[buys.date.eq(next_day), "symbol"].tolist() == ["Y"]
    assert not buys.date.gt(next_day).any()
    # Changing the future fill cannot alter a decision already made.
    changed = q.copy()
    changed.loc[next_day, "Y"] = 140.0
    other = execute(changed, b, s, rules, eligible)
    np.testing.assert_allclose(
        buys.loc[buys.date.eq(next_day), "cash_change"],
        other.fills.loc[other.fills.date.eq(next_day), "cash_change"],
    )


def test_monthly_and_reinvestment_overlap_creates_only_one_monthly_batch():
    q, b, s, rules, eligible = setup_account()
    due = q.index[q.index.searchsorted(q.index[1] + pd.Timedelta(days=90))]
    s = pd.concat([s, s.assign(date=due, score=[1.0, 2.0])], ignore_index=True)
    result = execute(q, b, s, rules, eligible)
    next_day = q.index[q.index.get_loc(due) + 1]
    decision = result.decisions.loc[result.decisions.date.eq(next_day)]
    assert len(decision) == 1 and decision.trigger.iloc[0] == "monthly"
    assert decision.model_score_date.iloc[0] == due
    buys = result.fills.loc[
        result.fills.date.eq(next_day) & result.fills.side.eq("BUY")
    ]
    assert len(buys) == 2 and buys.symbol.tolist() == ["Y", "X"]
    assert -buys.cash_change.sum() <= decision.prior_nav.iloc[0] * 0.5 + 1e-9


def test_budget_limit_precedes_caps_and_preserves_cash_nav_identity():
    rules = replace(TradingRules(), name_cap=0.5, top_n=2)
    orders = plan_sector_budgets(
        pd.Series({"X": 2.0, "Y": 1.0}),
        800.0,
        1000.0,
        {"OLD": 200.0},
        rules,
        {"X": "Energy", "Y": "Technology", "OLD": "Energy"},
        0.2,
        budget_limit=100.0,
    )
    assert "X" not in orders
    assert orders == {"Y": 50.0}  # Blocked $50 stays cash; no redistribution.


def test_extra_batch_carries_latest_monthly_scores_instead_of_initial_scores():
    q, b, s, rules, eligible = setup_account()
    rules = replace(rules, top_n=1, cohort_fraction=0.25)
    newer = q.index[20]
    s = pd.concat([s, s.assign(date=newer, score=[1.0, 2.0])], ignore_index=True)
    result = execute(q, b, s, rules, eligible)
    extra = result.decisions.query("trigger == 'maturity_reinvestment'").iloc[0]
    assert extra.model_score_date == newer
    purchases = result.fills.loc[
        result.fills.date.eq(extra.date) & result.fills.side.eq("BUY")
    ]
    assert purchases.symbol.tolist() == ["Y"]


def test_entry_quarter_membership_and_screen_use_only_previous_session():
    days = pd.to_datetime(["2013-03-27", "2013-03-28", "2013-04-01", "2013-04-02"])
    screen = pd.DataFrame({"X": [True, True, False, True], "Y": [True] * 4}, index=days)
    members = pd.DataFrame(
        {
            "symbol": ["X", "X", "Y"],
            "quarter_start": pd.to_datetime(["2013-01-01", "2013-04-01", "2013-04-01"]),
            "selection_date": pd.to_datetime(
                ["2012-12-31", "2013-03-28", "2013-04-01"]
            ),
        }
    )
    actual = eligibility_at_entry(screen, members)
    assert actual.X.tolist() == [False, True, True, False]
    assert actual.Y.tolist() == [False, False, False, True]
    changed = screen.copy()
    changed.iloc[-1] = False
    pd.testing.assert_frame_equal(actual, eligibility_at_entry(changed, members))
