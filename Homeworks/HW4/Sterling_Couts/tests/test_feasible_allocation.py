from dataclasses import replace

import numpy as np
import pandas as pd

from feasible_allocation import plan_feasible_budgets
from hw4.settings import TradingRules
from ranking_diagnostics import assign_buckets, momentum_scores


def test_research_engine_matches_original_and_new_orders_precede_fill_price():
    from cap_sensitivity_engine import run_sector_backtest as old
    from ranking_research_engine import run_sector_backtest as new

    dates = pd.bdate_range("2013-01-01", "2013-05-01")
    quotes = pd.DataFrame(100.0, index=dates, columns=["E1", "E2", "T"])
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame(
        {"date": [dates[0]] * 3, "symbol": quotes.columns, "score": [3.0, 2.0, 1.0]}
    )
    eligible = quotes.notna()
    rules = replace(
        TradingRules(), initial_cash=1000.0, top_n=2, cohort_fraction=0.5, name_cap=0.5
    )
    kwargs = dict(
        sectors={"E1": "Energy", "E2": "Energy", "T": "Tech"},
        sector_cap=0.2,
        reinvest_maturity_cash=True,
        entry_eligibility=eligible,
    )
    args = (quotes, benchmark, scores, dates[1], dates[-1], rules)
    a, b = old(*args, **kwargs), new(*args, **kwargs)
    for field in ["history", "fills", "open_lots", "incidents", "decisions"]:
        pd.testing.assert_frame_equal(getattr(a, field), getattr(b, field))
    feasible = new(*args, **kwargs, allocator="feasible")
    first = feasible.fills.loc[feasible.fills.date.eq(dates[1])]
    assert first.symbol.tolist() == ["E1", "T"]
    changed = quotes.copy()
    changed.loc[dates[1], "T"] = 150.0
    alternative = new(changed, *args[1:], **kwargs, allocator="feasible")
    other = alternative.fills.loc[alternative.fills.date.eq(dates[1])]
    np.testing.assert_allclose(first.cash_change, other.cash_change)
    due = dates[1] + pd.Timedelta(days=90)
    sale_day = dates[dates.searchsorted(due)]
    sells = feasible.fills.loc[feasible.fills.side.eq("SELL")]
    assert sells.date.eq(sale_day).all() and len(sells) == 2
    next_day = dates[dates.get_loc(sale_day) + 1]
    new_lots = feasible.fills.loc[feasible.fills.date.eq(next_day)]
    assert new_lots.maturity.eq(next_day + pd.Timedelta(days=90)).all()
    assert -new_lots.cash_change.sum() <= sells.cash_change.sum() + 1e-8


def test_diagnostics_preserve_unobserved_candidates_and_match_exit_horizon():
    from ranking_diagnostics import describe_scores

    date = pd.Timestamp("2013-01-01")
    entry = pd.Timestamp("2013-01-02")
    end = pd.Timestamp("2013-04-02")
    scores = pd.DataFrame(
        {
            "date": [date] * 10,
            "symbol": list("ABCDEFGHIJ"),
            "score": range(10),
            "target": [0.2] * 9 + [np.nan],
            "label_end": [end] * 10,
        }
    )
    panel = scores[["date", "symbol"]].assign(
        entry_date=entry, scheduled_exit=end, sector="Test"
    )
    benchmark = pd.Series([100.0, 110.0], index=[entry, end])
    tables = describe_scores(scores, panel, benchmark, "2013-12-31", "test", "test")
    summary = tables["decile_summary"]
    assert summary.candidates.sum() == 10 and summary.observed.sum() == 9
    assert summary.loc[summary.decile.eq(10), "observed"].iloc[0] == 0
    np.testing.assert_allclose(summary.mean_excess.dropna(), 0.1)
    future = describe_scores(scores, panel, benchmark, "2013-03-31", "test", "test")
    assert future["decile_summary"].observed.sum() == 0


def test_blocked_candidate_is_replaced_without_spending_same_sector_room_twice():
    rules = replace(
        TradingRules(), initial_cash=1000.0, top_n=2, name_cap=0.5, fee_bps=0.0
    )
    scores = pd.Series({"E": 3.0, "T": 2.0, "U": 1.0})
    sectors = {"E": "Energy", "OLD": "Energy", "T": "Technology", "U": "Utilities"}
    orders = plan_feasible_budgets(
        scores, 900.0, 1000.0, {"OLD": 100.0}, rules, sectors, 0.1
    )
    assert orders == {"T": 100.0, "U": 100.0}
    assert "E" not in orders


def test_leftover_ticket_is_redistributed_within_name_and_sector_limits():
    rules = replace(
        TradingRules(), top_n=3, name_cap=0.2, cohort_fraction=0.3, fee_bps=0.0
    )
    orders = plan_feasible_budgets(
        pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}),
        810.0,
        1000.0,
        {"A": 190.0},
        rules,
        {"A": "a", "B": "b", "C": "c"},
        1.0,
    )
    assert orders == {"A": 10.0, "B": 145.0, "C": 145.0}


def test_reinvestment_proceeds_limit_includes_fees_and_does_not_increase_cash():
    rules = replace(TradingRules(), top_n=2, name_cap=0.5, fee_bps=10.0)
    orders = plan_feasible_budgets(
        pd.Series({"A": 2.0, "B": 1.0}),
        1000.0,
        1000.0,
        {},
        rules,
        {"A": "a", "B": "a"},
        0.1,
        budget_limit=80.0,
    )
    np.testing.assert_allclose(sum(orders.values()), 80.0)
    assert sum(orders.values()) / 1.001 <= 100.0


def test_buckets_do_not_depend_on_target_availability_or_split_equal_scores():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2013-01-01")] * 10,
            "score": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "target": range(10),
        }
    )
    first = assign_buckets(frame)
    changed = frame.copy()
    changed["target"] = np.nan
    pd.testing.assert_series_equal(first.decile, assign_buckets(changed).decile)
    assert first.decile.iloc[0] == first.decile.iloc[1]


def test_trend_rank_uses_only_existing_momentum_and_keeps_candidate_keys():
    dates = [pd.Timestamp("2013-01-01")] * 2
    scores = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["A", "B"],
            "score": [0.2, 0.1],
            "target": [99.0, -99.0],
        }
    )
    panel = pd.DataFrame(
        {"date": dates, "symbol": ["A", "B"], "momentum_252_21": [-0.1, 0.4]}
    )
    got = momentum_scores(scores, panel)
    assert got.symbol.tolist() == ["A", "B"]
    assert got.score.tolist() == [-0.1, 0.4]
    changed = momentum_scores(scores.assign(target=np.nan), panel)
    pd.testing.assert_series_equal(got.score, changed.score)
