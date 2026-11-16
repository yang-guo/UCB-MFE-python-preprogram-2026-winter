"""Independent timing, causality and holding checks for the entry-only revision."""

import numpy as np
import pandas as pd
import pytest

from entry_universe import annotate_entry_ranks, select_entry_universe
from hw4.engine import run_backtest
from hw4.settings import TradingRules


def membership():
    return pd.DataFrame(
        {
            "symbol": ["X", "X", "Y", "Y"],
            "quarter_start": pd.to_datetime(["2023-01-01", "2023-04-01"] * 2),
            "selection_date": pd.to_datetime(["2022-12-30", "2023-03-31"] * 2),
            "effective_start": pd.to_datetime(["2023-01-02", "2023-04-03"] * 2),
            "universe_rank": [500, 501, 501, 500],
        }
    )


def test_entry_quarter_and_inclusive_cutoff_not_signal_quarter():
    calendar = pd.bdate_range("2023-03-29", "2023-04-04")
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-03-30"] * 2 + ["2023-03-31"] * 2),
            "symbol": ["X", "Y", "X", "Y"],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    annotated = annotate_entry_ranks(scores, membership(), calendar)
    kept = select_entry_universe(annotated)
    assert list(zip(kept.date, kept.symbol)) == [
        (pd.Timestamp("2023-03-30"), "X"),
        (pd.Timestamp("2023-03-31"), "Y"),
    ]
    assert kept.entry_date.tolist() == [
        pd.Timestamp("2023-03-31"),
        pd.Timestamp("2023-04-03"),
    ]


def test_missing_future_target_and_inactive_snapshot_do_not_exclude():
    calendar = pd.bdate_range("2023-01-30", "2023-02-03")
    scores = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-31")],
            "symbol": ["X"],
            "score": [0.12345],
            "target": [np.nan],
            "label_end": [pd.NaT],
        }
    )
    members = membership().assign(
        is_active_snapshot=False, last_price_date=pd.Timestamp("2023-02-02")
    )
    kept = select_entry_universe(annotate_entry_ranks(scores, members, calendar))
    assert kept.symbol.tolist() == ["X"]
    assert kept.score.iloc[0] == 0.12345
    assert kept.target.isna().all()


def test_future_quarter_changes_cannot_change_previous_decisions():
    calendar = pd.bdate_range("2023-01-30", "2023-02-03")
    scores = pd.DataFrame(
        {"date": [pd.Timestamp("2023-01-31")], "symbol": ["X"], "score": [0.1]}
    )
    before = select_entry_universe(annotate_entry_ranks(scores, membership(), calendar))
    later = membership()
    later.loc[later.quarter_start.eq("2023-04-01"), "universe_rank"] = [1, 2000]
    after = select_entry_universe(annotate_entry_ranks(scores, later, calendar))
    pd.testing.assert_frame_equal(before, after)


def test_future_announced_membership_is_rejected():
    calendar = pd.bdate_range("2023-03-30", "2023-04-04")
    scores = pd.DataFrame(
        {"date": [pd.Timestamp("2023-03-31")], "symbol": ["Y"], "score": [0.1]}
    )
    members = membership()
    members.loc[members.quarter_start.eq("2023-04-01"), "selection_date"] = (
        pd.Timestamp("2023-04-03")
    )
    with pytest.raises(ValueError, match="not yet known"):
        annotate_entry_ranks(scores, members, calendar)


def test_rank_exit_does_not_liquidate_or_reset_existing_lot():
    calendar = pd.bdate_range("2023-01-30", "2023-05-05")
    quotes = pd.DataFrame({"X": 10.0}, index=calendar)
    benchmark = pd.Series(100.0, index=calendar)
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-31", "2023-03-31"]),
            "symbol": ["X", "X"],
            "score": [0.1, 0.2],
        }
    )
    kept = select_entry_universe(annotate_entry_ranks(scores, membership(), calendar))
    assert len(kept) == 1
    rules = TradingRules(
        initial_cash=1000, top_n=1, cohort_fraction=1, name_cap=1, fee_bps=0
    )
    result = run_backtest(quotes, benchmark, kept, "2023-02-01", "2023-05-05", rules)
    assert result.fills.side.tolist() == ["BUY", "SELL"]
    assert result.fills.date.tolist() == [
        pd.Timestamp("2023-02-01"),
        pd.Timestamp("2023-05-02"),
    ]
    assert result.history.loc["2023-04-03", "open_lots"] == 1
    assert result.history.nav.iloc[-1] == pytest.approx(1000)


def test_duplicate_or_missing_membership_fails_instead_of_silent_drop():
    calendar = pd.bdate_range("2023-01-30", "2023-02-03")
    scores = pd.DataFrame(
        {"date": [pd.Timestamp("2023-01-31")], "symbol": ["X"], "score": [0.1]}
    )
    members = membership()
    with pytest.raises(ValueError, match="Duplicate"):
        annotate_entry_ranks(scores, pd.concat([members, members.iloc[:1]]), calendar)
    with pytest.raises(ValueError, match="lacks"):
        annotate_entry_ranks(scores, members.loc[members.symbol.eq("Y")], calendar)
