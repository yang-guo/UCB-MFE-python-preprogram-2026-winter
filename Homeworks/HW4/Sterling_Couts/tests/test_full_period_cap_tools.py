import numpy as np
import pandas as pd
import pytest

from full_period_cap_tools import best_tested, calendar_year_returns, refinement_grid


def test_selection_excludes_unresolved_accounts_and_reports_ties():
    rows = [
        dict(
            variant=str(cap),
            cap=cap,
            status="complete",
            specific_open_data_issue="",
            final_value=value,
        )
        for cap, value in [(0.01, 100.0), (0.02, 120.0), (0.03, 120.0)]
    ]
    rows += [
        dict(
            variant="flagged",
            cap=0.04,
            status="complete",
            specific_open_data_issue="price gap",
            final_value=200.0,
        ),
        dict(
            variant="blocked",
            cap=0.05,
            status="blocked_data_gap",
            specific_open_data_issue="",
            final_value=300.0,
        ),
    ]
    chosen, ties, maximum = best_tested(rows, "cap")
    assert chosen["cap"] == 0.03 and ties == [0.02, 0.03] and maximum == 120
    with pytest.raises(ValueError, match="No completed"):
        best_tested(rows[-2:], "cap")


def test_refinement_is_bounded_and_does_not_retune_unrestricted_winner():
    coarse = [n / 20 for n in range(21)]
    assert refinement_grid(0.05, coarse) == [
        0.01,
        0.02,
        0.03,
        0.04,
        0.06,
        0.07,
        0.08,
        0.09,
    ]
    assert refinement_grid(0.0, coarse) == [0.01, 0.02, 0.03, 0.04]
    assert refinement_grid(1.0, coarse) == []


def test_calendar_contributions_compound_without_restarting_account():
    days = pd.to_datetime(
        [
            "2012-12-31",
            "2013-01-02",
            "2013-12-31",
            "2014-01-02",
            "2014-12-31",
            "2015-01-02",
            "2015-12-31",
        ]
    )
    h = pd.DataFrame(
        {
            "nav": [100, 100, 110, 110, 99, 99, 118.8],
            "benchmark_nav": [100, 100, 120, 120, 120, 120, 132],
        },
        index=days,
    )
    r = calendar_year_returns(h)
    np.testing.assert_allclose(r.strategy_return, [0.1, -0.1, 0.2])
    np.testing.assert_allclose(r.benchmark_return, [0.2, 0.0, 0.1])
