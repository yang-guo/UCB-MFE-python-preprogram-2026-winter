import pandas as pd
import pytest

from review_development_risk import assert_development_dates, stock_contributions


def test_diagnostic_rejects_observations_after_cutoff():
    with pytest.raises(ValueError, match="through 2015"):
        assert_development_dates(pd.to_datetime(["2015-12-31", "2016-01-04"]))


def test_cash_flow_attribution_includes_fees_and_unsold_positions():
    # Closed X: spend 101 including entry fee; receive 119 after exit fee => +18.
    # Open Y: spend 201 including fee; retain two units worth 180 => -21.
    fills = pd.DataFrame(
        {"symbol": ["X", "X", "Y"], "cash_change": [-101.0, 119.0, -201.0]}
    )
    lots = pd.DataFrame({"symbol": ["Y"], "units": [2.0]})
    pnl = stock_contributions(fills, lots, pd.Series({"Y": 90.0}))
    assert pnl["X"] == 18 and pnl["Y"] == -21 and pnl.sum() == -3


def test_missing_terminal_mark_is_not_filled_with_zero():
    fills = pd.DataFrame({"symbol": ["Y"], "cash_change": [-201.0]})
    lots = pd.DataFrame({"symbol": ["Y"], "units": [2.0]})
    with pytest.raises(ValueError, match="terminal lot"):
        stock_contributions(fills, lots, pd.Series(dtype=float))
