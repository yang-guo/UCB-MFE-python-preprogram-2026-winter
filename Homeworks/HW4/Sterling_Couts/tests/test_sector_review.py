import numpy as np
import pandas as pd

from review_sector_risk import holdings_exposure, volatility_summary


def test_volatility_summary_uses_entry_year_and_never_requires_future_labels():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-12-31"] * 2),
            "entry_date": pd.to_datetime(["2015-01-02"] * 2),
            "symbol": ["X", "Y"],
            "sector": ["Energy"] * 2,
            "eligible": [True, True],
            "risk_screen_pass": [True, False],
            "volatility_63": [0.3, 0.5],
            "target": [np.nan, -1.0],
        }
    )
    result = volatility_summary(frame)
    annual = result.loc[result.frequency.eq("year")].set_index("scope")
    assert annual.period.eq("2015").all()
    assert annual.loc["before_risk_screen", "median_volatility"] == 0.4
    assert annual.loc["after_risk_screen", "median_volatility"] == 0.3
    changed = frame.assign(target=[1000, np.nan])
    pd.testing.assert_frame_equal(volatility_summary(changed), result)


def test_sector_exposure_aggregates_overlapping_lots_and_distinct_names():
    dates = pd.date_range("2015-01-01", periods=3)
    fills = pd.DataFrame(
        {
            "date": [dates[1], dates[2], dates[2]],
            "symbol": ["X", "X", "Y"],
            "side": ["BUY"] * 3,
            "units": [2.0, 2.0, 1.0],
        }
    )
    history = pd.DataFrame(
        {
            "cash": [1000.0, 800.0, 500.0],
            "holdings": [0.0, 200.0, 500.0],
            "nav": [1000.0] * 3,
        },
        index=dates,
    )
    quotes = pd.DataFrame({"X": [100.0] * 3, "Y": [100.0] * 3}, index=dates)
    sector_map = pd.Series({"X": "Energy", "Y": "Energy"}, name="sector")
    exposure = holdings_exposure(fills, history, quotes, sector_map)
    # Two X lots plus Y are one combined Energy exposure, not three isolated caps.
    np.testing.assert_allclose(exposure.Energy, [0.0, 0.2, 0.5])
