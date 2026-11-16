"""Counterexamples for censored labels and prospective financial information."""

import numpy as np
import pandas as pd

from diagnose_development_winner import forecast_groups
from fundamental_changes import build_change_events


def test_decile_membership_keeps_unknown_outcomes_and_no_fake_zero_returns():
    calendar = pd.bdate_range("2013-01-01", periods=4)
    scores = pd.DataFrame(
        {
            "date": [calendar[0]] * 30 + [calendar[1]] * 30,
            "symbol": list(map(str, range(30))) * 2,
            "score": list(range(30)) * 2,
            "target": list(np.arange(30) / 100) + [np.nan] * 30,
            "label_end": [pd.Timestamp("2013-04-10")] * 30 + [pd.NaT] * 30,
        }
    )
    scores.loc[29, "target"] = np.nan  # The highest-ranked name lacks an outcome.
    monthly, bins = forecast_groups(scores, calendar)
    first = monthly.iloc[0]
    assert first.eligible == 30 and first.observed == 29
    assert first.top20 == 20 and first.top20_observed == 19
    assert bins.query("date == @calendar[0] and decile == 10").iloc[0].candidates == 3
    assert bins.query("date == @calendar[0] and decile == 10").iloc[0].observed == 2
    assert pd.isna(monthly.iloc[1].eligible_mean)
    assert pd.isna(monthly.iloc[1].top20_mean)
    assert bins.loc[bins.date.eq(calendar[1]), "mean_return"].isna().all()


def test_appending_a_future_financial_revision_cannot_change_earlier_events():
    base = dict(
        symbol="A",
        period="quarter",
        revenue=100.0,
        operating_income=10.0,
        operating_cash_flow=20.0,
        total_assets=200.0,
        total_debt=80.0,
        cash_and_equivalents=20.0,
        source="course",
    )
    f = pd.DataFrame(
        [
            dict(base, fiscal_date="2012-03-31", filing_date="2012-05-01"),
            dict(base, fiscal_date="2013-03-31", filing_date="2013-05-01"),
        ]
    )
    before = build_change_events(f)
    future = dict(
        base, fiscal_date="2012-03-31", filing_date="2016-06-01", operating_income=500.0
    )
    after = build_change_events(pd.concat([f, pd.DataFrame([future])]))
    pd.testing.assert_frame_equal(before, after)
