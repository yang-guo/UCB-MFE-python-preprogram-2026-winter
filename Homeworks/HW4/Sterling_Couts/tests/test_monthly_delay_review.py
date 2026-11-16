"""Hand-calculated boundaries for the independent review, not more strategy tuning."""

import numpy as np
import pandas as pd
import pytest

from monthly_delay_review import direct_financial_record, drawdown_episode


def records():
    return pd.DataFrame(
        dict(
            fiscal_date=pd.to_datetime(
                ["2012-12-31", "2013-03-31", "2012-09-30", "2013-06-30"]
            ),
            filing_date=pd.to_datetime(
                ["2013-02-01", "2013-05-01", "2013-05-05", "2013-06-29"]
            ),
            period=["quarter"] * 4,
            total_assets=[100.0] * 4,
            operating_income=[10.0, 20.0, 999.0, 999.0],
            free_cash_flow=[5.0, 8.0, 999.0, 999.0],
        )
    )


@pytest.mark.parametrize(
    "signal,lag,expected",
    [
        ("2013-02-01", 0, np.nan),
        ("2013-02-02", 0, 0.1),
        ("2013-03-03", 30, np.nan),
        ("2013-03-04", 30, 0.1),
        ("2013-05-06", 0, 0.2),
        ("2013-07-01", 0, 0.2),
    ],
)
def test_exact_availability_and_invalid_lags(signal, lag, expected):
    np.testing.assert_allclose(
        direct_financial_record(records(), signal, lag)["operating_roa"],
        expected,
        equal_nan=True,
    )


def test_future_mutations_cannot_change_known_record():
    original = records()
    changed = original.copy()
    changed.loc[changed.filing_date.gt("2013-03-04"), "operating_income"] = -1e20
    assert direct_financial_record(
        original, "2013-03-04", 30
    ) == direct_financial_record(changed, "2013-03-04", 30)


def test_invalid_denominators_are_missing_not_zero():
    f = records().iloc[:1].copy()
    f.loc[:, "total_assets"] = 0.0
    assert np.isnan(direct_financial_record(f, "2013-03-04", 0)["operating_roa"])


def test_stale_boundary():
    f = records().iloc[:1]
    start = f.fiscal_date.iloc[0]
    assert (
        direct_financial_record(f, start + pd.Timedelta(days=550), 0)["operating_roa"]
        == 0.1
    )
    assert np.isnan(
        direct_financial_record(f, start + pd.Timedelta(days=551), 0)["operating_roa"]
    )


def test_missing_filing_and_annual_record_excluded():
    f = records().iloc[:2].copy()
    f.loc[f.index[0], "filing_date"] = pd.NaT
    f.loc[f.index[1], "period"] = "annual"
    assert pd.isna(direct_financial_record(f, "2013-07-01", 0)["available_date"])


@pytest.mark.parametrize("lag", [-1, 2.5])
def test_invalid_delay_rejected(lag):
    with pytest.raises(ValueError):
        direct_financial_record(records(), "2013-07-01", lag)


def test_drawdown_peak_trough_and_recovery():
    dates = pd.date_range("2013-01-01", periods=6)
    nav = pd.Series([100.0, 120.0, 120.0, 90.0, 100.0, 125.0], index=dates)
    result = drawdown_episode(nav)
    assert result["max_drawdown"] == -0.25
    assert result["peak_date"] == dates[2] and result["trough_date"] == dates[3]
    assert result["recovery_date"] == dates[5]
    assert pd.isna(drawdown_episode(nav.iloc[:5])["recovery_date"])
