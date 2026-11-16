"""Hand cases guarding annual boundaries and immature-label leakage."""

import pandas as pd
import pytest

from delay_stability import check_fit_chronology, scores_for_entry_year


def fixture():
    calendar = pd.to_datetime(
        ["2012-12-31", "2013-01-02", "2013-12-30", "2013-12-31", "2014-01-02"]
    )
    scores = pd.DataFrame(
        dict(date=calendar, symbol="A", target=[0.1, 0.2, None, 0.3, None])
    )
    return calendar, scores


def test_slice_uses_entry_year_and_keeps_missing_outcomes():
    calendar, scores = fixture()
    observed = scores_for_entry_year(scores, calendar, 2013)
    assert observed.date.tolist() == list(calendar[:3])
    assert pd.isna(observed.target.iloc[-1])
    assert scores_for_entry_year(scores, calendar, 2014).date.tolist() == [calendar[3]]


@pytest.mark.parametrize(
    "problem",
    ["absent_signal", "duplicate_calendar", "unordered_calendar", "final_year"],
)
def test_invalid_calendar_or_scope_rejected(problem):
    calendar, scores = fixture()
    year = 2013
    if problem == "absent_signal":
        scores.loc[0, "date"] = pd.Timestamp("2012-12-30")
    elif problem == "duplicate_calendar":
        calendar = calendar.append(calendar[:1])
    elif problem == "unordered_calendar":
        calendar = calendar[::-1]
    else:
        year = 2016
    with pytest.raises(ValueError):
        scores_for_entry_year(scores, calendar, year)


def chronology():
    audit = pd.DataFrame(
        dict(
            first_training_signal=pd.to_datetime(["2010-01-29"]),
            last_training_signal=pd.to_datetime(["2012-08-31"]),
            last_training_label=pd.to_datetime(["2012-12-03"]),
            fit_date=pd.to_datetime(["2012-12-31"]),
            alpha=[1000],
        )
    )
    scores = pd.DataFrame(
        dict(
            date=pd.to_datetime(["2012-12-31", "2013-01-31"]),
            fit_date=pd.Timestamp("2012-12-31"),
        )
    )
    return audit, scores


def test_valid_chronology():
    check_fit_chronology(*chronology())


@pytest.mark.parametrize(
    "column,value",
    [
        ("first_training_signal", "2009-12-31"),
        ("last_training_label", "2012-12-31"),
        ("fit_date", "2013-01-02"),
    ],
)
def test_invalid_training_chronology_rejected(column, value):
    audit, scores = chronology()
    audit.loc[0, column] = pd.Timestamp(value)
    with pytest.raises(AssertionError):
        check_fit_chronology(audit, scores)
