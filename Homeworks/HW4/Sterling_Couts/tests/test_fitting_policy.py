"""Hand arithmetic, row identities and causality for averaging fitted models."""

import numpy as np
import pandas as pd
import pytest

from fitting_policy import combine_forecasts


def components():
    anchor = pd.DataFrame(
        dict(
            date=pd.to_datetime(["2013-12-31"] * 3),
            symbol=["A", "B", "C"],
            target=[0.2, -0.1, np.nan],
            label_end=pd.to_datetime(["2014-04-02", "2014-04-02", None]),
            score=[0.1, 0.3, -0.1],
            fit_date=pd.Timestamp("2012-12-31"),
        )
    )
    latest = anchor.assign(score=[0.3, -0.1, 0.1], fit_date=pd.Timestamp("2013-12-31"))
    return anchor, latest


def test_half_average_matches_hand_arithmetic_despite_row_order():
    anchor, latest = components()
    result = combine_forecasts(anchor, latest.iloc[::-1])
    np.testing.assert_allclose(result.score, [0.2, 0.1, 0], atol=1e-15)
    assert result.symbol.tolist() == ["A", "B", "C"]
    assert pd.isna(result.target.iloc[-1]) and pd.isna(result.label_end.iloc[-1])
    assert result.anchor_fit_date.eq("2012-12-31").all()
    assert result.latest_fit_date.eq("2013-12-31").all()


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_endpoints_reproduce_component_predictions(weight):
    anchor, latest = components()
    expected = anchor if weight == 0 else latest
    np.testing.assert_allclose(
        combine_forecasts(anchor, latest, weight).score, expected.score
    )


def test_future_outcomes_cannot_change_forecasts_or_candidates():
    anchor, latest = components()
    before = combine_forecasts(anchor, latest)
    for frame in [anchor, latest]:
        frame["target"] = [1e9, -1e9, 0.7]
        frame["label_end"] = pd.Timestamp("2020-01-01")
    after = combine_forecasts(anchor, latest)
    pd.testing.assert_frame_equal(
        before.drop(columns=["target", "label_end"]),
        after.drop(columns=["target", "label_end"]),
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_candidate",
        "duplicate",
        "future_fit",
        "unknown_fit",
        "infinite_score",
        "different_target",
        "different_exit",
    ],
)
def test_invalid_component_is_rejected(problem):
    anchor, latest = components()
    if problem == "missing_candidate":
        latest = latest.iloc[:-1]
    elif problem == "duplicate":
        latest = pd.concat([latest, latest.iloc[:1]], ignore_index=True)
    elif problem == "future_fit":
        latest.loc[0, "fit_date"] = pd.Timestamp("2014-01-02")
    elif problem == "unknown_fit":
        latest.loc[0, "fit_date"] = pd.NaT
    elif problem == "infinite_score":
        latest.loc[0, "score"] = np.inf
    elif problem == "different_target":
        latest.loc[0, "target"] = 123.0
    else:
        latest.loc[0, "label_end"] = pd.Timestamp("2014-04-03")
    with pytest.raises(ValueError):
        combine_forecasts(anchor, latest)


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError):
        combine_forecasts(*components(), latest_weight=weight)
