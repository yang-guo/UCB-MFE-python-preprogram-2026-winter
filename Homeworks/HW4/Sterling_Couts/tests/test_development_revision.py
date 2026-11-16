import numpy as np
import pandas as pd
import pytest

from development_revision import normalize_prices, select_entries


def test_reverse_split_removes_unit_change_without_changing_raw_frame():
    raw = pd.DataFrame(
        {
            "symbol": ["X"] * 3,
            "date": pd.to_datetime(["2010-05-26", "2010-05-27", "2010-05-28"]),
            "adj_close": [1.0, 2.0, 20.0],
        }
    )
    registry = {
        "scope_start": "2008-01-01",
        "scope_end": "2015-12-31",
        "repairs": [
            {
                "symbol": "X",
                "previous_session": "2010-05-27",
                "boundary": "2010-05-28",
                "expected_previous": 2.0,
                "expected_boundary": 20.0,
                "side": "before",
                "factor": 10.0,
            }
        ],
    }
    corrected = normalize_prices(raw, registry)
    assert corrected.adj_close.to_list() == [10.0, 20.0, 20.0]
    assert raw.adj_close.to_list() == [1.0, 2.0, 20.0]
    assert (
        corrected.adj_close.pct_change().iloc[1] == raw.adj_close.pct_change().iloc[1]
    )
    assert corrected.adj_close.pct_change().iloc[-1] == 0
    with pytest.raises(ValueError, match="differs"):
        normalize_prices(corrected, registry)


def test_price_overlay_rejects_extension_beyond_reviewed_period():
    raw = pd.DataFrame(
        {"symbol": ["X"], "date": pd.to_datetime(["2016-01-04"]), "adj_close": [10.0]}
    )
    with pytest.raises(ValueError, match="restricted"):
        normalize_prices(raw, {"scope_end": "2015-12-31"})


def test_two_reverse_splits_preserve_flat_economic_value():
    # One share at $2 becomes half a share at $4, then one sixth at $12.
    # On one consistent share basis there is no investment gain at either split.
    dates = pd.to_datetime(["2010-01-04", "2010-01-05", "2010-01-06"])
    raw = pd.DataFrame(
        {"symbol": ["X"] * 3, "date": dates, "adj_close": [2.0, 4.0, 12.0]}
    )
    rules = []
    for before, boundary, old, new, factor in [
        (dates[0], dates[1], 2, 4, 2),
        (dates[1], dates[2], 4, 12, 3),
    ]:
        rules.append(
            {
                "symbol": "X",
                "previous_session": before,
                "boundary": boundary,
                "expected_previous": old,
                "expected_boundary": new,
                "side": "before",
                "factor": factor,
            }
        )
    result = normalize_prices(
        raw, {"scope_start": "2008-01-01", "scope_end": "2015-12-31", "repairs": rules}
    )
    np.testing.assert_allclose(result.adj_close, [12, 12, 12])
    np.testing.assert_allclose(result.adj_close.pct_change().dropna(), 0)


def test_vendor_scale_correction_preserves_subsequent_economic_return():
    # A spurious 16x change must disappear while the next actual 10% gain remains.
    raw = pd.DataFrame(
        {
            "symbol": ["X"] * 3,
            "date": pd.to_datetime(["2013-12-12", "2013-12-13", "2013-12-16"]),
            "close": [10.0, 10.0, 11.0],
            "adj_close": [10.0, 160.0, 176.0],
        }
    )
    registry = {
        "scope_start": "2008-01-01",
        "scope_end": "2015-12-31",
        "repairs": [
            {
                "symbol": "X",
                "previous_session": "2013-12-12",
                "boundary": "2013-12-13",
                "expected_previous": 10.0,
                "expected_boundary": 160.0,
                "side": "on_or_after",
                "factor": 1 / 16,
            }
        ],
    }
    result = normalize_prices(raw, registry)
    np.testing.assert_allclose(result.adj_close.pct_change().dropna(), [0.0, 0.1])
    pd.testing.assert_series_equal(result.close, raw.close)
    assert raw.adj_close.iloc[1] == 160.0


def test_entry_filters_use_known_features_not_future_targets():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-30"] * 3),
            "symbol": ["X", "Y", "Z"],
            "score": [3.0, 2.0, 1.0],
            "target": [np.nan, 1000.0, -1.0],
        }
    )
    panel = scores[["date", "symbol"]].assign(
        momentum_63=[0.1, -0.2, 0.0], volatility_63=[0.3, 0.5, 0.4]
    )
    assert select_entries(scores, panel, "positive_63d_trend").symbol.to_list() == ["X"]
    assert select_entries(scores, panel, "volatility_40pct").symbol.to_list() == [
        "X",
        "Z",
    ]
    changed = scores.assign(target=[-1000, np.nan, 1000])
    assert select_entries(changed, panel, "positive_63d_trend").symbol.to_list() == [
        "X"
    ]
    pd.testing.assert_frame_equal(select_entries(scores, panel, "unchanged"), scores)
