import numpy as np
import pandas as pd
import pytest

from hw4.data import holding_labels, stock_features, financial_events, join_financials
from hw4.modeling import ModelSpec, make_model, training_rows


def test_features_are_causal_and_do_not_fill_gaps():
    calendar = pd.bdate_range("2010-01-01", periods=330)
    p = pd.DataFrame(
        {
            "date": calendar,
            "close": np.arange(330) + 100.0,
            "adj_close": np.arange(330) + 100.0,
            "volume": 100,
        }
    )
    full = stock_features(p, calendar)
    prefix = stock_features(p.iloc[:290], calendar[:290])
    pd.testing.assert_frame_equal(full.iloc[:290], prefix)
    p.loc[275, "adj_close"] = np.nan
    with_gap = stock_features(p, calendar)
    assert not with_gap.iloc[280].eligible_history
    assert np.isnan(with_gap.iloc[280].volatility_63)


def test_labels_use_next_close_and_calendar_days_not_trading_rows():
    calendar = pd.bdate_range("2022-12-30", "2023-05-05")
    quotes = pd.Series(10.0, index=calendar)
    quotes.loc["2023-04-03":] = 12.0
    row = holding_labels([calendar[0]], quotes, calendar).iloc[0]
    assert row.entry_date == pd.Timestamp("2023-01-02")
    assert row.label_end == pd.Timestamp("2023-04-03")
    assert row.target == pytest.approx(0.2)
    terminal = holding_labels([calendar[-2]], quotes, calendar).iloc[0]
    assert np.isnan(terminal.target)
    assert pd.notna(terminal.entry_date)


def test_fit_requires_mature_labels_not_just_earlier_signals():
    panel = pd.DataFrame(
        {
            "eligible": [True] * 3,
            "date": pd.to_datetime(["2012-10-31", "2012-11-30", "2012-12-31"]),
            "label_end": pd.to_datetime(["2013-01-30", "2013-03-01", None]),
            "target": [0.1, 0.2, np.nan],
        }
    )
    rows = training_rows(panel, pd.Timestamp("2013-02-01"))
    assert rows.index.tolist() == [0]


def test_preprocessing_is_fitted_once_and_batch_independent():
    spec = ModelSpec("tiny", ("x", "z"))
    model = make_model(spec)
    X = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, 4.0, 5.0], "z": [0.0, 1.0, 2.0, 3.0, 4.0]}
    )
    model.fit(X, [0.1, 0.2, 0.3, 0.4, 0.5])
    bounds = model["clip"].bounds_.copy()
    new = pd.DataFrame({"x": [10.0, 1000.0], "z": [2.0, -1000.0]})
    assert model.predict(new.iloc[:1])[0] == pytest.approx(model.predict(new)[0])
    np.testing.assert_array_equal(bounds, model["clip"].bounds_)


def test_target_clipping_learns_only_training_bounds_and_preserves_y():
    from hw4.preprocessing import TargetClippedRegressor
    from sklearn.linear_model import Ridge

    X = np.arange(100.0).reshape(-1, 1)
    y = np.linspace(-0.1, 0.1, 100)
    y[-1] = 50000.0
    original = y.copy()
    model = TargetClippedRegressor(quantiles=(0.01, 0.99)).fit(X, y)
    expected = np.quantile(y, [0.01, 0.99])
    np.testing.assert_array_equal(model.target_bounds_, expected)
    reference = Ridge(alpha=100).fit(X, np.clip(y, *expected))
    np.testing.assert_allclose(model.predict(X), reference.predict(X), rtol=1e-12)
    np.testing.assert_array_equal(y, original)
    assert model.clipped_labels_ == 2
    # Predicting arbitrary future observations cannot change the learned bounds.
    model.predict([[100000.0]])
    np.testing.assert_array_equal(model.target_bounds_, expected)


def test_named_features_cannot_be_silently_reordered():
    model = make_model(ModelSpec("tiny", ("x", "z")))
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "z": [4.0, 1.0, 2.0]})
    model.fit(X, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="names and order"):
        model.predict(X[["z", "x"]])


def test_historical_risk_screen_never_uses_future_jumps():
    calendar = pd.bdate_range("2010-01-01", periods=330)
    p = pd.DataFrame(
        {"date": calendar, "adj_close": 100.0, "close": 100.0, "volume": 100}
    )
    before = stock_features(p, calendar)
    assert before.iloc[280].risk_screen_pass
    p.loc[290:, "adj_close"] = 1000.0
    after = stock_features(p, calendar)
    assert after.iloc[280].risk_screen_pass
    assert not after.iloc[290].risk_screen_pass
    pd.testing.assert_frame_equal(before.iloc[:290], after.iloc[:290])


def test_financials_wait_until_day_after_filing_and_expire():
    financials = pd.DataFrame(
        {
            "symbol": ["X", "X"],
            "period": ["quarter"] * 2,
            "fiscal_date": pd.to_datetime(["2012-12-31", "2012-09-30"]),
            "filing_date": pd.to_datetime(["2013-02-01", "2013-02-03"]),
            "total_assets": [100.0, 100.0],
            "operating_income": [10.0, 99.0],
            "revenue": [50.0, 50.0],
            "net_income": [5.0, 5.0],
            "total_debt": [10.0, 10.0],
            "cash_and_equivalents": [10.0, 10.0],
            "free_cash_flow": [2.0, 2.0],
            "operating_cash_flow": [3.0, 3.0],
        }
    )
    events = financial_events(financials)
    # An older fiscal quarter filed late must not displace the newer period.
    assert len(events) == 1
    dates = pd.to_datetime(["2013-02-01", "2013-02-04", "2015-01-01"])
    panel = pd.DataFrame({"symbol": "X", "date": dates})
    joined = join_financials(
        panel, events, pd.DataFrame({"symbol": ["X"], "sector": ["Test"]})
    )
    assert np.isnan(joined.iloc[0].operating_roa)
    assert joined.iloc[1].operating_roa == pytest.approx(0.1)
    assert np.isnan(joined.iloc[2].operating_roa)
