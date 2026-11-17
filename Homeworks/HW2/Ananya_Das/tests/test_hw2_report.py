import sqlite3

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import hw2_report as hr  # noqa: E402


def make_db(path, rows_by_ticker):
    """Write a small ohlc_hw_cleaned table: {ticker: (dates, adj_closes)}."""
    frames = []
    for ticker, (dates, closes) in rows_by_ticker.items():
        closes = np.asarray(closes, dtype=float)
        frames.append(pd.DataFrame({"ts": dates, "ticker": ticker, "open": closes, "high": closes * 1.01,
                                    "low": closes * 0.99, "close": closes, "adj_close": closes,
                                    "volume": 1_000}))
    with sqlite3.connect(path) as con:
        pd.concat(frames).to_sql("ohlc_hw_cleaned", con, index=False)
    return path


def days(n, start="2024-01-02"):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


# ------------------------------------------------------------------ metrics

def test_calculate_returns_is_log_return_and_does_not_fill():
    prices = pd.Series([100.0, 110.0, np.nan, 99.0, 99.0])
    r = hr.calculate_returns(prices)
    assert np.isnan(r[0]) and r[1] == pytest.approx(np.log(1.1))
    assert np.isnan(r[2]) and np.isnan(r[3])          # both sides of a missing price
    assert r[4] == 0.0


def test_volatility_warm_up_and_value():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40))))
    returns = hr.calculate_returns(prices)
    vol = hr.calculate_volatility(returns, window=20)
    assert vol.iloc[:20].isna().all() and vol.iloc[20:].notna().all()   # 20 returns need 21 prices
    expected = returns.iloc[1:21].std(ddof=1) * np.sqrt(252)
    assert vol.iloc[20] == pytest.approx(expected)


def test_drawdown_is_per_date_and_can_recover():
    dd = hr.calculate_max_drawdown(pd.Series([100.0, 120.0, 90.0, 108.0, 130.0]))
    np.testing.assert_allclose(dd, [0, 0, -0.25, -0.10, 0], atol=1e-12)
    assert dd.min() == pytest.approx(-0.25)


def test_sharpe_matches_course_formula():
    r = pd.Series([np.nan, 0.01, -0.005, 0.02, 0.0])
    obs = r.dropna()
    expected = (obs.mean() * 252 - 0.02) / (obs.std(ddof=1) * np.sqrt(252))
    assert hr.calculate_sharpe_ratio(r) == pytest.approx(expected)
    assert hr.calculate_sharpe_ratio(r, risk_free_rate=0.0) == pytest.approx(obs.mean() * 252 / (obs.std() * np.sqrt(252)))


@pytest.mark.parametrize("returns", [pd.Series([0.01, 0.01, 0.01]), pd.Series([0.01]), pd.Series([np.nan, np.nan]),
                                     pd.Series([], dtype=float)])
def test_sharpe_undefined_is_nan(returns):
    assert np.isnan(hr.calculate_sharpe_ratio(returns))


def test_annualized_return_and_volatility_formulas():
    prices = pd.Series([100.0, 101.0, 99.0, 102.0])
    r = hr.calculate_returns(prices)
    n = 3
    assert hr.calculate_annualized_return(r) == pytest.approx(np.exp(np.log(1.02) * 252 / n) - 1)
    assert hr.calculate_annualized_volatility(r) == pytest.approx(r.dropna().std(ddof=1) * np.sqrt(252))
    assert np.isnan(hr.calculate_annualized_return(pd.Series([np.nan])))
    assert np.isnan(hr.calculate_annualized_volatility(pd.Series([np.nan, 0.01])))


def test_summary_statistics_by_hand():
    prices = pd.Series([100.0, 110.0, 110.0, 99.0, 104.5])
    s = hr.summary_statistics(prices, window=2)
    logs = np.log([1.1, 1.0, 0.9, 104.5 / 99])
    assert s["total_return"] == pytest.approx(0.045)
    assert np.expm1(logs.sum()) == pytest.approx(s["total_return"])
    assert s["best_day"] == pytest.approx(logs.max()) and s["worst_day"] == pytest.approx(logs.min())
    assert s["positive_days"] == pytest.approx(2 / 4)          # zero-return day stays in the denominator
    assert s["avg_up_day"] == pytest.approx((logs[0] + logs[3]) / 2)
    assert s["avg_down_day"] == pytest.approx(logs[2])
    assert s["max_drawdown"] == pytest.approx(-0.1)
    assert s["avg_daily_return"] == pytest.approx(logs.mean())
    assert (s["n_prices"], s["n_returns"]) == (5, 4)


def test_summary_statistics_zero_volatility_and_empty_subsets():
    s = hr.summary_statistics(pd.Series([50.0] * 30), window=20)
    assert s["total_return"] == 0 and s["annualized_volatility"] == 0 and s["avg_volatility"] == 0
    assert np.isnan(s["sharpe_ratio"])
    assert s["positive_days"] == 0
    assert np.isnan(s["avg_up_day"]) and np.isnan(s["avg_down_day"])
    table = hr.format_summary_table(s).set_index("Metric")["Value"]
    assert table["Sharpe Ratio"] == "N/A" and table["Average Up Day"] == "N/A"
    assert table["Total Return"] == "+0.00%" and table["Positive Days"] == "0.0%"


def test_summary_table_rows_and_formatting():
    s = hr.summary_statistics(pd.Series([100.0, 110.0, 99.0, 105.0]), window=2)
    table = hr.format_summary_table(s)
    assert list(table.columns) == ["Metric", "Value"]
    assert table["Metric"].tolist() == ["Total Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
                                        "Maximum Drawdown", "Best Day", "Worst Day", "Positive Days",
                                        "Average Up Day", "Average Down Day"]
    values = table.set_index("Metric")["Value"]
    assert values["Total Return"] == "+5.00%" and values["Maximum Drawdown"] == "-10.00%"


def test_glue_values_are_json_safe():
    assert hr.to_glue_value(np.float64(0.5)) == 0.5 and hr.to_glue_value(np.nan) is None
    assert hr.to_glue_value(np.int64(3)) == 3 and hr.to_glue_value("AAPL") == "AAPL"


# --------------------------------------------------------- data and inputs

def test_validate_parameters():
    assert hr.validate_parameters(" aapl ", "2024-01-01", "2024-12-31") == ("AAPL", "2024-01-01", "2024-12-31")
    for bad in [("", "2024-01-01", "2024-02-01"), ("AAPL", "2024-03-01", "2024-02-01"),
                ("AAPL", "not a date", "2024-02-01")]:
        with pytest.raises(ValueError):
            hr.validate_parameters(*bad)
    with pytest.raises(ValueError):
        hr.validate_parameters("AAPL", "2024-01-01", "2024-02-01", volatility_window=1)


def test_load_filters_inclusive_dates_and_one_ticker(tmp_path):
    dates = days(40)
    db = make_db(tmp_path / "t.db", {"AAA": (dates, np.arange(40) + 10.0), "BBB": (dates, np.arange(40) + 50.0)})
    frame = hr.load_report_data(db, "AAA", dates[5], dates[30])
    assert frame.index.min() == pd.Timestamp(dates[5]) and frame.index.max() == pd.Timestamp(dates[30])
    assert len(frame) == 26 and frame["adj_close"].iloc[0] == 15.0
    assert frame.index.is_monotonic_increasing and isinstance(frame.index, pd.DatetimeIndex)


def test_load_reports_insufficient_history_clearly(tmp_path):
    db = make_db(tmp_path / "t.db", {"AAA": (days(25), np.arange(25) + 10.0)})
    with pytest.raises(ValueError, match=r"insufficient history.*20 prices, need at least 21"):
        hr.load_report_data(db, "AAA", days(25)[5], days(25)[24])
    assert len(hr.load_report_data(db, "AAA", days(25)[4], days(25)[24])) == 21


@pytest.mark.parametrize("ticker, start, end, error, pattern", [
    ("ZZZ", "2024-01-01", "2024-12-31", LookupError, "not in ohlc_hw_cleaned; available: AAA"),
    ("AAA", "2030-01-01", "2030-12-31", ValueError, "0 prices.*covers 2024-01-02 to"),
])
def test_load_errors_name_the_cause(tmp_path, ticker, start, end, error, pattern):
    db = make_db(tmp_path / "t.db", {"AAA": (days(30), np.arange(30) + 10.0)})
    with pytest.raises(error, match=pattern):
        hr.load_report_data(db, ticker, start, end)


def test_load_missing_database_or_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        hr.load_report_data(tmp_path / "absent.db", "AAA", "2024-01-01", "2024-12-31")
    sqlite3.connect(tmp_path / "empty.db").close()
    with pytest.raises(LookupError, match="run 01_data_cleaning"):
        hr.load_report_data(tmp_path / "empty.db", "AAA", "2024-01-01", "2024-12-31")


def test_check_report_data_catches_missing_and_invalid_rows():
    idx = pd.to_datetime(days(4))
    frame = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "adj_close": 5.0,
                          "volume": 100}, index=idx)
    assert hr.check_report_data(frame).sum() == 0
    frame.loc[idx[1], "adj_close"] = np.nan
    frame.loc[idx[2], "close"] = 12.0
    frame.loc[idx[3], "volume"] = -1
    checks = hr.check_report_data(frame)
    assert checks["missing or non-finite value"] == 1 and checks["OHLC bounds violated"] == 1
    assert checks["negative volume"] == 1


# ---------------------------------------------------------------- plotting

def test_volatility_colors_threshold_and_warm_up():
    vol = pd.Series([np.nan, 0.1, 0.2, 0.3])
    colors = hr.volatility_colors(vol, threshold=0.2)
    assert colors.tolist() == [hr.COLORS["warmup"], hr.COLORS["low"], hr.COLORS["low"], hr.COLORS["high"]]


def report_frame(n, constant=False):
    rng = np.random.default_rng(3)
    closes = np.full(n, 50.0) if constant else 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    return pd.DataFrame({"adj_close": closes}, index=pd.to_datetime(days(n)))


def test_plot_layout_titles_and_size():
    fig = hr.plot_risk_report(report_frame(120), "AAA", "2024-01-01", "2024-06-30")
    axes = fig.axes
    assert tuple(fig.get_size_inches()) == (16, 12) and len(axes) == 4
    assert fig._suptitle.get_text() == "AAA Risk Analysis Report - 2024-01-01 to 2024-06-30"
    assert [ax.get_title() for ax in axes] == ["AAA Price and 50-day MA", "AAA Daily Returns",
                                               "AAA Rolling Volatility (20-day)", "AAA Maximum Drawdown"]
    assert all(ax.get_xlabel() and ax.get_ylabel() for ax in axes)
    assert all(ax.xaxis.get_gridlines()[0].get_visible() for ax in axes)
    assert len({ax.get_xlim() for ax in axes}) == 1          # warm-up gap visible on the volatility panel
    matplotlib.pyplot.close(fig)


def test_plot_explains_missing_moving_average_and_handles_flat_prices():
    fig = hr.plot_risk_report(report_frame(30), "AAA", "2024-01-01", "2024-02-15")
    assert any("50-day MA unavailable: only 30 prices" in t.get_text() for t in fig.axes[0].texts)
    matplotlib.pyplot.close(fig)
    fig = hr.plot_risk_report(report_frame(30, constant=True), "FLAT", "2024-01-01", "2024-02-15")
    assert fig.axes[3].get_title() == "FLAT Maximum Drawdown"
    matplotlib.pyplot.close(fig)
