"""Tests for 02_stock_report_template.ipynb.

The metric functions are taken from the notebook's ``report-functions`` cell, so
the tests check exactly the code the reports run.
"""

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "02_stock_report_template.ipynb"


def tagged_cells(tag):
    nb = nbformat.read(TEMPLATE, as_version=4)
    return [c for c in nb.cells if tag in c.get("metadata", {}).get("tags", [])]


@pytest.fixture(scope="module")
def fn():
    (cell,) = tagged_cells("report-functions")
    namespace = {"np": np, "pd": pd}
    exec(cell.source, namespace)
    return namespace


def test_parameters_cell_is_tagged_and_complete():
    (cell,) = tagged_cells("parameters")
    namespace = {}
    exec(cell.source, namespace)
    assert namespace["ticker"] == "AAPL"
    assert {"start_date", "end_date", "db_path"} <= namespace.keys()


def test_log_returns(fn):
    r = fn["calculate_returns"](pd.Series([100.0, 110.0, 99.0]))
    assert np.isnan(r.iloc[0])
    assert np.allclose(r.iloc[1:], [np.log(1.1), np.log(0.9)])


def test_rolling_volatility_warmup_and_scale(fn):
    returns = fn["calculate_returns"](pd.Series(100 * np.exp(np.cumsum(np.r_[0, np.tile([0.01, -0.01], 15)]))))
    vol = fn["calculate_volatility"](returns, 20)
    assert vol.iloc[:20].isna().all() and vol.iloc[20:].notna().all()   # 21 prices -> first estimate
    expected = returns.iloc[1:21].std(ddof=1) * np.sqrt(252)
    assert np.isclose(vol.iloc[20], expected)


def test_drawdown_recovers_and_max_is_min(fn):
    dd = fn["calculate_max_drawdown"](pd.Series([80.0, 100.0, 70.0, 85.0]))
    assert np.allclose(dd, [0.0, 0.0, -0.30, -0.15])
    assert np.isclose(dd.min(), -0.30)
    assert np.allclose(dd.cummin(), [0.0, 0.0, -0.30, -0.30])


def test_sharpe_formula_and_undefined_cases(fn):
    r = pd.Series([np.nan, 0.01, -0.005, 0.002, 0.007])
    clean = r.dropna()
    expected = (clean.mean() * 252 - 0.02) / (clean.std(ddof=1) * np.sqrt(252))
    assert np.isclose(fn["calculate_sharpe_ratio"](r), expected)
    assert np.isnan(fn["calculate_sharpe_ratio"](pd.Series([0.003] * 10)))   # zero volatility
    assert np.isnan(fn["calculate_sharpe_ratio"](pd.Series([np.nan, 0.01])))  # one return


def test_summary_metrics_exact_formulas(fn):
    prices = pd.Series([100.0, 102.0, 102.0, 99.0, 104.0])
    returns = fn["calculate_returns"](prices)
    dd = fn["calculate_max_drawdown"](prices)
    s = fn["summary_metrics"](prices, returns, dd)
    r = returns.dropna()
    assert len(s) == 10
    assert np.isclose(s["Total Return"], 104 / 100 - 1)
    assert np.isclose(s["Annualized Return"], np.exp(r.sum() * 252 / 4) - 1)
    assert np.isclose(s["Annualized Volatility"], r.std(ddof=1) * np.sqrt(252))
    assert np.isclose(s["Maximum Drawdown"], 99 / 102 - 1)
    assert np.isclose(s["Best Day"], np.log(104 / 99)) and np.isclose(s["Worst Day"], np.log(99 / 102))
    assert np.isclose(s["Positive Days"], 2 / 4)          # zero-return day stays in the denominator
    assert np.isclose(s["Average Up Day"], r[r > 0].mean())
    assert np.isclose(s["Average Down Day"], np.log(99 / 102))


def test_summary_metrics_empty_subsets_are_nan(fn):
    prices = pd.Series([100.0, 101.0, 102.0])
    returns = fn["calculate_returns"](prices)
    s = fn["summary_metrics"](prices, returns, fn["calculate_max_drawdown"](prices))
    assert np.isnan(s["Average Down Day"]) and s["Maximum Drawdown"] == 0
