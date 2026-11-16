"""Hand checks for lagged sector stress, budgets and immutable lot maturity."""

from dataclasses import replace
import numpy as np
import pandas as pd

from hw4.settings import TradingRules
from cap_sensitivity_engine import run_sector_backtest as control_engine
from sector_stress import build_sector_stress, scale_orders, forward_risk
from sector_stress_engine import run_sector_backtest


def test_prior_twelve_months_exclude_current_and_future_readings():
    dates = pd.date_range("2011-01-31", periods=15, freq="ME")
    p = pd.DataFrame(
        [
            {
                "date": d,
                "symbol": s,
                "sector": "Energy",
                "eligible": True,
                "volatility_63": 0.2 if i < 12 else 0.4,
                "target": 0.0,
            }
            for i, d in enumerate(dates)
            for s in ["A", "B"]
        ]
    )
    stress = build_sector_stress(p, min_stocks=2)
    assert stress.stress_ratio.iloc[:12].isna().all()
    np.testing.assert_allclose(stress.stress_ratio.iloc[12], 2.0)
    changed = p.copy()
    changed.loc[changed.date.gt(dates[12]), "volatility_63"] = 100
    changed["target"] = np.nan
    pd.testing.assert_frame_equal(
        stress.iloc[:13], build_sector_stress(changed, min_stocks=2).iloc[:13]
    )
    missing = p.loc[p.date.ne(dates[4]) | p.symbol.eq("A")]
    assert pd.isna(build_sector_stress(missing, min_stocks=2).stress_ratio.iloc[12])


def test_scaling_threshold_unknown_and_cash_not_redistributed():
    budgets = {"A": 100.0, "B": 200.0, "C": 300.0}
    sectors = {"A": "Energy", "B": "Tech", "C": "Other"}
    readings = {"Energy": 1.5, "Tech": 1.4999}
    actual, audit = scale_orders(budgets, sectors, readings, 0.5)
    assert actual == {"A": 50.0, "B": 200.0}
    assert sum(x["withheld_budget"] for x in audit) == 350
    assert scale_orders(budgets, sectors, readings, 0)[0] == {"B": 200.0}
    assert scale_orders(budgets, sectors, readings, 1)[0] == budgets


def test_control_parity_stress_does_not_sell_old_lots_or_use_future_fill():
    days = pd.bdate_range("2013-01-01", "2013-05-01")
    quotes = pd.DataFrame(100.0, index=days, columns=["A", "B"])
    benchmark = pd.Series(100.0, index=days)
    signals = [days[0], days[20]]
    scores = pd.DataFrame(
        [
            {"date": d, "symbol": s, "score": score}
            for d in signals
            for s, score in [("A", 2.0), ("B", 1.0)]
        ]
    )
    rules = replace(
        TradingRules(),
        initial_cash=1000.0,
        name_cap=0.5,
        cohort_fraction=0.5,
        top_n=2,
        fee_bps=10,
    )
    sectors = {"A": "Energy", "B": "Tech"}
    stress = pd.DataFrame(
        [
            {"date": d, "sector": s, "stress_ratio": r}
            for d, r in zip(signals, [1.0, 2.0])
            for s in ["Energy", "Tech"]
        ]
    )
    args = (quotes, benchmark, scores, days[1], days[-1], rules)
    base = control_engine(*args, sectors=sectors, sector_cap=0.5)
    replay = run_sector_backtest(
        *args,
        sectors=sectors,
        sector_cap=0.5,
        sector_stress=stress,
        stress_multiplier=1,
    )
    pd.testing.assert_frame_equal(base.history, replay.history)
    pd.testing.assert_frame_equal(base.fills, replay.fills)
    paused = run_sector_backtest(
        *args,
        sectors=sectors,
        sector_cap=0.5,
        sector_stress=stress,
        stress_multiplier=0,
    )
    buys = paused.fills.query("side=='BUY'")
    assert len(buys) == 2 and buys.date.eq(days[1]).all()
    sales = paused.fills.query("side=='SELL'")
    due = days[days.searchsorted(days[1] + pd.Timedelta(days=90))]
    assert len(sales) == 2 and sales.date.eq(due).all()
    changed = quotes.copy()
    changed.loc[days[21], "A"] = 170.0
    other = run_sector_backtest(
        changed,
        benchmark,
        scores,
        days[1],
        days[-1],
        rules,
        sectors=sectors,
        sector_cap=0.5,
        sector_stress=stress,
        stress_multiplier=0,
    )
    pd.testing.assert_frame_equal(paused.stress_orders, other.stress_orders)


def test_forward_risk_is_diagnostic_and_missing_path_stays_missing():
    dates = pd.bdate_range("2013-01-01", periods=4)
    q = pd.DataFrame(
        {"A": [100.0, 120.0, 90.0, 110.0], "B": [100.0, np.nan, 90.0, 110.0]},
        index=dates,
    )
    rows = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "entry_date": [dates[0]] * 2,
            "label_end": [dates[-1]] * 2,
            "target": [0.1, 0.1],
        }
    )
    r = forward_risk(rows, q, dates[-1])
    np.testing.assert_allclose(r.future_drawdown.iloc[0], -0.25)
    np.testing.assert_allclose(
        r.future_volatility.iloc[0],
        np.std([0.2, -0.25, 110 / 90 - 1], ddof=1) * np.sqrt(252),
    )
    assert r.observed.all() and pd.isna(r.future_volatility.iloc[1])


def test_positive_reduced_order_budget_is_independent_of_future_fill_price():
    days = pd.bdate_range("2013-01-01", periods=4)
    quotes = pd.DataFrame(100.0, index=days, columns=["A"])
    benchmark = pd.Series(100.0, index=days)
    scores = pd.DataFrame({"date": [days[0]], "symbol": ["A"], "score": [1.0]})
    stress = pd.DataFrame(
        {"date": [days[0]], "sector": ["Energy"], "stress_ratio": [2.0]}
    )
    rules = replace(
        TradingRules(),
        initial_cash=1000.0,
        name_cap=0.5,
        top_n=1,
        cohort_fraction=0.5,
        fee_bps=10,
    )
    kwargs = dict(
        sectors={"A": "Energy"},
        sector_cap=0.5,
        sector_stress=stress,
        stress_multiplier=0.5,
    )
    first = run_sector_backtest(
        quotes, benchmark, scores, days[1], days[-1], rules, **kwargs
    )
    changed = quotes.copy()
    changed.loc[days[1], "A"] = 125.0
    second = run_sector_backtest(
        changed, benchmark, scores, days[1], days[-1], rules, **kwargs
    )
    pd.testing.assert_frame_equal(first.stress_orders, second.stress_orders)
    np.testing.assert_allclose(first.fills.cash_change, [-250.0])
    np.testing.assert_allclose(second.fills.cash_change, [-250.0])
    np.testing.assert_allclose(second.fills.units, first.fills.units * 0.8)
