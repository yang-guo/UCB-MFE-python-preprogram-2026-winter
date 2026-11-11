"""Hand-checkable tests of the backtester's accounting and timing.

Run from the repository root:
    uv run --with pytest pytest Homeworks/HW4/Oscar_Sallusti/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtester import (Backtester, benchmark_nav, check_accounting,  # noqa: E402
                        performance_metrics)

FEE = 0.001  # 10 bps per side


def frame(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates), dtype=float)


def once(schedule):
    """Strategy that places pre-set orders: {signal_date: {symbol: budget}}."""
    schedule = {pd.Timestamp(k): v for k, v in schedule.items()}
    return lambda view: schedule.get(view.date, {})


# Thu, Fri, (weekend), Mon, Tue, Wed
DAYS = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]


def test_round_trip_matches_hand_calculation():
    prices = frame(DAYS, AAA=[10, 10, 11, 12, 12])
    bt = Backtester(prices, initial_cash=1000, fee_bps=10, holding_days=3)
    r = bt.run(once({"2020-01-02": {"AAA": 1000}}), DAYS[0], DAYS[-1])

    buy, sell = r.fills.iloc[0], r.fills.iloc[1]
    # Decided Thu close, filled Fri close at 10. Fee comes out of the budget.
    assert buy.date == pd.Timestamp("2020-01-03") and buy.price == 10
    assert buy.units == pytest.approx(1000 / (10 * 1.001))          # 99.9001 units
    assert buy.fee == pytest.approx(999.000999 * FEE)
    assert r.history.cash["2020-01-03"] == pytest.approx(0, abs=1e-9)
    # Due Fri + 3 days = Mon; sold at Mon close of 11.
    assert sell.date == pd.Timestamp("2020-01-06") and sell.price == 11
    expected_cash = 1000 * (11 / 10) * (1 - FEE) / (1 + FEE)        # 1097.8022
    assert r.history.cash.iloc[-1] == pytest.approx(expected_cash)
    assert sell.realized_pnl == pytest.approx(expected_cash - 1000)
    # Day-by-day NAV: 1000 before entry, 999.0010 after the entry fee, then cash.
    assert r.history.nav.tolist() == pytest.approx(
        [1000, 999.000999, expected_cash, expected_cash, expected_cash])


def test_weekend_maturity_sells_next_session():
    prices = frame(DAYS, AAA=[10, 10, 11, 12, 12])
    bt = Backtester(prices, initial_cash=1000, fee_bps=0, holding_days=1)
    r = bt.run(once({"2020-01-02": {"AAA": 1000}}), DAYS[0], DAYS[-1])
    sell = r.fills[r.fills.side == "SELL"].iloc[0]
    assert sell.due_date == pd.Timestamp("2020-01-04")   # Saturday
    assert sell.date == pd.Timestamp("2020-01-06")       # Monday close
    assert r.history.cash.iloc[-1] == pytest.approx(1100)


def test_ninety_calendar_days_on_real_calendar():
    cal = pd.bdate_range("2016-03-31", "2016-07-15")
    prices = pd.DataFrame({"AAA": np.linspace(100, 120, len(cal))}, index=cal)
    bt = Backtester(prices, initial_cash=1_000_000, fee_bps=0)
    r = bt.run(once({"2016-03-31": {"AAA": 1_000_000}}), cal[0], cal[-1])
    buy, sell = r.fills.iloc[0], r.fills.iloc[1]
    assert buy.date == pd.Timestamp("2016-04-01")
    assert sell.date == pd.Timestamp("2016-06-30")       # Apr 1 + 90 days
    assert sell.price == prices.AAA["2016-06-30"]


def test_missing_fill_quote_cancels_order_and_keeps_cash():
    prices = frame(DAYS, AAA=[10, np.nan, 11, 12, 12])
    bt = Backtester(prices, initial_cash=1000, fee_bps=10, holding_days=3)
    r = bt.run(once({"2020-01-02": {"AAA": 600}}), DAYS[0], DAYS[-1])
    assert r.fills.empty
    assert r.cancelled.symbol.tolist() == ["AAA"]
    assert (r.history.cash == 1000).all()


def test_orders_cannot_exceed_cash():
    prices = frame(DAYS, AAA=[10] * 5, BBB=[20] * 5)
    bt = Backtester(prices, initial_cash=1000)
    with pytest.raises(ValueError, match="> cash"):
        bt.run(once({"2020-01-02": {"AAA": 600, "BBB": 600}}), DAYS[0], DAYS[-1])


def test_cash_is_not_spent_twice_and_sale_cash_is_recycled():
    """Day 1 spends all cash on AAA. A strategy asking for more before the
    sale must fail; after the sale it can reuse the proceeds."""
    prices = frame(DAYS, AAA=[10, 10, 10, 10, 10], BBB=[5] * 5)
    seen = []

    def strategy(view):
        seen.append((view.date, view.cash))
        if view.date == pd.Timestamp("2020-01-02"):
            return {"AAA": 1000}
        if view.date == pd.Timestamp("2020-01-06"):       # AAA sold this close
            return {"BBB": view.cash}
        return {}

    bt = Backtester(prices, initial_cash=1000, fee_bps=0, holding_days=3)
    r = bt.run(strategy, DAYS[0], DAYS[-1])
    cash_seen = dict(seen)
    assert cash_seen[pd.Timestamp("2020-01-03")] == 0      # nothing left to promise
    assert cash_seen[pd.Timestamp("2020-01-06")] == pytest.approx(1000)
    assert r.fills.side.tolist() == ["BUY", "SELL", "BUY"]
    assert r.fills.iloc[2].date == pd.Timestamp("2020-01-07")
    # Strategy is never asked on the final session (its fill would be after the end).
    assert pd.Timestamp(DAYS[-1]) not in cash_seen


def test_price_gap_at_maturity_and_permanent_disappearance():
    days = pd.bdate_range("2020-01-01", periods=15)
    # AAA misses its maturity session and comes back; BBB stops trading forever.
    aaa = [10.0] * 15
    aaa[3] = aaa[4] = np.nan          # no quote Mon 6th (maturity) or Tue 7th
    bbb = [20.0] * 3 + [np.nan] * 12
    prices = pd.DataFrame({"AAA": aaa, "BBB": bbb}, index=days)
    bt = Backtester(prices, initial_cash=1000, fee_bps=0, holding_days=3,
                    stale_flag_sessions=5)
    r = bt.run(once({days[0]: {"AAA": 500, "BBB": 500}}), days[0], days[-1])

    aaa_sell = r.fills[(r.fills.symbol == "AAA") & (r.fills.side == "SELL")].iloc[0]
    assert aaa_sell.due_date == pd.Timestamp("2020-01-05")   # Sunday; next session has no quote
    assert aaa_sell.date == days[5]                          # Wed 8th, first quote afterwards
    # BBB never trades again: it stays open at its last mark (no invented sale).
    assert r.open_lots.symbol.tolist() == ["BBB"]
    assert r.open_lots.mark.iloc[0] == 20
    last = r.history.iloc[-1]
    assert last.nav == pytest.approx(1000)
    assert last.stale_value == pytest.approx(500)
    assert last.nav_ex_stale == pytest.approx(500)
    # Marked at its last close, BBB is not stale until more than 5 sessions pass.
    assert r.history.stale_lots.iloc[:8].eq(0).all()


def test_open_positions_are_valued_on_the_last_session():
    prices = frame(DAYS, AAA=[10, 10, 11, 12, 15])
    bt = Backtester(prices, initial_cash=1000, fee_bps=0, holding_days=90)
    r = bt.run(once({"2020-01-02": {"AAA": 400}}), DAYS[0], DAYS[-1])
    assert r.history.nav.iloc[-1] == pytest.approx(600 + 40 * 15)
    assert r.open_lots.value.iloc[0] == pytest.approx(600)


def test_independent_accounting_check_on_random_run():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2018-01-01", "2019-06-30")
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.02, (len(days), 8)), axis=0)) * 50,
                          index=days, columns=[f"S{i}" for i in range(8)])
    prices = prices.mask(rng.random(prices.shape) < 0.03)          # random missing quotes

    def strategy(view):
        if view.date.dayofweek != 0 or view.cash < 1:
            return {}
        picks = rng.choice(prices.columns, 3, replace=False)
        return {s: view.cash * 0.3 for s in picks}

    bt = Backtester(prices, initial_cash=1_000_000, fee_bps=10)
    r = bt.run(strategy, days[0], days[-1])
    checks = check_accounting(r, prices)
    assert len(r.fills) > 50 and len(r.cancelled) > 0
    assert checks.max_cash_error < 1e-6
    assert checks.max_holdings_error < 1e-6
    assert checks.nav_identity_error < 1e-6
    assert checks.min_cash >= 0
    assert checks.bad_buy_timing == 0 and checks.bad_sell_timing == 0
    # Fee total equals 10 bps of all traded notional.
    assert r.fills.fee.sum() == pytest.approx(r.fills.notional.sum() * FEE)


def test_metrics_by_hand():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2021-01-01"])
    nav = pd.Series([100.0, 110.0, 99.0, 121.0], index=idx)
    m = performance_metrics(nav, rf_annual=0.02)
    assert m.final_value == 121
    assert m.cagr == pytest.approx(1.21 ** (365.25 / 366) - 1)
    assert m.max_drawdown == pytest.approx(99 / 110 - 1)              # -10%
    rets = np.array([0.10, -0.10, 121 / 99 - 1]) - (1.02 ** (1 / 252) - 1)
    assert m.sharpe == pytest.approx(np.sqrt(252) * rets.mean() / rets.std(ddof=1))


def test_benchmark_starts_at_capital_on_first_close():
    idx = pd.bdate_range("2016-03-30", periods=5)
    bench = pd.Series([90.0, 100.0, 101.0, 99.0, 110.0], index=idx)
    nav = benchmark_nav(bench, "2016-03-31", idx[-1])
    assert nav.index[0] == pd.Timestamp("2016-03-31")
    assert nav.tolist() == pytest.approx([1e6, 1.01e6, 0.99e6, 1.10e6])
