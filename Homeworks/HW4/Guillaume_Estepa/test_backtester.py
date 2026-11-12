"""Hand-checkable accounting and timing tests.

Run from this folder:  uv run --with pytest pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from data import CLASS_DB, load_market_data
from features import FEATURES, build_panel, weekly_grid
from metrics import buy_and_hold, max_drawdown, performance_metrics
from strategy import TopNTrancheStrategy, build_schedule

SESSIONS = pd.bdate_range("2021-01-01", "2021-12-31")  # weekdays as "market sessions"


class Scripted:
    """Strategy that submits pre-written orders: {signal_date: {symbol: dollars}}."""

    def __init__(self, orders):
        self.orders = {pd.Timestamp(d): o for d, o in orders.items()}
        self.snapshots = {}

    def plan(self, snapshot):
        self.snapshots[snapshot.date] = snapshot
        return self.orders.get(snapshot.date, {})


def flat_prices(**levels):
    return pd.DataFrame({s: np.full(len(SESSIONS), float(p)) for s, p in levels.items()}, index=SESSIONS)


def run(prices, orders, cost_rate=0.0, start="2021-01-04", end="2021-12-31", cash=1000.0):
    return Backtester(prices, start, end, initial_cash=cash, cost_rate=cost_rate).run(Scripted(orders))


def test_buy_hold_sell_by_hand():
    """$1,000 -> 100 units at $10; sold 90 calendar days later at $12 -> $1,200."""
    prices = flat_prices(A=10.0)
    prices.loc["2021-02-01":, "A"] = 12.0
    result = run(prices, {"2021-01-01": {"A": 1000.0}})
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert (buy.date, buy.units, buy.price) == (pd.Timestamp("2021-01-04"), 100.0, 10.0)
    # 2021-01-04 + 90 days = Sunday 2021-04-04 -> next session, Monday 2021-04-05.
    assert sell.date == pd.Timestamp("2021-04-05") and sell.cash_flow == 1200.0
    daily = result.daily.total_value
    assert daily.loc["2021-01-01"] == 1000.0  # untouched capital before the first fill
    assert daily.loc["2021-01-29"] == 1000.0 and daily.loc["2021-02-01"] == 1200.0
    assert result.daily.cash.loc["2021-04-01"] == 0.0 and result.daily.cash.iloc[-1] == 1200.0
    assert all(v for v in result.audit.values() if isinstance(v, bool))


def test_transaction_costs_on_both_sides():
    prices = flat_prices(A=10.0)
    result = run(prices, {"2021-01-01": {"A": 1000.0}}, cost_rate=0.01)
    units = 1000.0 / (10.0 * 1.01)  # the budget includes the purchase fee
    assert result.trades.units.iloc[0] == pytest.approx(units)
    assert result.daily.total_value.loc["2021-01-04"] == pytest.approx(units * 10.0)  # fee shows on day 1
    assert result.daily.cash.iloc[-1] == pytest.approx(units * 10.0 * 0.99)
    assert result.audit["total_fees"] == pytest.approx(units * 10.0 * 0.02)


def test_sale_proceeds_are_reused_only_from_the_next_session():
    """Cash from a sale at close t can fund a purchase at close t+1, not at t."""
    prices = flat_prices(A=10.0, B=5.0)
    result = run(prices, {"2021-01-01": {"A": 1000.0}, "2021-04-05": {"B": 1000.0}})
    assert result.trades[["date", "side", "symbol"]].values.tolist() == [
        [pd.Timestamp("2021-01-04"), "BUY", "A"], [pd.Timestamp("2021-04-05"), "SELL", "A"],
        [pd.Timestamp("2021-04-06"), "BUY", "B"], [pd.Timestamp("2021-07-05"), "SELL", "B"]]
    assert (result.daily.cash >= 0).all()


def test_cannot_spend_more_than_available_cash():
    with pytest.raises(ValueError, match="exceed available cash"):
        run(flat_prices(A=10.0, B=5.0), {"2021-01-01": {"A": 600.0, "B": 600.0}})
    # ... including cash that is already invested.
    with pytest.raises(ValueError, match="exceed available cash"):
        run(flat_prices(A=10.0, B=5.0), {"2021-01-01": {"A": 1000.0}, "2021-01-05": {"B": 1.0}})


def test_missing_quote_at_entry_rejects_the_order():
    prices = flat_prices(A=10.0)
    prices.loc["2021-01-04", "A"] = np.nan
    result = run(prices, {"2021-01-01": {"A": 1000.0}})
    assert result.trades.empty and len(result.rejected) == 1
    assert (result.daily.cash == 1000.0).all()


def test_missing_quote_at_maturity_delays_the_sale_without_inventing_a_price():
    prices = flat_prices(A=10.0)
    prices.loc["2021-04-01":"2021-04-07", "A"] = np.nan  # no quotes around maturity
    prices.loc["2021-04-08":, "A"] = 8.0
    result = run(prices, {"2021-01-01": {"A": 1000.0}})
    assert result.trades.date.iloc[1] == pd.Timestamp("2021-04-08")
    assert result.trades.cash_flow.iloc[1] == 800.0
    # While unquoted, the lot is carried at the last observed $10 and flagged stale.
    assert result.daily.total_value.loc["2021-04-06"] == 1000.0
    assert result.daily.stale_value.loc["2021-04-06"] == 1000.0
    assert result.audit["n_sales_delayed_by_missing_quote"] == 1


def test_stock_that_stops_trading_stays_open_unless_cash_out_is_assumed():
    prices = flat_prices(A=10.0)
    prices.loc["2021-03-01":, "A"] = np.nan  # e.g. acquired: the price history just ends
    orders = {"2021-01-01": {"A": 1000.0}}
    result = run(prices, orders)
    assert result.trades.side.tolist() == ["BUY"]  # default: no invented sale
    assert result.daily.total_value.iloc[-1] == 1000.0 and result.daily.cash.iloc[-1] == 0.0
    assert result.audit["final_overdue_value"] == 1000.0
    # Sensitivity option: cash out at the last observed price once matured and 10 sessions stale.
    assumed = Backtester(prices, "2021-01-04", "2021-12-31", 1000.0, 0.0,
                         assume_cash_out_after=10).run(Scripted(orders))
    sale = assumed.trades.iloc[1]
    assert sale.date == pd.Timestamp("2021-04-05") and sale.price == 10.0 and sale.assumed_fill
    assert assumed.audit["n_assumed_cash_outs"] == 1


def test_open_positions_are_valued_at_the_end_and_later_prices_are_ignored():
    prices = flat_prices(A=10.0)
    prices.loc["2021-03-01":, "A"] = 11.0
    prices.loc["2021-03-16":, "A"] = 99.0  # after the test end: must not matter
    result = run(prices, {"2021-01-01": {"A": 1000.0}}, end="2021-03-15")
    assert result.trades.side.tolist() == ["BUY"]  # not yet matured -> still open
    assert result.daily.index[-1] == pd.Timestamp("2021-03-15")
    assert result.daily.total_value.iloc[-1] == pytest.approx(1100.0)
    assert len(result.open_lots) == 1


def test_strategy_never_sees_the_fill_price():
    """Changing prices after a signal date cannot change what the strategy saw."""
    prices = flat_prices(A=10.0)
    shocked = prices.copy()
    shocked.loc["2021-01-04":, "A"] = 20.0
    orders = {"2021-01-01": {"A": 1000.0}}
    first, second = Scripted(orders), Scripted(orders)
    Backtester(prices, "2021-01-04", "2021-12-31", 1000.0, 0.0).run(first)
    Backtester(shocked, "2021-01-04", "2021-12-31", 1000.0, 0.0).run(second)
    assert first.snapshots[pd.Timestamp("2021-01-01")] == second.snapshots[pd.Timestamp("2021-01-01")]


def test_metrics_by_hand():
    index = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2022-01-01"])
    values = pd.Series([100.0, 120.0, 90.0, 121.0], index=index)
    metrics = performance_metrics(values)
    assert max_drawdown(values) == pytest.approx(90 / 120 - 1)
    assert metrics.cagr == pytest.approx(1.21 ** (365.25 / 731) - 1)
    returns = values.pct_change().dropna() - (1.02 ** (1 / 252) - 1)
    assert metrics.sharpe == pytest.approx(returns.mean() / returns.std() * np.sqrt(252))


def test_benchmark_is_bought_at_the_first_test_close():
    benchmark = pd.Series(np.linspace(100, 200, len(SESSIONS)), index=SESSIONS)
    values = buy_and_hold(benchmark, "2021-01-04", "2021-12-31", 1000.0)
    assert values.iloc[0] == 1000.0 and values.loc["2021-01-04"] == 1000.0
    assert values.iloc[-1] == pytest.approx(1000.0 * 200 / benchmark.loc["2021-01-04"])


def test_schedule_and_tranche_build_up():
    schedule = build_schedule(SESSIONS, "2021-01-01", "2021-12-31", n_tranches=3, spacing=21)
    dates = list(schedule)
    assert dates[:3] == [SESSIONS[0], SESSIONS[21], SESSIONS[42]]
    assert pd.Timestamp("2021-04-05") in schedule  # tranche 0 re-decides when its lots mature
    predictions = pd.DataFrame({"date": np.repeat(dates, 2), "symbol": ["A", "B"] * len(dates),
                                "prediction": [1.0, 0.5] * len(dates), "sector": "x",
                                "liquidity_rank": [1, 2] * len(dates)})
    strategy = TopNTrancheStrategy(predictions, schedule, n_stocks=1, n_tranches=3, sector_cap=1.0)
    result = Backtester(flat_prices(A=10.0, B=5.0), "2021-01-04", "2021-12-31", 900.0, 0.0).run(strategy)
    first_three = result.trades[result.trades.side == "BUY"].head(3)
    assert first_three.symbol.tolist() == ["A", "A", "A"]  # always the best prediction
    assert first_three.cash_flow.round(6).tolist() == [-300.0, -300.0, -300.0]  # one third each
    assert all(v for v in result.audit.values() if isinstance(v, bool))


@pytest.mark.skipif(not CLASS_DB.is_file(), reason="class database not downloaded")
def test_features_do_not_change_when_future_data_is_added():
    """Look-ahead test: features built with data to 2014 must equal the same
    rows built with data to 2015. Only unfinished labels may differ."""
    early = load_market_data(CLASS_DB, end="2014-12-31")
    late = load_market_data(CLASS_DB, end="2015-12-31")
    panel_early = build_panel(early, weekly_grid(early.sessions))
    panel_late = build_panel(late, weekly_grid(late.sessions))
    keys = ["date", "symbol"]
    merged = panel_early.merge(panel_late, on=keys, suffixes=("_early", "_late"), how="left")
    assert len(merged) == len(panel_early) and merged["sector_late"].notna().all()
    for feature in FEATURES + ["liquidity_rank", "universe_rank"]:
        pd.testing.assert_series_equal(merged[feature + "_early"], merged[feature + "_late"],
                                       check_names=False)
    finished = merged.forward_return_early.notna()
    assert finished.any() and (merged.loc[finished, "label_end_early"] <= pd.Timestamp("2014-12-31")).all()
    np.testing.assert_allclose(merged.loc[finished, "forward_return_early"],
                               merged.loc[finished, "forward_return_late"])
