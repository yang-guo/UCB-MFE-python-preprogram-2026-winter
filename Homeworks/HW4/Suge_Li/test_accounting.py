"""Hand-checkable accounting and timing tests. No market database required."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import Backtester, MonthlyTopNAllocator, OrderBook, performance
from features import add_peer_features


def test_inclusive_fee_units_and_cash():
    book = OrderBook(initial_cash=1_000_000, fee_bps=5)
    price = 50.0
    budget = 100_000.0
    units = book.size_for_budget(budget, price)
    expected_units = budget / (price * 1.0005)
    assert np.isclose(units, expected_units)
    book.buy(units, price, symbol="AAA", date="2020-04-02", signal_date="2020-04-01", holding_days=90)
    assert np.isclose(book.cash, 1_000_000 - budget)
    assert np.isclose(book.total_fees, units * price * 0.0005)
    nav = book.nav({"AAA": price})
    # NAV drops by the entry fee immediately; units still mark at the fill price.
    assert np.isclose(nav, 1_000_000 - book.total_fees)


def test_sale_returns_cash_minus_fee_and_does_not_reuse_cash_twice():
    book = OrderBook(initial_cash=10_000, fee_bps=10)
    units = book.size_for_budget(10_000, 100.0)
    book.buy(units, 100.0, symbol="AAA", date="2020-01-03", signal_date="2020-01-02", holding_days=90)
    assert book.cash == 0.0
    book.sell(units, 110.0, lot_id=0, date="2020-04-02")
    expected_cash = units * 110.0 * (1 - 0.001)
    assert np.isclose(book.cash, expected_cash)
    try:
        book.buy(book.size_for_budget(expected_cash + 1, 110.0), 110.0,
                 symbol="BBB", date="2020-04-03", signal_date="2020-04-02")
        raise AssertionError("Overspending should fail")
    except ValueError:
        pass


def test_two_lots_same_ticker_independent_maturities():
    book = OrderBook(initial_cash=20_000, fee_bps=0)
    book.buy(10, 100.0, symbol="AAA", date="2020-01-03", signal_date="2020-01-02", holding_days=90)
    book.buy(10, 120.0, symbol="AAA", date="2020-02-03", signal_date="2020-02-02", holding_days=90)
    assert len(book.lots) == 2
    assert book.lots[0].due_date == pd.Timestamp("2020-01-03") + pd.Timedelta(days=90)
    assert book.lots[1].due_date == pd.Timestamp("2020-02-03") + pd.Timedelta(days=90)
    book.sell(10, 130.0, lot_id=0, date="2020-04-02")
    assert 0 not in book.lots and 1 in book.lots
    assert np.isclose(book.lots[1].units, 10)


def test_signal_must_precede_fill():
    book = OrderBook()
    try:
        book.buy(1, 10.0, symbol="AAA", date="2020-01-02", signal_date="2020-01-02")
        raise AssertionError("Same-day signal and fill should fail")
    except ValueError:
        pass


def test_end_to_end_tiny_calendar():
    """One buy on the first session of April, sell 90 calendar days later."""
    calendar = pd.bdate_range("2020-03-31", "2020-07-15")
    prices = pd.DataFrame({
        "date": np.repeat(calendar, 1),
        "symbol": ["AAA"] * len(calendar),
        "adj_close": np.linspace(100.0, 110.0, len(calendar)),
    })
    bench = pd.DataFrame({"date": calendar, "adj_close": np.linspace(200.0, 220.0, len(calendar))})
    signal = calendar[calendar < "2020-04-01"][-1]
    scores = pd.DataFrame({"date": [signal], "symbol": ["AAA"], "score": [1.0]})
    allocator = MonthlyTopNAllocator(scores, top_n=1, cohort_fraction=1.0, holding_days=90)
    bt = Backtester(prices, bench, allocator, initial_cash=1_000_000, fee_bps=5)
    result = bt.run("2020-04-01", "2020-07-15")
    fills = result.fills
    buys = fills.loc[fills.side.eq("BUY")]
    sells = fills.loc[fills.side.eq("SELL")]
    assert len(buys) == 1
    assert buys.iloc[0].date == pd.Timestamp("2020-04-01")
    assert buys.iloc[0].signal_date < buys.iloc[0].date
    due = pd.Timestamp("2020-04-01") + pd.Timedelta(days=90)
    assert sells.iloc[0].date >= due
    # After the sale, cash equals NAV and no lots remain.
    sale_date = sells.iloc[0].date
    row = result.history.loc[sale_date]
    assert np.isclose(row.cash, row.nav)
    assert int(row.open_lots) == 0
    # Inclusive budget: first-day cash plus holdings*price*(1+fee) reconcilable via fees.
    first = result.history.loc["2020-04-01"]
    assert first.nav < 1_000_000
    assert np.isclose(first.cash + first.holdings_value, first.nav)
    # Benchmark is 1e6 at the first test close.
    assert np.isclose(result.history.loc["2020-04-01", "benchmark_nav"], 1_000_000)


def test_open_position_marked_at_end_without_later_prices():
    calendar = pd.bdate_range("2016-03-31", "2016-04-08")
    prices = pd.DataFrame({
        "date": list(calendar) * 1,
        "symbol": ["AAA"] * len(calendar),
        "adj_close": 10.0,
    })
    bench = pd.DataFrame({"date": calendar, "adj_close": 100.0})
    scores = pd.DataFrame({"date": [pd.Timestamp("2016-03-31")], "symbol": ["AAA"], "score": [0.2]})
    allocator = MonthlyTopNAllocator(scores, top_n=1, cohort_fraction=0.5, holding_days=90)
    result = Backtester(prices, bench, allocator).run("2016-04-01", "2016-04-08")
    assert len(result.open_lots) == 1
    last = result.history.iloc[-1]
    assert last.holdings_value > 0
    assert np.isclose(last.nav, last.cash + last.holdings_value)


def test_performance_formula_on_known_path():
    idx = pd.bdate_range("2020-01-02", periods=253)
    nav = pd.Series(1_000_000 * (1.0002 ** np.arange(len(idx))), index=idx)
    stats = performance(nav, annual_rf=0.02)
    assert stats.final_value > 1_000_000
    assert stats.max_drawdown == 0
    assert stats.CAGR > 0


def test_stale_quotes_are_marked_not_sold():
    calendar = pd.bdate_range("2020-03-31", "2020-04-20")
    rows = []
    for day in calendar:
        rows.append({"date": day, "symbol": "AAA", "adj_close": 10.0})
        if day <= pd.Timestamp("2020-04-03"):
            rows.append({"date": day, "symbol": "BBB", "adj_close": 20.0})
    prices = pd.DataFrame(rows)
    bench = pd.DataFrame({"date": calendar, "adj_close": 100.0})
    scores = pd.DataFrame({
        "date": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-03-31")],
        "symbol": ["AAA", "BBB"],
        "score": [1.0, 0.9],
    })
    allocator = MonthlyTopNAllocator(scores, top_n=2, cohort_fraction=1.0, holding_days=90)
    result = Backtester(prices, bench, allocator, max_stale_sessions=5).run("2020-04-01", "2020-04-20")
    gaps = result.diagnostics.loc[result.diagnostics.event.eq("unresolved_held_gap")]
    assert gaps.symbol.eq("BBB").any()
    sells = result.fills.loc[result.fills.side.eq("SELL") & result.fills.symbol.eq("BBB")]
    assert sells.empty
    last = result.history.iloc[-1]
    assert int(last.open_lots) == 2
    assert last.holdings_value > 0


def test_orders_are_sized_before_todays_close_and_sales():
    """July 1 sale cash and a same-day price jump cannot increase the July 1 buy."""
    calendar = pd.bdate_range("2020-03-31", "2020-07-02")
    prices = pd.DataFrame({"date": calendar, "symbol": "AAA", "adj_close": 10.0})
    prices.loc[prices.date.eq("2020-07-01"), "adj_close"] = 20.0
    bench = pd.DataFrame({"date": calendar, "adj_close": 100.0})
    march = calendar[calendar.to_period("M") == "2020-03"][-1]
    june = calendar[calendar.to_period("M") == "2020-06"][-1]
    scores = pd.DataFrame({"date": [march, june], "symbol": ["AAA", "AAA"], "score": [1.0, 1.0]})
    allocator = MonthlyTopNAllocator(scores, top_n=1, cohort_fraction=1.0, holding_days=91)
    result = Backtester(prices, bench, allocator, fee_bps=0).run("2020-04-01", "2020-07-02")
    july = result.fills.loc[result.fills.date.eq("2020-07-01")]
    assert set(july.side) == {"SELL"}
    assert np.isclose(result.history.loc["2020-07-01", "cash"], 2_000_000)


def test_budget_uses_previous_close_nav():
    calendar = pd.bdate_range("2020-03-31", "2020-05-04")
    prices = pd.DataFrame({"date": calendar, "symbol": "AAA", "adj_close": 100.0})
    prices.loc[prices.date.eq("2020-05-01"), "adj_close"] = 200.0
    bench = pd.DataFrame({"date": calendar, "adj_close": 100.0})
    march = calendar[calendar.to_period("M") == "2020-03"][-1]
    april = calendar[calendar.to_period("M") == "2020-04"][-1]
    scores = pd.DataFrame({"date": [march, april], "symbol": ["AAA", "AAA"], "score": [1.0, 1.0]})
    allocator = MonthlyTopNAllocator(scores, top_n=1, cohort_fraction=0.25, holding_days=90)
    result = Backtester(prices, bench, allocator, fee_bps=0).run("2020-04-01", "2020-05-04")
    may = result.fills.loc[result.fills.date.eq("2020-05-01") & result.fills.side.eq("BUY")]
    assert len(may) == 1
    assert np.isclose(may.iloc[0].notional, 250_000)


def test_second_lot_same_name_allowed():
    calendar = pd.bdate_range("2020-01-31", "2020-03-05")
    prices = pd.DataFrame({"date": calendar, "symbol": "AAA", "adj_close": 10.0})
    bench = pd.DataFrame({"date": calendar, "adj_close": 100.0})
    jan = calendar[calendar.to_period("M") == "2020-01"][-1]
    feb = calendar[calendar.to_period("M") == "2020-02"][-1]
    scores = pd.DataFrame({"date": [jan, feb], "symbol": ["AAA", "AAA"], "score": [1.0, 1.0]})
    allocator = MonthlyTopNAllocator(scores, top_n=1, cohort_fraction=1.0 / 3.0, holding_days=90)
    result = Backtester(prices, bench, allocator).run("2020-02-03", "2020-03-05")
    buys = result.fills.loc[result.fills.side.eq("BUY")]
    assert len(buys) >= 2
    assert buys.lot_id.nunique() >= 2


def test_peer_features_ignore_other_dates():
    base = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-31"] * 3 + ["2014-02-28"] * 3),
        "symbol": ["A", "B", "C"] * 2,
        "sector": ["Tech"] * 6,
        "eligible": True,
        "momentum_63": [0.10, 0.00, -0.10, 0.10, 0.00, -0.10],
        "operating_roa": [0.20, 0.10, 0.00, 0.90, 0.10, 0.00],
    })
    peers = add_peer_features(base)
    day1 = peers.loc[peers.date.eq("2014-01-31")].set_index("symbol")
    mutated = base.copy()
    mutated.loc[mutated.date.eq("2014-02-28") & mutated.symbol.eq("A"), "operating_roa"] = -99.0
    peers2 = add_peer_features(mutated)
    day1_b = peers2.loc[peers2.date.eq("2014-01-31")].set_index("symbol")
    pd.testing.assert_series_equal(day1.roa_sector_rank.sort_index(), day1_b.roa_sector_rank.sort_index())
    assert day1.loc["A", "roa_sector_rank"] > day1.loc["C", "roa_sector_rank"]


if __name__ == "__main__":
    tests = [
        test_inclusive_fee_units_and_cash,
        test_sale_returns_cash_minus_fee_and_does_not_reuse_cash_twice,
        test_two_lots_same_ticker_independent_maturities,
        test_signal_must_precede_fill,
        test_end_to_end_tiny_calendar,
        test_open_position_marked_at_end_without_later_prices,
        test_performance_formula_on_known_path,
        test_stale_quotes_are_marked_not_sold,
        test_orders_are_sized_before_todays_close_and_sales,
        test_budget_uses_previous_close_nav,
        test_second_lot_same_name_allowed,
        test_peer_features_ignore_other_dates,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} accounting/timing checks passed.")
