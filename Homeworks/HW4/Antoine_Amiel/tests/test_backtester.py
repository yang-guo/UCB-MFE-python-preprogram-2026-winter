"""Hand-checkable accounting and timing tests for the HW4 backtester.

Run from the submission folder:  python tests/test_backtester.py
(or ``python -m pytest tests`` if pytest is installed).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw4.backtester import Backtester, Ledger, accounting_checks, performance  # noqa: E402


class FixedOrders:
    """Buy given dollar budgets on given execution dates; record what it was shown."""

    def __init__(self, plan):
        self.plan = {pd.Timestamp(d): pd.Series(o, dtype=float) for d, o in plan.items()}
        self.calls = []

    def orders(self, signal_date, exec_date, cash, nav, holdings):
        self.calls.append((signal_date, exec_date, cash, nav))
        return self.plan.get(exec_date, pd.Series(dtype=float))


def business_days(start, n):
    return pd.bdate_range(start, periods=n)


def test_single_lot_by_hand():
    days = business_days("2020-01-01", 80)          # 2020-01-01 .. 2020-04-21
    quotes = pd.DataFrame({"AAA": 50.0}, index=days)
    exit_day = pd.Timestamp("2020-04-01")           # 2020-01-02 + 90 calendar days
    quotes.loc[quotes.index >= exit_day, "AAA"] = 60.0
    strat = FixedOrders({"2020-01-02": {"AAA": 100_000.0}})
    bt = Backtester(quotes, strat, initial_cash=1_000_000, fee_bps=5)
    res = bt.run("2020-01-02", days[-1])

    units = 100_000 / (50 * 1.0005)                 # fee is inside the budget
    entry_fee = units * 50 * 0.0005
    exit_fee = units * 60 * 0.0005
    buy, sell = res.fills.iloc[0], res.fills.iloc[1]
    assert np.isclose(buy.units, units) and np.isclose(buy.fee, entry_fee)
    assert np.isclose(units * 50 + entry_fee, 100_000)
    assert sell.date == exit_day and (sell.date - buy.date).days == 90
    assert np.isclose(sell.realized_pnl, units * 10 - entry_fee - exit_fee)
    expected_cash = 1_000_000 - 100_000 + units * 60 - exit_fee
    assert np.isclose(res.history.cash.iloc[-1], expected_cash)
    assert np.isclose(res.history.nav.iloc[-1], expected_cash)
    # Day before the exit: NAV = cash + units * 50.
    assert np.isclose(res.history.nav.loc["2020-03-31"], 900_000 + units * 50)


def test_strategy_plans_before_seeing_fill_prices():
    days = business_days("2020-01-01", 10)
    quotes = pd.DataFrame({"AAA": np.arange(10, 20, dtype=float)}, index=days)
    strat = FixedOrders({days[3]: {"AAA": 1_000.0}})
    Backtester(quotes, strat).run(days[1], days[-1])
    for signal_date, exec_date, *_ in strat.calls:
        assert signal_date < exec_date
        assert days.get_loc(exec_date) - days.get_loc(signal_date) == 1


def test_missing_quotes_reject_buys_and_delay_exits():
    days = business_days("2020-01-01", 80)
    quotes = pd.DataFrame({"AAA": 50.0, "BBB": np.nan}, index=days)
    quotes.loc["2020-04-01":"2020-04-03", "AAA"] = np.nan    # no quote at maturity
    strat = FixedOrders({"2020-01-02": {"AAA": 10_000.0, "BBB": 10_000.0}})
    res = Backtester(quotes, strat, fee_bps=0).run("2020-01-02", days[-1])
    assert (res.events.event == "buy_rejected_no_quote").sum() == 1
    sell = res.fills[res.fills.side == "SELL"].iloc[0]
    assert sell.date == pd.Timestamp("2020-04-06")           # first quoted session after maturity
    assert np.isclose(res.history.nav.iloc[-1], 1_000_000)   # flat price, zero fees
    assert (res.events.event == "exit_delayed_no_quote").sum() == 3


def test_open_lot_valued_at_end_and_marked_through_gaps():
    days = business_days("2020-01-01", 30)
    quotes = pd.DataFrame({"AAA": 100.0}, index=days)
    quotes.iloc[20:, 0] = np.nan                    # quotes disappear
    quotes.iloc[15:20, 0] = 120.0
    strat = FixedOrders({days[1]: {"AAA": 500_000.0}})
    res = Backtester(quotes, strat, fee_bps=0).run(days[1], days[-1])
    assert len(res.open_lots) == 1
    assert np.isclose(res.history.nav.iloc[-1], 500_000 + 5_000 * 120)   # last observed mark


def test_cannot_spend_cash_twice():
    days = business_days("2020-01-01", 5)
    quotes = pd.DataFrame({"AAA": 10.0, "BBB": 10.0}, index=days)
    strat = FixedOrders({days[1]: {"AAA": 600_000.0, "BBB": 600_000.0}})
    try:
        Backtester(quotes, strat).run(days[1], days[-1])
    except ValueError:
        pass
    else:
        raise AssertionError("orders above signal-date cash must be refused")
    ledger = Ledger(1_000)
    ledger.buy("AAA", 1_000, 10, signal_date=days[0], date=days[1], holding_days=90)
    try:
        ledger.buy("BBB", 1, 10, signal_date=days[0], date=days[1], holding_days=90)
    except ValueError:
        pass
    else:
        raise AssertionError("the ledger must refuse purchases without cash")


def test_accounting_checks_and_metrics_on_a_rolling_book():
    rng = np.random.default_rng(0)
    days = business_days("2019-01-01", 400)
    quotes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 3)), axis=0)),
                          index=days, columns=["AAA", "BBB", "CCC"])
    plan = {days[k]: {s: 50_000.0 for s in ["AAA", "BBB", "CCC"]} for k in range(1, 300, 20)}
    res = Backtester(quotes, FixedOrders(plan)).run(days[1], days[-1])
    checks = accounting_checks(res, quotes, 1_000_000)
    assert checks.max_cash_reconstruction_error < 1e-6
    assert checks.max_nav_identity_error < 1e-6
    assert checks.final_holdings_rebuild_error < 1e-6 and checks.pnl_identity_error < 1e-6
    for key in ["cash_never_negative", "every_buy_after_its_signal", "every_sale_held_at_least_90_days",
                "fills_at_that_day_close", "lots_sold_at_most_once", "sold_plus_open_equals_bought"]:
        assert checks[key], key
    nav = pd.Series([100.0, 110.0, 99.0, 121.0],
                    index=pd.to_datetime(["2020-01-01", "2020-07-01", "2020-12-31", "2021-12-31"]))
    perf = performance(nav)
    assert np.isclose(perf.max_drawdown, 99 / 110 - 1)
    assert np.isclose(perf.CAGR, 1.21 ** (365.25 / 730) - 1)


def test_price_jump_on_a_held_stock_is_logged():
    days = business_days("2020-01-01", 20)
    quotes = pd.DataFrame({"AAA": 10.0}, index=days)
    quotes.iloc[10:, 0] = 40.0                       # +300% in one session
    res = Backtester(quotes, FixedOrders({days[1]: {"AAA": 100_000.0}}), fee_bps=0).run(days[1], days[-1])
    assert len(res.incidents) == 1
    inc = res.incidents.iloc[0]
    assert inc.date == days[10] and np.isclose(inc.move, 3.0) and np.isclose(inc.pnl_impact, 10_000 * 30)


def test_capped_allocation_redistributes_excess():
    from hw4.strategy import _capped_allocation
    alloc = _capped_allocation(100.0, np.array([6.0, 2.0, 1.0, 1.0]), np.array([30.0, 100.0, 100.0, 100.0]))
    assert np.allclose(alloc, [30.0, 35.0, 17.5, 17.5]) and np.isclose(alloc.sum(), 100.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
