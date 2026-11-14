"""Hand-checked tests for my backtester and the price repair.

Run from this folder with:  python test_backtester.py

Every test uses a tiny made-up price table, so the expected numbers can be
worked out with a calculator. The comments show that arithmetic.
"""

import unittest

import numpy as np
import pandas as pd

from backtester import Backtester, benchmark_nav, performance
from research import clean_prices

DAYS = pd.bdate_range("2020-01-01", "2020-06-30")  # Mon-Fri sessions


def flat_prices(**symbols):
    """Each stock gets a constant price; tests then edit single days."""
    return pd.DataFrame({s: float(p) for s, p in symbols.items()}, index=DAYS)


def buy_once(on_signal_day, budgets):
    """A strategy that places one set of orders at one signal close."""
    def strategy(account):
        if account.date == pd.Timestamp(on_signal_day):
            return pd.Series(budgets, dtype=float)
        return pd.Series(dtype=float)
    return strategy


class TestBacktester(unittest.TestCase):

    def test_one_round_trip_by_hand(self):
        # Signal Fri Jan 3, so the buy fills at the close of Mon Jan 6.
        # Price is 50 on the signal day and 100 on Jan 6: we must pay 100.
        prices = flat_prices(A=100)
        prices.loc["2020-01-03", "A"] = 50
        due = pd.Timestamp("2020-01-06") + pd.Timedelta(days=90)  # Sun Apr 5
        prices.loc["2020-04-06":, "A"] = 110                       # Mon Apr 6 = first session after
        bt = Backtester(prices, cost_bps=10)
        result = bt.run(buy_once("2020-01-03", {"A": 10_000}), "2020-01-06", "2020-06-30")

        buy, sell = result.trades.iloc[0], result.trades.iloc[1]
        # Budget includes the fee: units = 10,000 / (100 x 1.001) = 99.9001
        units = 10_000 / (100 * 1.001)
        self.assertEqual(buy.date, pd.Timestamp("2020-01-06"))
        self.assertEqual(buy.price, 100)
        self.assertAlmostEqual(buy.units, units)
        self.assertAlmostEqual(buy.fee, units * 100 * 0.001)  # 9.99
        self.assertAlmostEqual(result.daily.loc["2020-01-06", "cash"], 990_000)

        # The due date is a Sunday, so the sale is at Monday's close.
        self.assertEqual(due, pd.Timestamp("2020-04-05"))
        self.assertEqual(sell.date, pd.Timestamp("2020-04-06"))
        self.assertEqual(sell.price, 110)
        proceeds = units * 110                                  # 10,989.01
        expected_cash = 990_000 + proceeds * (1 - 0.001)        # 1,000,978.02
        self.assertAlmostEqual(result.daily.loc["2020-04-06", "cash"], expected_cash)
        self.assertAlmostEqual(sell.pnl, expected_cash - 1_000_000)
        self.assertEqual(result.daily.loc["2020-04-06", "open_lots"], 0)

    def test_value_between_trades(self):
        # On Feb 3 the price is 120, so NAV = 990,000 + 99.9001 x 120.
        prices = flat_prices(A=100)
        prices.loc["2020-02-03", "A"] = 120
        result = Backtester(prices).run(buy_once("2020-01-03", {"A": 10_000}), "2020-01-06", "2020-03-31")
        units = 10_000 / (100 * 1.001)
        self.assertAlmostEqual(result.daily.loc["2020-02-03", "nav"], 990_000 + units * 120)
        self.assertAlmostEqual(result.daily.loc["2020-02-04", "nav"], 990_000 + units * 100)

    def test_nav_always_equals_cash_plus_holdings(self):
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .02, (len(DAYS), 3)), axis=0)),
                              index=DAYS, columns=["A", "B", "C"])

        def monthly(account):  # buy all three with a third of NAV each month-end
            if account.date.month != (account.date + pd.offsets.BDay(1)).month:
                budget = min(account.cash, account.nav / 3)
                return pd.Series(budget / 3, index=["A", "B", "C"])
            return pd.Series(dtype=float)

        result = Backtester(prices).run(monthly, "2020-01-02", "2020-06-30")
        d = result.daily
        self.assertTrue(np.allclose(d.nav, d.cash + d.holdings_value))
        self.assertTrue((d.cash >= 0).all())
        # Final NAV = start + realized PnL + open lots' gain - their entry fees
        lots = result.open_lots
        unrealized = (lots.units * lots.last_price - lots.units * lots.entry_price - lots.entry_fee).sum()
        realized = result.trades.loc[result.trades.side == "SELL", "pnl"].sum()
        self.assertAlmostEqual(d.nav.iloc[-1], 1_000_000 + realized + unrealized, places=4)
        # Every sale happens on or after its due date
        sells = result.trades[result.trades.side == "SELL"]
        self.assertTrue((sells.date >= sells.due_date).all())

    def test_cannot_spend_more_than_cash(self):
        prices = flat_prices(A=100, B=100)
        bt = Backtester(prices)
        with self.assertRaises(ValueError):
            bt.run(buy_once("2020-01-03", {"A": 600_000, "B": 600_000}), "2020-01-06", "2020-03-31")
        # Spending exactly all the cash works and leaves zero, never negative.
        result = Backtester(prices).run(buy_once("2020-01-03", {"A": 500_000, "B": 500_000}),
                                        "2020-01-06", "2020-03-31")
        self.assertAlmostEqual(result.daily.cash.iloc[0], 0, places=6)
        self.assertAlmostEqual(result.daily.nav.iloc[0], 1_000_000 / 1.001)  # only fees are lost

    def test_buy_skipped_when_no_quote(self):
        prices = flat_prices(A=100)
        prices.loc["2020-01-06", "A"] = np.nan  # no quote on the fill day
        result = Backtester(prices).run(buy_once("2020-01-03", {"A": 10_000}), "2020-01-06", "2020-03-31")
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.events.event.tolist(), ["buy_skipped_no_quote"])
        self.assertAlmostEqual(result.daily.nav.iloc[-1], 1_000_000)  # no invented fill

    def test_exit_waits_for_next_quote(self):
        prices = flat_prices(A=100)
        prices.loc["2020-04-06":"2020-04-07", "A"] = np.nan  # missing on the due session
        prices.loc["2020-04-08", "A"] = 90
        result = Backtester(prices).run(buy_once("2020-01-03", {"A": 10_000}), "2020-01-06", "2020-06-30")
        sell = result.trades[result.trades.side == "SELL"].iloc[0]
        self.assertEqual(sell.date, pd.Timestamp("2020-04-08"))
        self.assertEqual(sell.price, 90)
        # While waiting, the lot is valued at its last close (100).
        units = 10_000 / (100 * 1.001)
        self.assertAlmostEqual(result.daily.loc["2020-04-07", "nav"], 990_000 + units * 100)

    def test_stock_that_stops_trading(self):
        prices = flat_prices(A=100)
        prices.loc["2020-02-03":, "A"] = np.nan  # last quote Jan 31, never comes back
        last = Backtester(prices, stale_limit=5, dark_policy="last").run(
            buy_once("2020-01-03", {"A": 10_000}), "2020-01-06", "2020-06-30")
        zero = Backtester(prices, stale_limit=5, dark_policy="zero").run(
            buy_once("2020-01-03", {"A": 10_000}), "2020-01-06", "2020-06-30")
        units = 10_000 / (100 * 1.001)
        # Optimistic: still worth its last close. Pessimistic: worth nothing.
        self.assertAlmostEqual(last.nav.iloc[-1], 990_000 + units * 100)
        self.assertAlmostEqual(zero.nav.iloc[-1], 990_000)
        # Up to 5 missing sessions it is still marked at 100 under both rules.
        self.assertAlmostEqual(zero.daily.loc["2020-02-07", "nav"], 990_000 + units * 100)
        self.assertAlmostEqual(zero.daily.loc["2020-02-10", "nav"], 990_000)
        self.assertEqual(len(last.open_lots), 1)  # never sold: there is no price to sell at

    def test_performance_metrics_by_hand(self):
        # Start with 1M cash: 1M -> 1.1M -> 0.99M -> 1.21M over two calendar years.
        dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01", "2022-01-01"])
        nav = pd.Series([1_000_000, 1_100_000, 990_000, 1_210_000], index=dates)
        m = performance(nav)
        years = (dates[-1] - dates[0]).days / 365.25
        self.assertAlmostEqual(m.CAGR, 1.21 ** (1 / years) - 1)
        self.assertAlmostEqual(m.max_drawdown, 990_000 / 1_100_000 - 1)  # -10%
        r = np.array([0.0, 0.1, -0.1, 1_210_000 / 990_000 - 1])  # first return is from starting cash
        rf = 1.02 ** (1 / 252) - 1
        self.assertAlmostEqual(m.Sharpe, (r - rf).mean() / r.std(ddof=1) * np.sqrt(252))

    def test_benchmark_starts_at_one_million(self):
        bench = pd.Series([200.0, 210.0, 220.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
        b = benchmark_nav(bench, "2020-01-03", "2020-01-06")
        self.assertEqual(b.iloc[0], 1_000_000)
        self.assertAlmostEqual(b.iloc[-1], 1_000_000 * 220 / 210)


def repair(adj, close=None, volume=None, market=None):
    """Run research.clean_prices on one made-up stock over 200 sessions."""
    n = len(adj)
    close = adj if close is None else close
    volume = np.full(n, 1000.0) if volume is None else volume
    market = np.zeros(n) if market is None else market
    cal = pd.bdate_range("2020-01-01", periods=n)
    fixed, log = clean_prices(np.array(adj, float)[:, None], np.array(close, float)[:, None],
                              np.array(volume, float)[:, None], pd.Index(["A"]), cal, market)
    return fixed[:, 0], log


class TestPriceRepair(unittest.TestCase):
    N = 200

    def test_zero_volume_quote_is_dropped(self):
        adj = np.full(self.N, 10.0)
        volume = np.full(self.N, 1000.0)
        adj[50], volume[50] = 0.41, 0  # AAM-style quote with nothing traded
        fixed, log = repair(adj, volume=volume)
        self.assertTrue(np.isnan(fixed[50]))
        self.assertEqual(fixed[51], 10.0)
        self.assertEqual(log.repair.tolist(), ["zero_volume_quote"])

    def test_frozen_adjusted_close_is_dropped(self):
        close = 30 + np.arange(self.N) * 0.1  # the raw close keeps moving...
        adj = close / 3
        adj[100:120] = adj[99]                 # ...but adj_close is stuck (CSC)
        fixed, _ = repair(adj, close=close)
        self.assertTrue(np.isnan(fixed[100:120]).all())
        self.assertFalse(np.isnan(fixed[120]))

    def test_one_day_spike_is_dropped(self):
        adj = np.full(self.N, 10.0)
        adj[80] = 50.0  # 5x for one day, then straight back
        fixed, log = repair(adj)
        self.assertTrue(np.isnan(fixed[80]))
        self.assertEqual(log.repair.tolist(), ["one_day_spike"])

    def test_missed_2_for_1_split_is_rescaled(self):
        # Price halves and share volume doubles; dollars traded stay the same.
        adj = np.r_[np.full(100, 100.0), np.full(100, 50.0)]
        volume = np.r_[np.full(100, 1000.0), np.full(100, 2000.0)]
        fixed, log = repair(adj, volume=volume)
        self.assertTrue(np.allclose(fixed, 50.0))  # history divided by 2: no fake -50% day
        self.assertEqual(log.factor.tolist(), [0.5])

    def test_crash_is_not_called_a_split(self):
        # Same -50% move, but the crash day trades 10x the usual dollars.
        adj = np.r_[np.full(100, 100.0), np.full(100, 50.0)]
        volume = np.r_[np.full(100, 1000.0), [20_000.0], np.full(99, 2000.0)]
        fixed, log = repair(adj, volume=volume)
        self.assertEqual(fixed[0], 100.0)
        self.assertEqual(len(log), 0)

    def test_no_split_repair_on_a_market_crash_day(self):
        adj = np.r_[np.full(100, 100.0), np.full(100, 50.0)]
        volume = np.r_[np.full(100, 1000.0), np.full(100, 2000.0)]
        market = np.zeros(self.N)
        market[100] = -0.10  # the S&P 500 fell 10% that day
        fixed, log = repair(adj, volume=volume, market=market)
        self.assertEqual(fixed[0], 100.0)
        self.assertEqual(len(log), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
