"""Hand-checkable tests of the backtester's timing and accounting."""

import unittest

import numpy as np
import pandas as pd

from backtest import FEE_RATE, INITIAL_CAPITAL, TopNStrategy, performance, run_backtest


SIGNAL = pd.Timestamp("2024-01-05")


def flat_quotes(end, symbols=("A", "B"), price=100.0):
    sessions = pd.bdate_range("2024-01-08", end)
    return pd.DataFrame({symbol: price for symbol in symbols}, index=sessions)


class AccountingTests(unittest.TestCase):
    def test_fees_weekend_maturity_and_missing_quotes(self):
        quotes = flat_quotes("2024-04-10")
        sessions = quotes.index
        quotes.loc[sessions[0], "B"] = np.nan
        quotes.loc["2024-04-08":"2024-04-09", "A"] = np.nan
        quotes.loc["2024-04-10", "A"] = 110.0
        result = run_backtest(quotes, TopNStrategy({SIGNAL: ["A", "B"]}), SIGNAL)
        buy, sell = list(result.fills.itertuples(index=False))
        self.assertEqual((buy.side, buy.symbol, buy.date, buy.signal_date), ("buy", "A", sessions[0], SIGNAL))
        self.assertEqual(buy.due, pd.Timestamp("2024-04-07"))  # Sunday
        self.assertEqual((sell.side, sell.date), ("sell", sessions[-1]))
        self.assertEqual(result.events.event.tolist(), ["canceled_buy", "delayed_sale"])

        # A quarter of $1,000,000 is split between two names, and only A has an entry quote.
        budget = 125_000.0
        units = budget / 1.001 / 100
        proceeds = units * 110 * 0.999
        self.assertAlmostEqual(buy.units, units)
        self.assertAlmostEqual(buy.fee, budget - units * 100)
        self.assertAlmostEqual(result.daily.cash.iloc[0], INITIAL_CAPITAL - budget)
        self.assertAlmostEqual(result.daily.nav.iloc[0], INITIAL_CAPITAL - budget + units * 100)
        self.assertAlmostEqual(result.daily.loc["2024-04-09", "stale_value"], units * 100)
        self.assertAlmostEqual(sell.realized_pnl, proceeds - budget)
        self.assertAlmostEqual(result.daily.nav.iloc[-1], INITIAL_CAPITAL - budget + proceeds)
        self.assertEqual(result.open_lots, [])

    def test_pending_buy_cannot_use_sales_cash_from_its_fill_day(self):
        quotes = flat_quotes("2024-04-08")
        sessions = quotes.index
        quotes.loc[sessions[1]:, "A"] = 300.0
        rankings = {SIGNAL: ["A"], sessions[1]: ["A"], sessions[3]: ["A"], sessions[-2]: ["B"]}
        result = run_backtest(quotes, TopNStrategy(rankings), SIGNAL)
        before_sale = result.daily.iloc[-2]
        fill_day = result.fills.loc[result.fills.date.eq(sessions[-1])]
        self.assertEqual(fill_day.side.tolist(), ["sell", "buy"])
        self.assertEqual(fill_day.symbol.tolist(), ["A", "B"])
        self.assertLess(before_sale.cash, before_sale.nav / 4)
        self.assertAlmostEqual(-fill_day.cash_change.iloc[1], before_sale.cash)
        self.assertEqual(len(result.open_lots), 3)

    def test_unresolved_position_stays_open_at_its_last_mark(self):
        quotes = flat_quotes("2024-04-15", symbols=("A",))
        quotes.loc["2024-04-05":, "A"] = np.nan
        result = run_backtest(quotes, TopNStrategy({SIGNAL: ["A"]}), SIGNAL)
        self.assertEqual(result.fills.side.tolist(), ["buy"])
        self.assertEqual(result.events.event.tolist(), ["long_quote_gap"])
        self.assertEqual(len(result.open_lots), 1)
        self.assertAlmostEqual(result.daily.long_stale_value.iloc[-1], 250_000 / (1 + FEE_RATE))
        self.assertAlmostEqual(performance(result.daily.nav).max_drawdown,
                               result.daily.nav.iloc[0] / INITIAL_CAPITAL - 1)

    def test_ledger_reconciles_with_daily_values(self):
        rng = np.random.default_rng(0)
        sessions = pd.bdate_range("2024-01-08", "2024-12-31")
        symbols = list("ABCDEF")
        steps = rng.normal(0, 0.02, (len(sessions), len(symbols)))
        quotes = pd.DataFrame(50 * np.exp(steps.cumsum(axis=0)), index=sessions, columns=symbols)
        month_ends = sessions[:-1][sessions[:-1].month != sessions[1:].month]
        rankings = {date: list(rng.permutation(symbols)) for date in [SIGNAL, *month_ends]}
        result = run_backtest(quotes, TopNStrategy(rankings, top_n=3), SIGNAL)
        daily, fills = result.daily, result.fills

        np.testing.assert_allclose(daily.nav, daily.cash + daily.holdings)
        cash_changes = fills.groupby("date").cash_change.sum().reindex(daily.index, fill_value=0)
        np.testing.assert_allclose(daily.cash, INITIAL_CAPITAL + cash_changes.cumsum())
        self.assertGreaterEqual(daily.cash.min(), 0)
        buys, sells = fills.loc[fills.side.eq("buy")], fills.loc[fills.side.eq("sell")]
        self.assertTrue(buys.signal_date.lt(buys.date).all())
        self.assertTrue(sells.date.ge(sells.due).all())
        self.assertTrue((sells.due - sells.entry_date).eq(pd.Timedelta(days=90)).all())
        unrealized = sum(lot.units * quotes.iloc[-1][lot.symbol] - lot.cost for lot in result.open_lots)
        self.assertAlmostEqual(daily.nav.iloc[-1] - INITIAL_CAPITAL, fills.realized_pnl.sum() + unrealized, places=6)

    def test_order_budgets_ignore_prices_after_the_signal(self):
        quotes = flat_quotes("2024-03-29")
        rankings = {SIGNAL: ["A", "B"], pd.Timestamp("2024-01-31"): ["B", "A"]}
        changed = quotes.copy()
        changed.loc["2024-02-01":, "B"] = 400.0
        base = run_backtest(quotes, TopNStrategy(rankings), SIGNAL).fills
        moved = run_backtest(changed, TopNStrategy(rankings), SIGNAL).fills
        # The February 1 fill price quadruples, yet the budget set on January 31 is unchanged.
        columns = ["date", "side", "symbol", "cash_change"]
        pd.testing.assert_frame_equal(base[columns], moved[columns])
        self.assertAlmostEqual(moved.units.iloc[2] * 4, base.units.iloc[2])

    def test_performance_matches_a_hand_calculation(self):
        nav = pd.Series([1_000_000.0, 1_100_000.0, 990_000.0],
                        index=pd.to_datetime(["2024-01-02", "2024-07-02", "2025-01-01"]))
        result = performance(nav)
        returns = np.array([0.0, 0.10, -0.10])
        hurdle = 1.02 ** (1 / 252) - 1
        self.assertAlmostEqual(result.final_value, 990_000.0)
        self.assertAlmostEqual(result.cagr, 0.99 ** (365.25 / 365) - 1)
        self.assertAlmostEqual(result.max_drawdown, -0.10)
        self.assertAlmostEqual(result.sharpe, (returns.mean() - hurdle) / returns.std(ddof=1) * np.sqrt(252))


if __name__ == "__main__":
    unittest.main()
