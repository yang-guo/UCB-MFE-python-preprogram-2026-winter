"""Hand-checkable artificial accounting fixtures, not substitute market data."""

import unittest

import numpy as np
import pandas as pd

from backtester import DataGapError, StrategyConfig, entry_schedule, performance, run_backtest


class AccountingTests(unittest.TestCase):
    def setUp(self):
        # This fixture treats every weekday as a session; it is not a US calendar.
        self.calendar = pd.bdate_range("2015-01-02", "2015-07-31")
        self.prices = pd.DataFrame({"date": self.calendar, "symbol": "A", "adj_close": 100.0})
        self.scores = pd.DataFrame({"date": [self.calendar[0]], "symbol": ["A"], "prediction": [0.1]})
        self.config = StrategyConfig(initial_cash=1000, top_n=1, monthly_fraction=1, cost_bps=0)

    def run_case(self, prices=None, scores=None, end="2015-07-31", config=None):
        return run_backtest(
            self.prices if prices is None else prices, self.calendar,
            self.scores if scores is None else scores,
            "2015-01-05", end, self.config if config is None else config,
        )

    def test_round_trip_fees_and_weekend_maturity(self):
        prices = self.prices.copy()
        prices.loc[prices.date.ge("2015-04-06"), "adj_close"] = 110.0
        config = StrategyConfig(initial_cash=1000, top_n=1, monthly_fraction=1, cost_bps=100)
        result = self.run_case(prices=prices, config=config)
        buy, sell = result.fills.iloc[0], result.fills.iloc[1]
        self.assertEqual(buy.date, pd.Timestamp("2015-01-05"))
        self.assertEqual(sell.date, pd.Timestamp("2015-04-06"))  # Jan 5 + 90 is Sunday.
        self.assertAlmostEqual(buy.notional + buy.fee, 1000)
        self.assertAlmostEqual(sell.notional, (1000 / 1.01 / 100) * 110)
        self.assertAlmostEqual(result.history.nav.iloc[-1], (1000 / 1.01) * 1.1 * 0.99)
        self.assertEqual(len(result.open_lots), 0)

    def test_terminal_open_position_included_without_forced_sale(self):
        result = self.run_case(end="2015-01-09")
        self.assertEqual(result.fills.side.tolist(), ["BUY"])
        self.assertEqual(len(result.open_lots), 1)
        self.assertAlmostEqual(result.history.cash.iloc[-1], 0)
        self.assertAlmostEqual(result.history.nav.iloc[-1], 1000)
        self.assertAlmostEqual(result.open_lots.market_value.sum(), 1000)

    def test_cash_not_spent_twice_across_multiple_orders(self):
        prices = pd.concat([self.prices, self.prices.assign(symbol="B", adj_close=200.0)])
        scores = pd.concat([self.scores, self.scores.assign(symbol="B")])
        config = StrategyConfig(initial_cash=1000, top_n=2, monthly_fraction=1, cost_bps=100)
        result = self.run_case(prices, scores, end="2015-01-09", config=config)
        self.assertAlmostEqual((result.fills.notional + result.fills.fee).sum(), 1000)
        self.assertTrue(result.history.cash.ge(0).all())
        np.testing.assert_allclose(result.history.nav, result.history.cash + result.history.market_value)

    def test_only_prior_session_scores_can_place_orders(self):
        scores = self.scores.assign(date=pd.Timestamp("2015-01-05"))
        result = self.run_case(scores=scores)
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.history.nav.eq(1000).all())

    def test_entry_fill_price_does_not_change_preplanned_budget(self):
        prices = self.prices.copy()
        prices.loc[prices.date.ge("2015-01-05"), "adj_close"] = 200
        result = self.run_case(prices=prices, end="2015-01-09")
        self.assertAlmostEqual(result.fills.iloc[0].notional, 1000)
        self.assertAlmostEqual(result.fills.iloc[0].units, 5)

    def test_same_ticker_lots_have_independent_maturities(self):
        entries = entry_schedule(self.calendar, "2015-01-05", "2015-03-31")
        signals = self.calendar[self.calendar.get_indexer(entries) - 1]
        scores = pd.DataFrame({"date": signals, "symbol": "A", "prediction": 0.1})
        config = StrategyConfig(initial_cash=1000, top_n=1, monthly_fraction=1 / 3, cost_bps=0)
        result = self.run_case(scores=scores, config=config)
        buys = result.fills.loc[result.fills.side.eq("BUY")].set_index("lot_id")
        sells = result.fills.loc[result.fills.side.eq("SELL")].set_index("lot_id")
        self.assertEqual(len(buys), 3)
        for lot_id, buy in buys.iterrows():
            due = buy.date + pd.Timedelta(days=90)
            expected = self.calendar[self.calendar.searchsorted(due)]
            self.assertEqual(sells.loc[lot_id, "date"], expected)
        self.assertAlmostEqual(result.history.nav.iloc[-1], 1000)

    def test_missing_entry_is_cancelled_without_redistribution(self):
        prices = pd.concat([self.prices, self.prices.assign(symbol="B")])
        prices = prices.loc[~(prices.symbol.eq("B") & prices.date.eq("2015-01-05"))]
        scores = pd.concat([self.scores, self.scores.assign(symbol="B")])
        config = StrategyConfig(initial_cash=1000, top_n=2, monthly_fraction=1, cost_bps=0)
        result = self.run_case(prices, scores, end="2015-01-09", config=config)
        self.assertEqual(result.fills.symbol.tolist(), ["A"])
        self.assertAlmostEqual(result.history.cash.iloc[-1], 500)
        self.assertEqual(result.diagnostics.event.tolist(), ["missing_entry_quote"])

    def test_missing_maturity_quote_stops_instead_of_fabricating_sale(self):
        prices = self.prices.loc[self.prices.date.ne("2015-04-06")]
        with self.assertRaisesRegex(DataGapError, "Missing maturity exit"):
            self.run_case(prices=prices)

    def test_short_quote_gap_is_flagged_and_long_gap_stops(self):
        prices = self.prices.loc[self.prices.date.ne("2015-01-06")]
        result = self.run_case(prices=prices, end="2015-01-09")
        self.assertEqual(result.history.loc["2015-01-06", "stale_lots"], 1)
        prices = self.prices.loc[~self.prices.date.between("2015-01-06", "2015-01-14")]
        with self.assertRaisesRegex(DataGapError, "Held quote gap"):
            self.run_case(prices=prices)

    def test_first_entry_fee_counts_toward_drawdown(self):
        config = StrategyConfig(initial_cash=1000, top_n=1, monthly_fraction=1, cost_bps=100)
        result = self.run_case(end="2015-01-09", config=config)
        metrics = performance(result.history.nav, initial_cash=1000)
        self.assertAlmostEqual(metrics["max_drawdown"], 1 / 1.01 - 1)

    def test_nonpositive_predictions_leave_cash_uninvested(self):
        result = self.run_case(scores=self.scores.assign(prediction=-0.1))
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.history.nav.eq(1000).all())

    def test_matrix_input_matches_long_price_input(self):
        expected = self.run_case(end="2015-01-09")
        quotes = self.prices.pivot(index="date", columns="symbol", values="adj_close")
        actual = run_backtest(None, self.calendar, self.scores, "2015-01-05",
                              "2015-01-09", self.config, price_matrix=quotes)
        pd.testing.assert_frame_equal(actual.history, expected.history)
        pd.testing.assert_frame_equal(actual.fills, expected.fills)

    def test_same_close_sale_does_not_expand_prior_cash_budget(self):
        calendar = self.calendar[~self.calendar.isin(pd.bdate_range("2015-04-01", "2015-04-03"))]
        scores = pd.concat([self.scores, self.scores.assign(date=pd.Timestamp("2015-03-31"))])
        result = run_backtest(self.prices, calendar, scores, "2015-01-05",
                              "2015-04-07", self.config)
        self.assertEqual(result.fills.side.tolist(), ["BUY", "SELL"])
        self.assertAlmostEqual(result.history.cash.iloc[-1], 1000)

    def test_mandatory_stock_conversion_preserves_original_lot_maturity(self):
        old = self.prices.loc[self.prices.date.lt("2015-02-02")]
        new = self.prices.assign(symbol="B", adj_close=50.0)
        new.loc[new.date.ge("2015-04-06"), "adj_close"] = 55.0
        prices = pd.concat([old, new])
        actions = [{"effective_date": "2015-02-02", "old_symbol": "A",
                    "new_symbol": "B", "share_exchange_ratio": 2.0,
                    "adjusted_unit_ratio": 2.0}]
        result = run_backtest(prices, self.calendar, self.scores, "2015-01-05",
                              "2015-07-31", self.config, corporate_actions=actions)
        self.assertEqual(result.fills.side.tolist(), ["BUY", "SELL"])
        self.assertEqual(result.fills.symbol.tolist(), ["A", "B"])
        self.assertEqual(result.fills.iloc[1].date, pd.Timestamp("2015-04-06"))
        self.assertAlmostEqual(result.fills.iloc[1].units, 20)
        self.assertAlmostEqual(result.history.nav.iloc[-1], 1100)
        self.assertEqual(len(result.action_ledger), 1)
        self.assertEqual(result.action_ledger.iloc[0].original_maturity, pd.Timestamp("2015-04-05"))

    def test_cash_merger_entitlement_is_reserved_until_original_maturity(self):
        prices = self.prices.loc[self.prices.date.lt("2015-02-02")]
        actions = [{"kind": "cash", "effective_date": "2015-02-02", "old_symbol": "A",
                    "new_symbol": "CASH:A", "cash_per_adjusted_unit": 90.0}]
        result = run_backtest(prices, self.calendar, self.scores, "2015-01-05",
                              "2015-07-31", self.config, corporate_actions=actions)
        self.assertAlmostEqual(result.history.loc["2015-02-02", "nav"], 900)
        self.assertAlmostEqual(result.history.loc["2015-02-02", "cash"], 0)
        self.assertEqual(result.fills.side.tolist(), ["BUY", "SETTLEMENT"])
        self.assertEqual(result.fills.iloc[1].date, pd.Timestamp("2015-04-06"))
        self.assertAlmostEqual(result.history.cash.iloc[-1], 900)
        self.assertAlmostEqual(result.action_ledger.cash_amount.sum(), 900)

    def test_rename_maps_existing_lots_and_later_old_symbol_orders(self):
        entries = entry_schedule(self.calendar, "2015-01-05", "2015-03-31")
        signals = self.calendar[self.calendar.get_indexer(entries) - 1]
        scores = pd.DataFrame({"date": signals, "symbol": "A", "prediction": 0.1})
        prices = pd.concat([self.prices.loc[self.prices.date.lt("2015-02-02")],
                            self.prices.assign(symbol="B")])
        actions = [{"kind": "rename", "effective_date": "2015-02-02", "old_symbol": "A",
                    "new_symbol": "B", "share_exchange_ratio": 1.0, "adjusted_unit_ratio": 1.0}]
        config = StrategyConfig(initial_cash=1000, top_n=1, monthly_fraction=1 / 3, cost_bps=0)
        result = run_backtest(prices, self.calendar, scores, "2015-01-05",
                              "2015-07-31", config, corporate_actions=actions)
        self.assertEqual(result.fills.loc[result.fills.side.eq("BUY"), "symbol"].tolist(), ["A", "B", "B"])
        self.assertTrue(result.fills.loc[result.fills.side.eq("SELL"), "symbol"].eq("B").all())
        self.assertAlmostEqual(result.history.nav.iloc[-1], 1000)
        self.assertEqual(result.action_ledger.event.tolist(),
                          ["mandatory_symbol_change", "entry_symbol_alias", "entry_symbol_alias"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
