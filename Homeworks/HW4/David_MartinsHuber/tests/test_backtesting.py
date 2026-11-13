"""Hand-verifiable accounting and timing cases; no external database needed."""

import unittest

import numpy as np
import pandas as pd

from backtesting import Backtester, HeldPriceGapError, OMS, Phase, Strategy


class BuyOnce(Strategy):
    def __init__(self):
        super().__init__("test_strategy")
        self.sent = False

    def propose_buys(self, context):
        if self.sent:
            return {}
        self.sent = True
        return {"A": 100.0}


def inputs(periods=75):
    dates = pd.bdate_range("2020-01-02", periods=periods)
    prices = pd.DataFrame({"date": dates, "symbol": "A", "adj_close": 10.0})
    benchmark = pd.DataFrame({"date": dates, "adj_close": 100.0})
    membership = pd.DataFrame({"symbol": ["A", "A"],
                               "quarter_start": pd.to_datetime(["2020-01-01", "2020-04-01"]),
                               "selection_date": pd.to_datetime(["2019-12-31", "2020-03-31"])})
    return dates, prices, benchmark, membership


class BacktestingChecks(unittest.TestCase):
    def test_flat_price_round_trip_and_strategy_identity(self):
        dates, p, b, m = inputs()
        result = Backtester(p, b, m, BuyOnce(), initial_cash=100, fee_bps=100).run(dates[1], dates[-1])
        buy, sell = result.fills.iloc[0], result.fills.iloc[1]
        expected_cash = 100 * .99 / 1.01
        self.assertAlmostEqual(result.history.cash.iloc[-1], expected_cash)
        self.assertEqual(buy.date, dates[1])
        self.assertEqual(sell.date, dates[dates.searchsorted(dates[1] + pd.Timedelta(days=90))])
        self.assertEqual(set(result.fills.strategy_id), {"test_strategy"})
        self.assertEqual(sell.trigger, "maturity")
        self.assertEqual(set(result.orders.status), {"filled"})
        np.testing.assert_allclose(result.history.nav, result.history.cash + result.history.holdings_value)
        flows = result.fills.groupby("date").cash_change.sum().reindex(result.history.index, fill_value=0)
        np.testing.assert_allclose(result.history.cash, 100 + flows.cumsum(), atol=1e-10)
        self.assertAlmostEqual(result.pnl.total, result.pnl.realized + result.pnl.unrealized)

    def test_reservation_quote_validation_and_duplicate_fill(self):
        oms = OMS(100, 0)
        order = oms.submit_buy("s", "A", 100, "2020-01-03", "2020-01-06")
        self.assertEqual(oms.cash, 100)
        self.assertEqual(oms.available_cash, 0)
        with self.assertRaises(ValueError):
            oms.submit_buy("s", "B", 1, "2020-01-03", "2020-01-06")
        for symbol, date, price in [("B", "2020-01-06", 10), ("A", "2020-01-03", 10),
                                    ("A", "2020-01-06", np.nan)]:
            with self.assertRaises(ValueError):
                oms.fill(order, symbol=symbol, quote_date=date, price=price, session="2020-01-06")
        oms.fill(order, symbol="A", quote_date="2020-01-06", price=20, session="2020-01-06")
        self.assertEqual(oms.lots[order].units, 5)
        self.assertEqual(oms.available_cash, 0)
        with self.assertRaises(ValueError):
            oms.fill(order, symbol="A", quote_date="2020-01-06", price=20, session="2020-01-06")
        with self.assertRaises(ValueError):
            oms.submit_maturity_sale(order, "2020-01-07")

    def test_missing_entry_and_stale_holdings(self):
        dates, p, b, m = inputs(5)
        missing = p.loc[p.date.ne(dates[1])]
        engine = Backtester(missing, b, m, BuyOnce(), initial_cash=100)
        result = engine.run(dates[1], dates[-1])
        self.assertTrue(result.fills.empty)
        self.assertEqual(engine.oms.available_cash, 100)
        missing = p.loc[p.date.ne(dates[2])]
        result = Backtester(missing, b, m, BuyOnce()).run(dates[1], dates[-1])
        self.assertEqual(result.history.stale_marks.sum(), 1)
        self.assertEqual(len(result.open_lots), 1)
        with self.assertRaises(HeldPriceGapError):
            Backtester(missing, b, m, BuyOnce(), max_stale_sessions=0).run(dates[1], dates[-1])

    def test_market_holiday_and_future_price_independence(self):
        dates, p, b, m = inputs(6)
        # Remove Monday from the market calendar: Friday's next session is Tuesday.
        holiday = pd.Timestamp("2020-01-06")
        p, b = p.loc[p.date.ne(holiday)], b.loc[b.date.ne(holiday)]
        start = pd.Timestamp("2020-01-07")
        first = Backtester(p, b, m, BuyOnce(), fee_bps=0).run(start, dates[-1])
        p.loc[p.date.eq(start), "adj_close"] = 20
        second = Backtester(p, b, m, BuyOnce(), fee_bps=0).run(start, dates[-1])
        self.assertEqual(first.orders.budget.iloc[0], second.orders.budget.iloc[0])
        self.assertEqual(second.fills.signal_date.iloc[0], pd.Timestamp("2020-01-03"))
        self.assertEqual(first.fills.units.iloc[0], 2 * second.fills.units.iloc[0])

    def test_refit_filters_labels_and_preserves_active_model_on_failure(self):
        class Learner(BuyOnce):
            def needs_refit(self, context):
                return context.model_version == 0

            def refit(self, data, context):
                self.training = data
                self.latest_seen = context.prices.index.max()

        dates, p, b, m = inputs(5)
        panel = pd.DataFrame({"date": pd.to_datetime(["2019-09-01", "2019-10-01"]),
                              "symbol": ["A", "A"], "available_date": pd.to_datetime(["2019-09-01", "2019-10-01"]),
                              "label_end": [dates[0] - pd.Timedelta(days=1), dates[-1]],
                              "target": [.1, 999.0], "feature": [1.0, 999.0]})
        engine = Backtester(p, b, m, Learner(), training_panel=panel, feature_columns=["feature"])
        result = engine.run(dates[1], dates[-1])
        self.assertEqual(engine.strategy.training.y.tolist(), [.1])
        self.assertEqual(engine.strategy.latest_seen, dates[0])
        self.assertEqual(len(result.refits), 1)
        self.assertIn("refitting", result.phases.phase.tolist())

        class Broken(Learner):
            def refit(self, data, context):
                self.corrupted = True
                raise RuntimeError("fit failed")

        original = Broken()
        engine = Backtester(p, b, m, original, training_panel=panel, feature_columns=["feature"])
        with self.assertRaises(RuntimeError):
            engine.run(dates[1], dates[-1])
        self.assertFalse(hasattr(original, "corrupted"))
        self.assertEqual(engine.phase, Phase.FAILED)
        self.assertFalse(engine.oms.fills)

    def test_cash_only_universe_and_repeated_run(self):
        class CashOnly(Strategy):
            def propose_buys(self, context):
                return {}

        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, CashOnly("cash"))
        result = engine.run(dates[1], dates[-1])
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.history.nav.eq(1_000_000).all())
        self.assertTrue(np.isnan(result.performance().loc["Sharpe", "Strategy"]))
        with self.assertRaises(ValueError):
            engine.run(dates[1], dates[-1])
        with self.assertRaises(ValueError):
            Backtester(p, b, m.iloc[0:0], BuyOnce()).run(dates[1], dates[-1])

    def test_multiple_lots_and_delayed_maturity(self):
        class Twice(BuyOnce):
            def propose_buys(self, context):
                if context.execution_date in [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]:
                    return {"A": 100}
                return {}

        dates, p, b, m = inputs()
        due = pd.Timestamp("2020-04-02")
        p = p.loc[p.date.ne(due)]
        result = Backtester(p, b, m, Twice(), initial_cash=200).run(dates[1], dates[-1])
        self.assertEqual(result.fills.lot_id.nunique(), 2)
        self.assertEqual(len(result.fills), 4)
        first_sale = result.fills.loc[result.fills.side.eq("SELL")].iloc[0]
        self.assertEqual(first_sale.date, pd.Timestamp("2020-04-03"))
        self.assertIn("maturity_exit_delayed", result.diagnostics.event.tolist())


if __name__ == "__main__":
    unittest.main()
