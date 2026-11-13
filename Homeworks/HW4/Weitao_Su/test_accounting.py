"""Small tests for cash, fees, prices, and sale dates."""
import unittest
import numpy as np
import pandas as pd
from backtester import simulate
from hw4_data import FEATURES
from run_project import metrics, predict


def example(fee=0, top_n=1):
    dates = pd.bdate_range("2024-01-01", "2024-05-15")
    quotes = pd.DataFrame({"AAA": 10.}, index=dates)
    benchmark = pd.Series(100., index=dates)
    scores = pd.DataFrame({"date": [dates[0]], "entry_date": [dates[1]],
                           "symbol": ["AAA"], "score": [1.]})
    settings = {"initial_cash": 100., "fee_bps": fee, "top_n": top_n}
    return dates, quotes, benchmark, scores, settings


class AccountingTests(unittest.TestCase):
    def test_flat_prices(self):
        d, q, b, s, cfg = example()
        run = simulate(q, b, s, d[1], d[-1], cfg)
        self.assertTrue(np.allclose(run["daily"].nav, 100))

    def test_profit(self):
        d, q, b, s, cfg = example()
        q.loc[d >= "2024-04-01", "AAA"] = 11.
        run = simulate(q, b, s, d[1], d[-1], cfg)
        # The 5% limit allows a $5 purchase. A 10% gain adds $0.50.
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 100.5)

    def test_fees(self):
        d, q, b, s, cfg = example(fee=100)
        run = simulate(q, b, s, d[1], d[-1], cfg)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 95 + 5 * .99 / 1.01)
        self.assertAlmostEqual(run["trades"].iloc[0].units, 5 / (10 * 1.01))

    def test_weekend_maturity(self):
        d, q, b, s, cfg = example()
        s.loc[0, ["date", "entry_date"]] = [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")]
        run = simulate(q, b, s, "2024-01-08", d[-1], cfg)
        sell = run["trades"].query("side == 'SELL'")
        self.assertEqual(sell.date.iloc[0], pd.Timestamp("2024-04-08"))

    def test_separate_lots_and_cash(self):
        d, q, b, s, cfg = example(top_n=10)
        second = s.assign(date=d[20], entry_date=d[21])
        run = simulate(q, b, pd.concat([s, second]), d[1], d[-1], cfg)
        self.assertEqual(len(run["trades"].query("side == 'BUY'")), 2)
        self.assertEqual(run["trades"].query("side == 'SELL'").maturity.nunique(), 2)
        self.assertTrue(run["daily"].cash.ge(0).all())

    def test_open_position_at_end(self):
        d, q, b, s, cfg = example()
        q.loc[d[10], "AAA"] = 12.
        q.loc[d[11]:, "AAA"] = 9999.
        run = simulate(q, b, s, d[1], d[10], cfg)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 101.)
        self.assertEqual(len(run["open_lots"]), 1)

    def test_missing_held_quote_stops_without_fill(self):
        d, q, b, s, cfg = example()
        q.loc[d[3], "AAA"] = np.nan
        run = simulate(q, b, s, d[1], d[-1], cfg)
        self.assertIsNotNone(run["problem"])
        self.assertEqual(run["daily"].date.iloc[-1], d[2])
        self.assertEqual(len(run["trades"]), 1)

    def test_missing_entry_keeps_cash(self):
        d, q, b, s, cfg = example()
        q.loc[d[1], "AAA"] = np.nan
        run = simulate(q, b, s, d[1], d[-1], cfg)
        self.assertTrue(run["trades"].empty)
        self.assertAlmostEqual(run["daily"].cash.iloc[-1], 100.)

    def test_same_day_signal_is_rejected(self):
        d, q, b, s, cfg = example()
        s.loc[0, "date"] = d[1]
        with self.assertRaises(AssertionError):
            simulate(q, b, s, d[1], d[-1], cfg)

    def test_flat_account_metrics(self):
        d, q, b, s, cfg = example()
        result = metrics(simulate(q, b, s, d[1], d[-1], cfg)["daily"])
        self.assertTrue(np.allclose(result.CAGR, 0))
        self.assertTrue(np.allclose(result.max_drawdown, 0))
        self.assertTrue(result.Sharpe.isna().all())

    def test_first_purchase_fee_counts_in_drawdown(self):
        d, q, b, s, cfg = example(fee=100)
        result = metrics(simulate(q, b, s, d[1], d[10], cfg)["daily"])
        self.assertAlmostEqual(result.loc["Strategy", "max_drawdown"], (95 + 5 / 1.01) / 100 - 1)

    def test_unknown_future_return_still_gets_prediction(self):
        class SimpleModel:
            def predict(self, inputs):
                self.columns = list(inputs.columns)
                return np.ones(len(inputs))

        panel = pd.DataFrame({feature: [0., 0.] for feature in FEATURES})
        panel["date"] = pd.Timestamp("2016-03-31")
        panel["entry_date"] = pd.Timestamp("2016-04-01")
        panel["symbol"] = ["AAA", "BBB"]
        panel["target"] = [0.1, np.nan]
        panel["exit_price"] = [11., np.nan]
        model = SimpleModel()
        scores = predict(model, panel, "2016-04-01", "2016-04-01", "2016-03-31")
        self.assertEqual(scores.symbol.tolist(), ["AAA", "BBB"])
        self.assertEqual(model.columns, FEATURES)

    def test_orders_are_fixed_before_the_fill_close(self):
        d, q, b, s, cfg = example()
        q["BBB"] = 10.
        q.loc[d >= "2024-03-29", "AAA"] = 20.
        q.loc[d >= "2024-04-01", "AAA"] = 22.
        second = s.assign(date=pd.Timestamp("2024-03-29"),
                          entry_date=pd.Timestamp("2024-04-01"), symbol="BBB")
        run = simulate(q, b, pd.concat([s, second]), d[1], d[-1], cfg)
        purchase = run["trades"].query("side == 'BUY' and symbol == 'BBB'").iloc[0]
        self.assertAlmostEqual(-purchase.cash_change, 5.25)

    def test_cash_merger_keeps_money_reserved_until_maturity(self):
        d, q, b, s, cfg = example()
        q.loc["2024-01-11":, "AAA"] = np.nan
        deal = {"symbol": "AAA", "effective_date": pd.Timestamp("2024-01-10"),
                "available_date": pd.Timestamp("2024-01-11"), "cash_price": 12., "source": "test"}
        run = simulate(q, b, s, d[1], "2024-01-19", cfg, [deal])
        self.assertIsNone(run["problem"])
        self.assertAlmostEqual(run["daily"].cash.iloc[-1], 95.)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 101.)
        self.assertEqual(len(run["trades"]), 1)
        complete = simulate(q, b, s, d[1], d[-1], cfg, [deal])
        release = complete["trades"].query("side == 'CASH_RELEASE'").iloc[0]
        self.assertEqual(release.date, pd.Timestamp("2024-04-01"))
        self.assertEqual(release.fee, 0.)
        self.assertAlmostEqual(complete["daily"].nav.iloc[-1], 101.)

    def test_future_merger_does_not_hide_an_earlier_missing_price(self):
        d, q, b, s, cfg = example()
        q.loc["2024-01-09", "AAA"] = np.nan
        deal = {"symbol": "AAA", "effective_date": pd.Timestamp("2024-01-10"),
                "available_date": pd.Timestamp("2024-01-11"), "cash_price": 12., "source": "test"}
        run = simulate(q, b, s, d[1], d[-1], cfg, [deal])
        self.assertIsNotNone(run["problem"])
        self.assertEqual(run["daily"].date.iloc[-1], pd.Timestamp("2024-01-08"))

    def test_cash_merger_adjusted_unit_conversion(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import duckdb
        from hw4_data import load_actions

        deal = pd.DataFrame({"symbol": ["AAA"], "effective_date": [pd.Timestamp("2024-01-10")],
                             "available_date": [pd.Timestamp("2024-01-11")], "cash_per_share": [24.], "source": ["test"]})
        with tempfile.TemporaryDirectory(prefix="hw4-merger-test-") as folder:
            database = Path(folder) / "test.duckdb"
            with duckdb.connect(str(database)) as con:
                con.execute("CREATE TABLE prices(symbol VARCHAR,date DATE,close DOUBLE,adj_close DOUBLE)")
                con.execute("INSERT INTO prices VALUES ('AAA','2024-01-10',20,10)")
            with patch("hw4_data.pd.read_csv", return_value=deal):
                actions = load_actions(database, "2024-01-11")
            self.assertAlmostEqual(actions[0]["cash_price"], 12.)

    def test_share_exchange_keeps_original_maturity(self):
        d, q, b, s, cfg = example()
        q["BBB"] = 6.
        q.loc["2024-01-11":, "AAA"] = np.nan
        deal = {"kind": "stock", "symbol": "AAA", "new_symbol": "BBB", "unit_ratio": 2.,
                "effective_date": pd.Timestamp("2024-01-10"),
                "available_date": pd.Timestamp("2024-01-11"), "source": "test"}
        run = simulate(q, b, s, d[1], d[-1], cfg, [deal])
        self.assertIsNone(run["problem"])
        sale = run["trades"].query("side == 'SELL'").iloc[0]
        self.assertEqual(sale.symbol, "BBB")
        self.assertEqual(sale.date, pd.Timestamp("2024-04-01"))
        self.assertAlmostEqual(sale.units, 1.)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 101.)

    def test_mixed_deal_charges_sale_fee_only_on_stock(self):
        d, q, b, s, cfg = example(fee=100)
        q["BBB"] = 6.
        q.loc["2024-01-11":, "AAA"] = np.nan
        deal = {"kind": "mixed", "symbol": "AAA", "new_symbol": "BBB", "unit_ratio": .25,
                "cash_per_unit": 8., "effective_date": pd.Timestamp("2024-01-10"),
                "available_date": pd.Timestamp("2024-01-11"), "source": "test"}
        run = simulate(q, b, s, d[1], d[-1], cfg, [deal])
        sale = run["trades"].query("side == 'SELL'").iloc[0]
        old_units = 5 / 10.1
        self.assertAlmostEqual(sale.fee, old_units * .25 * 6 * .01)
        self.assertAlmostEqual(sale.cash_component, old_units * 8)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 95 + old_units * (8 + .25 * 6 * .99))

    def test_same_ticker_deal_is_applied_once_and_keeps_unpriced_right(self):
        d, q, b, s, cfg = example()
        q.loc["2024-01-11":, "AAA"] = 6.
        deal = {"kind": "mixed", "symbol": "AAA", "new_symbol": "AAA", "unit_ratio": .25,
                "cash_per_unit": 8., "cvr_per_unit": 1., "cvr_max": 5.,
                "effective_date": pd.Timestamp("2024-01-10"),
                "available_date": pd.Timestamp("2024-01-11"), "source": "test"}
        run = simulate(q, b, s, d[1], d[-1], cfg, [deal])
        self.assertEqual(len(run["events"]), 1)
        self.assertEqual(len(run["claims"]), 1)
        self.assertAlmostEqual(run["claims"].units.iloc[0], .5)
        self.assertAlmostEqual(run["claims"].assumed_value.iloc[0], 0.)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 99.75)

    def test_later_company_reusing_a_ticker_does_not_get_old_payout(self):
        d, q, b, s, cfg = example()
        deal = {"symbol": "AAA", "effective_date": pd.Timestamp("2023-01-10"),
                "available_date": pd.Timestamp("2023-01-11"), "cash_price": 999., "source": "test"}
        run = simulate(q, b, s, d[1], d[-1], cfg, [deal])
        self.assertTrue(run["events"].empty)
        self.assertAlmostEqual(run["daily"].nav.iloc[-1], 100.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
