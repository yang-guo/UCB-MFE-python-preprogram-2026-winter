"""Point-in-time panel tests; synthetic data, no external database."""

import unittest

import numpy as np
import pandas as pd

from backtesting import Backtester, MarketData, Strategy


def filing(symbol, fiscal, filed, revenue, net, assets, shares, *, period="quarter",
           currency="USD", debt=50.0, cash=20.0, fcf=5.0, ocf=15.0, operating=20.0):
    return dict(symbol=symbol, fiscal_date=pd.Timestamp(fiscal), period=period,
                filing_date=pd.Timestamp(filed), revenue=revenue, net_income=net,
                operating_income=operating, total_assets=assets, total_debt=debt,
                cash_and_equivalents=cash, free_cash_flow=fcf, operating_cash_flow=ocf,
                reported_currency=currency, shares_outstanding=shares, market_cap=shares * 10.0)


def frames():
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    prices = pd.DataFrame([
        dict(date=d, symbol=symbol, close=base, adj_close=base, volume=volume)
        for symbol, base, volume in [("A", 10.0, 1000), ("B", 20.0, 2000), ("SMMT", 30.0, 3000)]
        for d in dates
    ])
    benchmark = pd.DataFrame({"date": dates, "adj_close": 100.0})
    universe = pd.DataFrame({"symbol": ["A", "B", "SMMT"],
                             "sector": ["Tech", "Fin", "Health"]})
    membership = pd.DataFrame({
        "symbol": ["A", "B", "SMMT", "A", "B", "SMMT"],
        "quarter_start": pd.to_datetime(["2020-01-01"] * 3 + ["2020-04-01"] * 3),
        "selection_date": pd.to_datetime(["2019-12-31"] * 3 + ["2020-03-31"] * 3),
        "universe_rank": [1, 2, 3, 1, 2, 3],
    })
    fundamentals = pd.DataFrame([
        filing("A", "2019-12-31", "2020-01-15", 100.0, 10.0, 200.0, 100.0),
        # Restatement of the same fiscal period, filed later.
        filing("A", "2019-12-31", "2020-02-01", 100.0, 12.0, 200.0, 100.0),
        filing("A", "2020-03-31", "2020-04-15", 200.0, 40.0, 250.0, 200.0),
        filing("B", "2019-12-31", "2020-01-15", 80.0, 8.0, 160.0, 80.0),
        # Old period filed after Q1 is already known; must not replace Q1.
        filing("A", "2019-09-30", "2020-04-20", 50.0, 1.0, 150.0, 100.0),
        filing("SMMT", "2019-12-31", "2020-01-15", 100.0, 10.0, 200.0, 100.0, currency="GBP"),
        # Annual rows are not quarterly events.
        filing("A", "2019-12-31", "2020-01-15", 400.0, 40.0, 200.0, 100.0, period="annual"),
    ])
    return prices, fundamentals, universe, benchmark, membership


class MarketDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = MarketData.from_frames(*frames())

    def test_non_usd_restrictions_are_dated_and_history_is_preserved(self):
        self.assertEqual(self.data.non_usd_symbols, ("SMMT",))
        self.assertEqual(set(self.data.prices.symbol), {"A", "B", "SMMT"})
        self.assertIn("SMMT", set(self.data.membership_at("2020-01-15", "2020-01-16").symbol))
        self.assertNotIn("SMMT", set(self.data.membership_at("2020-01-16", "2020-01-17").symbol))
        row = self.data.financials_at("2020-01-16", fields=["net_income"]).loc["SMMT"]
        self.assertTrue(pd.isna(row.net_income))

    def test_no_backfill_before_first_filing(self):
        row = self.data.financials_at("2020-01-10", fields=["net_income"]).loc["A"]
        self.assertTrue(pd.isna(row.net_income))
        self.assertTrue(pd.isna(row.available_date))

    def test_forward_fill_until_next_update(self):
        def value(date):
            return self.data.financials_at(date, fields=["net_income"]).loc["A", "net_income"]
        self.assertTrue(pd.isna(value("2020-01-15")))   # filing day, not yet available
        self.assertEqual(value("2020-01-16"), 10.0)     # original filing
        self.assertEqual(value("2020-01-31"), 10.0)     # before restatement available
        self.assertEqual(value("2020-02-03"), 12.0)     # restatement (available Sunday)
        self.assertEqual(value("2020-04-15"), 12.0)     # Q1 filing day, not yet available
        self.assertEqual(value("2020-04-16"), 40.0)     # Q1
        self.assertEqual(value("2020-06-30"), 40.0)     # persists, no staleness cap

    def test_late_old_period_does_not_replace_newer_filing(self):
        value = self.data.financials_at("2020-04-21", fields=["net_income"]).loc["A", "net_income"]
        self.assertEqual(value, 40.0)

    def test_price_history_is_preserved(self):
        panel = self.data.financial_panel(fields=["net_income"],
                                          start="2020-01-02", end="2020-01-10")
        counts = panel.groupby("symbol").size().to_dict()
        expected = (self.data.prices
                    .loc[self.data.prices.date.between("2020-01-02", "2020-01-10")]
                    .groupby("symbol").size().to_dict())
        self.assertEqual(counts, expected)

    def test_market_cap_uses_raw_usd_close(self):
        early = self.data.financials_at("2020-01-16", fields=["market_cap_usd"]).loc["A"]
        self.assertEqual(early.market_cap_usd, 1000.0)  # 100 shares * $10
        late = self.data.financials_at("2020-04-16", fields=["market_cap_usd"]).loc["A"]
        self.assertEqual(late.market_cap_usd, 2000.0)   # 200 shares * $10

    def test_volume_panel(self):
        volume = self.data.volume_panel()
        self.assertEqual(volume.loc["2020-01-02", "A"], 1000)
        self.assertEqual(volume.loc["2020-06-30", "B"], 2000)
        self.assertIn("SMMT", volume.columns)  # Historical quotes/volume remain available.

    def test_view_stops_at_signal_close(self):
        view = self.data.view("2020-01-16", "2020-01-17", financial_fields=["net_income"])
        self.assertEqual(view.prices.index.max(), pd.Timestamp("2020-01-16"))
        self.assertEqual(view.volume.index.max(), pd.Timestamp("2020-01-16"))
        self.assertEqual(view.financials.loc["A", "net_income"], 10.0)

    def test_engine_supplies_declared_financials_and_volume(self):
        class UsesData(Strategy):
            required_financials = ("net_margin",)
            def __init__(self):
                super().__init__("uses_data")
                self.seen = []
            def propose_buys(self, context):
                self.seen.append((context.signal_date,
                                  context.financials.loc["A", "net_margin"],
                                  context.volume.loc[context.signal_date, "A"]))
                return {}

        engine = Backtester(market_data=self.data, strategy=UsesData())
        engine.run("2020-01-17", "2020-01-21")
        signal, margin, volume = engine.strategy.seen[0]
        self.assertEqual(signal, pd.Timestamp("2020-01-16"))
        self.assertAlmostEqual(margin, 0.10)  # 10 / 100
        self.assertEqual(volume, 1000)

    def test_unknown_required_financial_fails_at_construction(self):
        class Bad(Strategy):
            required_financials = ("not_a_column",)
            def __init__(self):
                super().__init__("bad")
            def propose_buys(self, context):
                return {}
        with self.assertRaisesRegex(ValueError, "Unknown financial fields"):
            Backtester(market_data=self.data, strategy=Bad())

    def test_training_panel_has_engine_schema(self):
        panel = self.data.training_panel(
            feature_columns=("momentum_21", "net_margin"),
            start="2020-01-02", end="2020-06-30")
        for column in ("date", "symbol", "available_date", "label_end", "target",
                       "momentum_21", "net_margin"):
            self.assertIn(column, panel.columns)
        known = panel.available_date.notna()
        self.assertTrue((panel.loc[known, "available_date"] <= panel.loc[known, "date"]).all())
        self.assertFalse(panel.duplicated(["date", "symbol"]).any())


if __name__ == "__main__":
    unittest.main()
