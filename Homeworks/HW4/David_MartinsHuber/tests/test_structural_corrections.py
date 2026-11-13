"""Counterexamples from submission audit, with independent expected outcomes."""

from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from backtesting import Backtester, HeldPriceGapError, MarketData, OMS, Strategy
from research.pipeline import forward_return
from research.market import cap_weighted_market, summed_universe_volume
from tests.test_market_data import frames, filing
from tests.test_backtesting import inputs, BuyOnce


class StructuralCorrectionChecks(unittest.TestCase):
    def test_future_currency_report_cannot_change_historical_prefix(self):
        p, f, u, b, m = frames()
        original = MarketData.from_frames(p, f, u, b, m)
        future = pd.DataFrame([filing("A", "2020-03-31", "2020-05-15", 300, 90, 400, 100, currency="EUR")])
        changed = MarketData.from_frames(p, pd.concat([f, future], ignore_index=True), u, b, m)
        pd.testing.assert_frame_equal(original.prices, changed.prices)
        for date in ["2020-01-16", "2020-04-20", "2020-05-15"]:
            pd.testing.assert_frame_equal(original.membership_at(date, pd.Timestamp(date) + pd.Timedelta(days=1)),
                                          changed.membership_at(date, pd.Timestamp(date) + pd.Timedelta(days=1)))
            pd.testing.assert_frame_equal(original.financials_at(date, ["net_income"]),
                                          changed.financials_at(date, ["net_income"]))
        original_panel = original.financial_panel(fields=["net_income"])
        changed_panel = changed.financial_panel(fields=["net_income"])
        pd.testing.assert_frame_equal(
            original_panel.loc[original_panel.date.lt("2020-05-16")].reset_index(drop=True),
            changed_panel.loc[changed_panel.date.lt("2020-05-16")].reset_index(drop=True))
        self.assertIn("A", changed.non_usd_at("2020-05-16"))
        self.assertNotIn("A", changed.non_usd_at("2020-05-15"))
        self.assertTrue(pd.isna(changed.financials_at("2020-05-18", ["net_income"]).loc["A", "net_income"]))

    def test_unknown_currency_does_not_erase_known_currency_and_usd_can_restore(self):
        p, f, u, b, m = frames()
        unknown = filing("SMMT", "2020-03-31", "2020-04-15", 100, 10, 200, 100, currency=None)
        usd = filing("SMMT", "2020-03-31", "2020-05-15", 100, 10, 200, 100, currency="USD")
        data = MarketData.from_frames(p, pd.concat([f, pd.DataFrame([unknown, usd])], ignore_index=True), u, b, m)
        self.assertIn("SMMT", data.non_usd_at("2020-04-16"))
        self.assertNotIn("SMMT", data.non_usd_at("2020-05-16"))
        self.assertIn("SMMT", set(data.membership_at("2020-05-18", "2020-05-19").symbol))

    def test_currency_restriction_does_not_liquidate_existing_lots(self):
        p, f, u, b, m = frames()
        extra = filing("A", "2019-12-31", "2020-02-01", 100, 10, 200, 100, currency="EUR")
        f = f.loc[~(f.symbol.eq("A") & f.filing_date.eq("2020-02-01"))]
        data = MarketData.from_frames(p, pd.concat([f, pd.DataFrame([extra])], ignore_index=True), u, b, m)
        result = Backtester(market_data=data, strategy=BuyOnce(), initial_cash=100, fee_bps=0).run("2020-01-17", "2020-04-30")
        self.assertEqual(result.fills.side.tolist(), ["BUY", "SELL"])
        self.assertEqual(result.fills.iloc[1].date, pd.Timestamp("2020-04-16"))
        self.assertTrue(result.fills.iloc[1].date >= result.fills.iloc[1].due_date)

    def test_stale_price_option_and_oms_early_bypass_are_rejected(self):
        dates, p, b, m = inputs(8)
        with self.assertRaisesRegex(ValueError, "stale mark"):
            Backtester(p, b, m, BuyOnce(), delisting="last_price")
        oms = OMS(100, 0)
        order = oms.submit_buy("s", "A", 100, dates[0], dates[1])
        oms.fill(order, symbol="A", quote_date=dates[1], price=10, session=dates[1])
        before = (oms.cash, len(oms.orders), len(oms.lots), len(oms.fills))
        with self.assertRaises(ValueError):
            oms.forced_sale(order, date=dates[2], price=10)
        self.assertEqual(before, (oms.cash, len(oms.orders), len(oms.lots), len(oms.fills)))
        with self.assertRaisesRegex(ValueError, "Early"):
            oms.fill(order, symbol="A", quote_date=dates[1], price=10, session=dates[1], allow_early=True)

    def test_completed_reservations_release_all_cash_without_dropping_small_pending_order(self):
        oms = OMS(100, 0)
        ids = [oms.submit_buy("s", "A", amount, "2020-01-01", "2020-01-02")
               for amount in (0.1, 0.2, 0.3, 1e-12)]
        oms.fill(ids[0], symbol="A", quote_date="2020-01-02", price=10,
                 session="2020-01-02")
        oms.reject(ids[1])
        oms.discard(ids[2])
        self.assertLess(oms.available_cash, oms.cash)
        self.assertAlmostEqual(oms.cash - oms.available_cash, 1e-12, delta=1e-14)
        oms.reject(ids[3])
        self.assertEqual(oms.available_cash, oms.cash)

    def test_gap_failure_never_fabricates_sale_even_before_maturity(self):
        dates, p, b, m = inputs(8)
        p = p.loc[p.date.le(dates[1])]
        engine = Backtester(p, b, m, BuyOnce(), initial_cash=100, fee_bps=0, max_stale_sessions=1)
        with self.assertRaises(HeldPriceGapError):
            engine.run(dates[1], dates[-1])
        self.assertEqual([f["side"] for f in engine.oms.fills], ["BUY"])
        self.assertEqual(len(engine.oms.lots), 1)

    def test_forward_returns_censor_unfinished_horizons_and_match_trade_dates(self):
        tail = pd.Series([100., 110., 121.], index=pd.bdate_range("2026-09-28", periods=3))
        self.assertTrue(forward_return(tail, [tail.index[0]], days=90).isna().all())
        dates = pd.bdate_range("2020-01-01", periods=100)
        dates = dates[dates != pd.Timestamp("2020-01-06")]  # Market holiday.
        prices = pd.Series(np.arange(10., 10 + len(dates)), index=dates)
        signal = pd.Timestamp("2020-01-03")
        entry = pd.Timestamp("2020-01-07")
        due = entry + pd.Timedelta(days=90)
        exit_date = dates[dates.searchsorted(due)]
        expected = prices.loc[exit_date] / prices.loc[entry] - 1
        self.assertAlmostEqual(forward_return(prices, [signal]).iloc[0], expected)

    def test_research_sql_quarantines_invalid_off_calendar_quotes_before_lag(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        prices = pd.DataFrame({"symbol": ["A"] * 5 + ["B"] * 4,
            "date": list(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-06", "2020-01-07"])) + list(dates),
            "adj_close": [10., 0., 100., 12., 13.] + [20.] * 4,
            "volume": [100., 100., 99999., 100., 100.] + [200.] * 4})
        benchmark = pd.DataFrame({"date": dates})
        membership = pd.DataFrame({"symbol": ["A", "B"], "quarter_start": pd.to_datetime(["2020-01-01"] * 2),
                                    "effective_start": pd.to_datetime(["2020-01-02"] * 2), "market_cap_proxy": [1., 1.]})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "quotes.duckdb"
            with duckdb.connect(str(path)) as con:
                con.execute("CREATE TABLE prices AS SELECT * FROM prices")
                con.execute("CREATE TABLE benchmark AS SELECT * FROM benchmark")
                con.execute("CREATE TABLE universe_membership AS SELECT * FROM membership")
            cap = cap_weighted_market(path, "2020-01-07")
            volume = summed_universe_volume(path, "2020-01-07")
            self.assertEqual(cap.index.tolist(), list(dates))
            self.assertEqual(volume.index.tolist(), list(dates))
            self.assertEqual(volume.loc["2020-01-03"], 200.)
            self.assertEqual(cap.loc["2020-01-06", "index_return"], 0.)  # A's gap is not a one-day return.
            self.assertAlmostEqual(cap.loc["2020-01-07", "index_return"], (13 / 12 - 1) / 2)
