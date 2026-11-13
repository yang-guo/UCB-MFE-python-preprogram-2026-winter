"""Behavioral equivalence and adversarial checks for the isolated sale index."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from backtesting import Backtester as OriginalBacktester
from backtesting import OMS, ShadowPortfolioBacktester as OriginalCoordinator
from backtesting.indexed_sales import Backtester, IndexedOMS, ShadowPortfolioBacktester
from tests.test_backtesting import inputs
from tests.test_portfolio import FractionStrategy


def purchased(oms):
    order = oms.submit_buy("s", "A", 100, "2020-01-01", "2020-01-02")
    oms.fill(order, symbol="A", quote_date="2020-01-02", price=10, session="2020-01-02")
    return order


class IndexedSaleChecks(unittest.TestCase):
    def test_date_progress_records_only_completed_sessions(self):
        dates, prices, benchmark, members = inputs(12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = ShadowPortfolioBacktester(
                    prices, benchmark, members, [FractionStrategy("a")],
                    initial_cash=1000, fee_bps=0).run(
                        dates[1], dates[-1], progress_every=3, progress_path=path)
            state = json.loads(path.read_text())
            self.assertEqual(state["simulated_date"], str(result.portfolio.history.index[-1].date()))
            self.assertEqual(state["completed_sessions"], 11)
            self.assertEqual(state["total_sessions"], 11)
            self.assertEqual(state["percent_sessions"], 100.)
            records = [json.loads(line.split(": ", 1)[1]) for line in output.getvalue().splitlines()]
            self.assertEqual([r["completed_sessions"] for r in records], [1, 3, 6, 9, 11])

    def test_maturity_and_quote_guards_and_single_fill(self):
        oms = IndexedOMS(100, 5)
        lot = purchased(oms)
        with self.assertRaises(ValueError):
            oms.submit_maturity_sale(lot, "2020-03-31")
        sale = oms.submit_maturity_sale(lot, "2020-04-01")
        with self.assertRaises(ValueError):
            oms.submit_maturity_sale(lot, "2020-04-02")
        for quote, price in [("2020-03-31", 10), ("2020-04-01", float("nan"))]:
            before = deepcopy(oms.__dict__)
            with self.assertRaises(ValueError):
                oms.fill(sale, symbol="A", quote_date=quote, price=price, session="2020-04-01")
            self.assertEqual(oms.__dict__, before)
        oms.fill(sale, symbol="A", quote_date="2020-04-01", price=12, session="2020-04-01")
        self.assertFalse(oms._pending_sales)
        self.assertEqual(oms.fills[-1]["trigger"], "maturity")
        self.assertGreaterEqual(oms.fills[-1]["date"], oms.fills[-1]["due_date"])
        with self.assertRaises(ValueError):
            oms.fill(sale, symbol="A", quote_date="2020-04-01", price=12, session="2020-04-01")

    def test_reject_and_discard_allow_a_replacement_sale(self):
        for action in ("reject", "discard"):
            with self.subTest(action=action):
                oms = IndexedOMS(100, 0)
                lot = purchased(oms)
                first = oms.submit_maturity_sale(lot, "2020-04-01")
                getattr(oms, action)(first)
                self.assertFalse(oms._pending_sales)
                oms.submit_maturity_sale(lot, "2020-04-02")

    def test_lookup_does_not_traverse_historical_orders(self):
        class NoScan(dict):
            def values(self):
                raise AssertionError("Historical orders must not be scanned")

        oms = IndexedOMS(100, 0)
        lot = purchased(oms)
        oms.orders = NoScan(oms.orders)
        sale = oms.submit_maturity_sale(lot, "2020-04-01")
        self.assertEqual(oms.orders[sale].lot_id, lot)
        with self.assertRaises(ValueError):
            oms.submit_maturity_sale(lot, "2020-04-01")

    def test_single_account_complete_ledgers_match(self):
        dates, prices, benchmark, members = inputs(120)
        prices.adj_close = 10 + pd.Series(range(len(prices))) / 100
        results = []
        for implementation in (OriginalBacktester, Backtester):
            result = implementation(prices, benchmark, members, FractionStrategy("a"),
                                    initial_cash=1000, fee_bps=5).run(dates[1], dates[-1])
            self.assertTrue(result.fills.side.eq("SELL").any())
            results.append(result)
        for field in ("history", "fills", "orders", "open_lots", "diagnostics", "refits", "phases"):
            pd.testing.assert_frame_equal(getattr(results[0], field), getattr(results[1], field))
        pd.testing.assert_series_equal(results[0].pnl, results[1].pnl)

    def test_shadow_portfolio_matches_and_does_not_patch_original(self):
        dates, prices, benchmark, members = inputs(105)
        bundles = []
        for implementation in (OriginalCoordinator, ShadowPortfolioBacktester):
            runner = implementation(prices, benchmark, members,
                                    [FractionStrategy("a"), FractionStrategy("b")],
                                    initial_cash=1000, shadow_initial_cash=1000, fee_bps=5)
            if implementation is ShadowPortfolioBacktester:
                self.assertTrue(all(isinstance(e.oms, IndexedOMS)
                                    for e in [runner._main, *runner._shadows.values()]))
            else:
                self.assertIs(type(runner._main.oms), OMS)
            bundles.append(runner.run(dates[5], dates[-1], shadow_start=dates[1]))
        for first, second in [(bundles[0].portfolio, bundles[1].portfolio),
                              *[(bundles[0].shadows[n], bundles[1].shadows[n]) for n in ("a", "b")]]:
            for field in ("history", "fills", "orders", "open_lots", "diagnostics", "refits", "phases"):
                pd.testing.assert_frame_equal(getattr(first, field), getattr(second, field))
        for field in ("allocations", "contributions", "approved_contributions"):
            pd.testing.assert_frame_equal(getattr(bundles[0], field), getattr(bundles[1], field))

    def test_failed_session_rolls_back_index_and_ledger(self):
        dates, prices, benchmark, members = inputs(85)
        engine = Backtester(prices, benchmark, members, FractionStrategy("a"),
                            initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        while engine.next_session < dates[1] + pd.Timedelta(days=90):
            engine.prepare()
            engine.advance()
        engine.prepare()
        before = deepcopy(engine.oms.__dict__)
        original = engine.oms.fill

        def fail_after_sale(order_id, **kwargs):
            side = engine.oms.orders[order_id].side
            original(order_id, **kwargs)
            if side == "SELL":
                raise RuntimeError("injected post-sale failure")

        engine.oms.fill = fail_after_sale
        with self.assertRaisesRegex(RuntimeError, "post-sale"):
            engine.advance()
        del engine.oms.fill
        self.assertEqual(engine.oms.__dict__, before)
