"""Hand-computable checks of the backtester (run: python -m unittest discover tests)."""

import unittest

import numpy as np
import pandas as pd

from quantlab.audit import assert_clean, audit_backtest
from quantlab.backtest import (Backtester, HeldPriceGapError, OrderManagementSystem, Strategy,
                               frozen_writeoff_nav)


class Scripted(Strategy):
    """Places pre-scripted budgets keyed by signal date; records what it saw."""

    def __init__(self, script, holding_days=90):
        self.script = {pd.Timestamp(k): v for k, v in script.items()}
        self.holding_days = holding_days
        self.seen = []

    def generate_orders(self, ctx):
        self.seen.append(ctx)
        return pd.Series(self.script.get(ctx.signal_date, {}), dtype=float)


def market(prices: dict, start="2020-01-01", periods=200, drop=()):
    cal = pd.bdate_range(start, periods=periods)
    cal = cal[~cal.isin(pd.to_datetime(list(drop)))]
    syms = pd.Index(sorted(prices))
    px = np.column_stack([np.broadcast_to(np.asarray(prices[s], float), (len(cal),)).copy()
                          if np.ndim(prices[s]) == 0 else np.asarray(prices[s], float)[: len(cal)]
                          for s in syms])
    return cal, syms, px


def run(cal, syms, px, script, start=None, end=None, fee_bps=5, **kw):
    strat = Scripted(script)
    bt = Backtester(cal, syms, px, strat, OrderManagementSystem(1_000_000, fee_bps), **kw)
    res = bt.run(start or cal[1], end or cal[-1])
    return res, strat


class RoundTrip(unittest.TestCase):
    def test_constant_price_round_trip_costs_exactly_two_fees(self):
        cal, syms, px = market({"AAA": 10.0})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}})
        buy, sell = res.fills.iloc[0], res.fills.iloc[1]
        units = 100_000 / (10 * 1.0005)
        self.assertAlmostEqual(buy.units, units, places=9)
        self.assertAlmostEqual(buy.fee, units * 10 * 0.0005, places=9)
        self.assertAlmostEqual(res.history.cash.loc[cal[1]], 900_000, places=6)
        # B(1 - (1-c)/(1+c)) = 100000 * 0.001/1.0005
        expected_cost = 100_000 * 0.001 / 1.0005
        self.assertAlmostEqual(1_000_000 - res.history.nav.iloc[-1], expected_cost, places=6)
        self.assertEqual(sell.date, cal[cal.searchsorted(cal[1] + pd.Timedelta(days=90))])
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_known_return_and_nav_path(self):
        p = np.full(200, 10.0)
        p[2:] = 11.0  # +10% on the session after entry
        cal, syms, px = market({"AAA": p})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, fee_bps=0)
        h = res.history.nav
        self.assertAlmostEqual(h.loc[cal[0]], 1_000_000)   # initial-capital row
        self.assertAlmostEqual(h.loc[cal[1]], 1_000_000)   # bought at 10, marked at 10
        self.assertAlmostEqual(h.loc[cal[2]], 1_010_000)   # 10,000 units x +1
        self.assertAlmostEqual(res.pnl.realized_pnl, 10_000)

    def test_zero_fee_buy_and_hold_matches_price_ratio(self):
        rng = np.random.default_rng(0)
        p = 50 * np.cumprod(1 + rng.normal(0, 0.01, 200))
        cal, syms, px = market({"AAA": p})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 1_000_000}}, fee_bps=0, end=cal[60])
        self.assertAlmostEqual(res.history.nav.iloc[-1] / 1e6, p[60] / p[1], places=10)


class Timing(unittest.TestCase):
    def test_maturity_on_a_holiday_exits_next_session(self):
        cal0 = pd.bdate_range("2020-01-01", periods=200)
        entry = cal0[1]
        due = entry + pd.Timedelta(days=90)
        due = due if due in cal0 else cal0[cal0.searchsorted(due)]
        cal, syms, px = market({"AAA": 10.0}, drop=[due])
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}})
        sell = res.fills.loc[res.fills.side.eq("SELL")].iloc[0]
        self.assertEqual(sell.date, cal[cal.searchsorted(due)])
        self.assertGreater(sell.date, due)

    def test_two_lots_same_ticker_keep_separate_maturities(self):
        cal, syms, px = market({"AAA": 10.0})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}, cal[10]: {"AAA": 50_000}})
        sells = res.fills.loc[res.fills.side.eq("SELL")].sort_values("date")
        self.assertEqual(len(sells), 2)
        self.assertEqual(sells.iloc[0].date, cal[cal.searchsorted(cal[1] + pd.Timedelta(days=90))])
        self.assertEqual(sells.iloc[1].date, cal[cal.searchsorted(cal[11] + pd.Timedelta(days=90))])
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_strategy_sees_only_prices_through_signal_close(self):
        p = np.arange(1, 201, dtype=float)
        cal, syms, px = market({"AAA": p})
        _, strat = run(cal, syms, px, {}, end=cal[30])
        for ctx in strat.seen:
            self.assertEqual(ctx.view.asof, ctx.signal_date)
            self.assertEqual(ctx.view.prices[-1, 0], p[cal.get_loc(ctx.signal_date)])
            self.assertLess(ctx.signal_date, ctx.execution_date)
            with self.assertRaises(ValueError):
                ctx.view.prices[-1, 0] = 0.0  # read-only

    def test_future_prices_cannot_change_past_decisions(self):
        """Look-ahead invariance: perturb prices after day k; fills up to k must not change."""
        class Momentum(Strategy):
            holding_days = 90

            def generate_orders(self, ctx):
                w = ctx.view.prices
                if len(w) < 6 or ctx.portfolio.cash < 1:
                    return pd.Series(dtype=float)
                r = w[-1] / w[-6] - 1
                best = ctx.view.symbols[int(np.nanargmax(r))]
                return pd.Series({best: ctx.portfolio.cash * 0.1})
        rng = np.random.default_rng(1)
        base = 20 * np.cumprod(1 + rng.normal(0, 0.02, (200, 3)), axis=0)
        cal = pd.bdate_range("2020-01-01", periods=200)
        syms = pd.Index(["A", "B", "C"])
        k = 80
        alt = base.copy()
        alt[k + 1:] *= rng.uniform(0.5, 1.5, alt[k + 1:].shape)
        runs = []
        for prices in (base, alt):
            bt = Backtester(cal, syms, prices, Momentum(), OrderManagementSystem())
            runs.append(bt.run(cal[10], cal[-1]))
        a = runs[0].fills.loc[runs[0].fills.date <= cal[k]].reset_index(drop=True)
        b = runs[1].fills.loc[runs[1].fills.date <= cal[k]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertGreater(len(a), 0)

    def test_sale_cash_is_not_available_to_orders_planned_before_it(self):
        cal, syms, px = market({"AAA": 10.0, "BBB": 10.0})
        entry = cal[1]
        sale_day = cal[cal.searchsorted(entry + pd.Timedelta(days=90))]
        sale_signal = cal[cal.get_loc(sale_day) - 1]
        # all cash invested; an order planned at the close before the sale must be rejected
        with self.assertRaisesRegex(ValueError, "exceed signal-close cash"):
            run(cal, syms, px, {cal[0]: {"AAA": 1_000_000}, sale_signal: {"BBB": 500_000}})
        # planned at the sale-day close (proceeds known) it is fine
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 1_000_000}, sale_day: {"BBB": 500_000}})
        b = res.fills.loc[res.fills.symbol.eq("BBB")].iloc[0]
        self.assertEqual(b.date, cal[cal.get_loc(sale_day) + 1])


class CashAndOrders(unittest.TestCase):
    def test_orders_above_signal_cash_raise(self):
        cal, syms, px = market({"AAA": 10.0})
        with self.assertRaisesRegex(ValueError, "exceed signal-close cash"):
            run(cal, syms, px, {cal[0]: {"AAA": 1_000_001}})

    def test_negative_or_nan_budget_rejected(self):
        cal, syms, px = market({"AAA": 10.0})
        for bad in (-1.0, np.nan):
            with self.assertRaises(ValueError):
                run(cal, syms, px, {cal[0]: {"AAA": bad}})

    def test_no_orders_keeps_cash(self):
        cal, syms, px = market({"AAA": 10.0})
        res, _ = run(cal, syms, px, {})
        self.assertTrue((res.history.nav == 1_000_000).all())
        self.assertEqual(len(res.fills), 0)

    def test_missing_entry_quote_rejects_order_and_keeps_cash(self):
        p = np.full(200, 10.0)
        p[1] = np.nan
        cal, syms, px = market({"AAA": p})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}})
        self.assertEqual(len(res.fills), 0)
        self.assertEqual(res.events.event.tolist(), ["rejected_no_quote"])
        self.assertTrue((res.history.cash == 1_000_000).all())

    def test_oms_refuses_sale_before_maturity(self):
        oms = OrderManagementSystem()
        d = pd.Timestamp("2020-01-02")
        lid = oms.buy(10, 10.0, symbol="AAA", col=0, date=d, signal_date=d - pd.Timedelta(days=1), holding_days=90)
        with self.assertRaises(ValueError):
            oms.sell_lot(lid, 10.0, date=d + pd.Timedelta(days=89))

    def test_buy_requires_signal_before_fill(self):
        oms = OrderManagementSystem()
        d = pd.Timestamp("2020-01-02")
        with self.assertRaises(ValueError):
            oms.buy(10, 10.0, symbol="AAA", col=0, date=d, signal_date=d, holding_days=90)


class MissingQuotes(unittest.TestCase):
    def gap_market(self, gap_start, gap_len):
        p = np.full(200, 10.0)
        p[gap_start:gap_start + gap_len] = np.nan
        return market({"AAA": p})

    def test_short_gap_carries_last_mark(self):
        cal, syms, px = self.gap_market(20, 3)
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, fee_bps=0)
        self.assertAlmostEqual(res.history.nav.loc[cal[21]], 1_000_000)
        self.assertEqual((res.events.event == "stale_mark").sum(), 3)
        self.assertFalse((res.events.event == "frozen_at_last_mark").any())
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_long_gap_freezes_then_resumes_and_exits_at_first_quote(self):
        entry_due = pd.bdate_range("2020-01-01", periods=200)[1] + pd.Timedelta(days=90)
        cal0 = pd.bdate_range("2020-01-01", periods=200)
        due_i = cal0.searchsorted(entry_due)
        cal, syms, px = self.gap_market(due_i - 5, 12)   # gap spans maturity
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, fee_bps=0)
        ev = res.events.event.tolist()
        self.assertIn("frozen_at_last_mark", ev)
        self.assertIn("quotes_resumed", ev)
        sell = res.fills.loc[res.fills.side.eq("SELL")].iloc[0]
        self.assertEqual(sell.date, cal[due_i - 5 + 12])  # first quote after the gap
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_quotes_never_return_lot_stays_at_last_mark_and_lower_bound_writes_off(self):
        p = np.full(200, 10.0)
        p[30:] = np.nan
        cal, syms, px = market({"AAA": p})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, fee_bps=0)
        self.assertAlmostEqual(res.history.nav.iloc[-1], 1_000_000)
        self.assertTrue(res.open_lots.frozen.iloc[0])
        self.assertAlmostEqual(frozen_writeoff_nav(res).iloc[-1], 900_000)
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_quotes_returning_after_long_gap_do_not_revive_the_lot(self):
        p = np.full(400, 10.0)
        p[30:250] = np.nan          # ~220 sessions without quotes (buyout) ...
        p[250:] = 55.0              # ... then a different security reuses the ticker
        cal, syms, px = market({"AAA": p}, periods=400)
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, fee_bps=0)
        self.assertEqual((res.fills.side == "SELL").sum(), 0)
        self.assertIn("terminal_freeze_after_long_gap", res.events.event.tolist())
        self.assertAlmostEqual(res.history.nav.iloc[-1], 1_000_000)   # still at the 10.0 mark, not 55
        self.assertTrue(res.open_lots.terminal.iloc[0])
        self.assertAlmostEqual(frozen_writeoff_nav(res).iloc[-1], 900_000)
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_settle_at_maturity_recycles_cash_at_last_mark(self):
        p = np.full(200, 10.0)
        p[30] = 12.0
        p[31:] = np.nan                      # acquired: quotes end at 12.0
        cal, syms, px = market({"AAA": p})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, fee_bps=0, on_long_gap="settle_at_maturity")
        sell = res.fills.loc[res.fills.side.eq("SELL")].iloc[0]
        self.assertEqual(sell.settlement, "stale_mark")
        self.assertEqual(sell.price, 12.0)
        self.assertEqual(sell.date, cal[cal.searchsorted(cal[1] + pd.Timedelta(days=90))])
        self.assertAlmostEqual(res.history.cash.iloc[-1], 900_000 + 10_000 * 12.0)
        self.assertEqual(len(res.open_lots), 0)
        assert_clean(audit_backtest(res, cal, syms, px))

    def test_halt_policy_raises(self):
        p = np.full(200, 10.0)
        p[30:] = np.nan
        cal, syms, px = market({"AAA": p})
        with self.assertRaises(HeldPriceGapError):
            run(cal, syms, px, {cal[0]: {"AAA": 100_000}}, on_long_gap="halt")

    def test_end_of_test_open_lot_valued_not_sold(self):
        p = np.linspace(10, 12, 200)
        cal, syms, px = market({"AAA": p})
        res, _ = run(cal, syms, px, {cal[100]: {"AAA": 100_000}}, end=cal[150], fee_bps=0)
        self.assertEqual((res.fills.side == "SELL").sum(), 0)
        self.assertEqual(len(res.open_lots), 1)
        self.assertAlmostEqual(res.history.holdings_value.iloc[-1], res.open_lots.units.iloc[0] * p[150])


class AuditDetectsErrors(unittest.TestCase):
    def test_audit_flags_a_tampered_ledger(self):
        cal, syms, px = market({"AAA": 10.0})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}})
        res.fills.loc[0, "cash_change"] += 1.0
        report = audit_backtest(res, cal, syms, px)
        self.assertFalse(report.set_index("check").loc["cash = initial + cumulative fill cash changes", "passed"])

    def test_audit_flags_a_wrong_fill_price(self):
        cal, syms, px = market({"AAA": 10.0})
        res, _ = run(cal, syms, px, {cal[0]: {"AAA": 100_000}})
        res.fills.loc[0, "price"] = 9.0
        report = audit_backtest(res, cal, syms, px)
        self.assertFalse(report.set_index("check").loc["buy price = that session's close", "passed"])


if __name__ == "__main__":
    unittest.main()
