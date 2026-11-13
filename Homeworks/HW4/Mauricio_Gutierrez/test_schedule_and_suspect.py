"""Synthetic checks for the cash schedule and the suspect-quote rule.

Run from the repository root:
python -m unittest discover -s Homeworks/HW4/Mauricio_Gutierrez -p 'test_*.py' -v
Business-day calendars are test fixtures, not exchange calendars.
"""
import unittest

import numpy as np
import pandas as pd

from strategy_backtest import (
    BacktestConfig, Backtester, reconcile_account, remove_suspect_quotes,
)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2010-01-01", "2010-12-31")
        self.symbols = list("ABCDEFG")
        self.prices = pd.MultiIndex.from_product(
            [self.calendar, self.symbols], names=["date", "symbol"]
        ).to_frame(index=False).assign(adj_close=100.0)
        self.forecasts = self.prices[["date", "symbol"]].copy()
        self.forecasts["prediction"] = self.forecasts.symbol.map(dict(zip(self.symbols, range(7, 0, -1))))
        self.sectors = {s: f"sector_{s}" for s in self.symbols}
        quarters = ["2010-01-01", "2010-04-01", "2010-07-01", "2010-10-01"]
        self.membership = pd.DataFrame([
            {"symbol": s, "quarter_start": pd.Timestamp(q), "selection_date": pd.Timestamp(q) - pd.Timedelta(days=1)}
            for q in quarters for s in self.symbols
        ])
        self.cfg = BacktestConfig()

    def run_case(self, prices=None, **options):
        return Backtester(self.cfg, **options).run(
            self.prices if prices is None else prices, self.calendar, self.forecasts,
            self.sectors, self.membership, "2010-01-01", "2010-12-31")


class CashScheduleTests(Fixture):
    def test_decisions_are_at_least_21_sessions_apart(self):
        result = self.run_case(schedule="cash")
        positions = self.calendar.get_indexer(result.decisions.signal_date)
        self.assertGreater(len(positions), 3)
        self.assertEqual(positions[0], 0)  # The first evaluation session always qualifies.
        self.assertTrue((np.diff(positions) >= 21).all())

    def test_no_decision_when_cash_is_below_five_percent(self):
        result = self.run_case(schedule="cash")
        self.assertTrue((result.decisions.cash >= 0.05 * result.decisions.nav - 1e-9).all())
        # After three tranches the account is fully invested; it must wait for a sale.
        third = result.decisions.signal_date.iloc[2]
        fourth = result.decisions.signal_date.iloc[3]
        first_sale = result.trades.loc[result.trades.side.eq("SELL"), "date"].min()
        self.assertGreaterEqual(fourth, first_sale)
        waiting = result.daily.loc[result.daily.date.gt(third) & result.daily.date.lt(first_sale)]
        self.assertTrue((waiting.cash.iloc[1:] < 0.05 * waiting.nav.iloc[1:]).all())

    def test_cash_is_never_spent_twice_and_account_reconciles(self):
        result = self.run_case(schedule="cash")
        self.assertGreaterEqual(result.daily.cash.min(), -1e-7)
        checks = reconcile_account(result, self.calendar, self.cfg)
        self.assertTrue(all(value == "PASS" for value in checks.values()))
        # A sale at an execution close never enlarges an already planned budget.
        self.assertTrue(result.decisions.planned_cash.le(result.decisions.cash + 1e-7).all())

    def test_cash_schedule_holds_less_idle_cash_than_monthly(self):
        monthly = self.run_case(schedule="monthly")
        cash = self.run_case(schedule="cash")
        late = lambda r: (r.daily.cash / r.daily.nav).loc[r.daily.date.ge("2010-06-01")].mean()
        self.assertLess(late(cash), late(monthly))

    def test_monthly_schedule_is_unchanged_by_default(self):
        result = self.run_case()
        first = self.calendar.to_series().groupby(self.calendar.to_period("M")).first()
        self.assertTrue(result.decisions.signal_date.isin(set(first)).all())

    def test_invalid_schedule_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            Backtester(self.cfg, schedule="weekly")
        with self.assertRaises(ValueError):
            Backtester(self.cfg, schedule="cash", min_sessions_between=0)


class SuspectQuoteTests(Fixture):
    def table(self, values):
        return pd.DataFrame({"A": values}, index=self.calendar[:len(values)], dtype=float)

    def test_normal_and_large_real_moves_are_kept(self):
        quotes = self.table([100, 120, 60, 240, 50])  # +20%, -50%, +300%, -79%
        clean, log = remove_suspect_quotes(quotes)
        pd.testing.assert_frame_equal(clean, quotes)
        self.assertTrue(log.empty)

    def test_error_run_is_removed_and_price_recovers(self):
        quotes = self.table([24100, 0.194, 0.192, 0.19, 25000, 25100])  # The AAM pattern.
        clean, log = remove_suspect_quotes(quotes)
        self.assertEqual(clean.A.isna().tolist(), [False, True, True, True, False, False])
        self.assertEqual(len(log), 3)
        self.assertTrue(log.last_accepted_price.eq(24100).all())

    def test_rule_uses_only_past_quotes(self):
        base = [100, 101, 0.01, 102, 103]
        changed_future = [100, 101, 0.01, 9999, 0.5]
        first, _ = remove_suspect_quotes(self.table(base))
        second, _ = remove_suspect_quotes(self.table(changed_future))
        self.assertEqual(first.A.isna().iloc[:3].tolist(), second.A.isna().iloc[:3].tolist())

    def test_gap_is_compared_with_last_accepted_quote(self):
        quotes = self.table([100, np.nan, np.nan, 700, 110])
        clean, log = remove_suspect_quotes(quotes)
        self.assertTrue(np.isnan(clean.A.iloc[3]) and clean.A.iloc[4] == 110)
        self.assertEqual(len(log), 1)

    def test_held_stock_is_marked_at_last_accepted_close(self):
        prices = self.prices.copy()
        bad = prices.symbol.eq("A") & prices.date.between("2010-02-01", "2010-02-05")
        prices.loc[bad, "adj_close"] = 0.001
        result = self.run_case(prices=prices, allow_stale_quotes=True, suspect_ratio=(0.2, 5.0))
        self.assertEqual(len(result.suspect_quotes), 5)
        # NAV never reflects the false collapse: only fees move it while prices are flat.
        np.testing.assert_allclose(result.daily.nav, self.cfg.initial_cash - result.daily.fees.cumsum(), atol=1e-6)
        flagged = result.valuation_flags.loc[result.valuation_flags.symbol.eq("A")]
        self.assertTrue(flagged.mark_price.eq(100.0).all() and len(flagged) >= 5)
        reconcile_account(result, self.calendar, self.cfg)

    def test_suspect_entry_quote_cancels_the_order(self):
        prices = self.prices.copy()
        entry = self.calendar[1]
        prices.loc[prices.symbol.eq("A") & prices.date.eq(entry), "adj_close"] = 100_000.0
        result = self.run_case(prices=prices, allow_stale_quotes=True, suspect_ratio=(0.2, 5.0))
        first = result.orders.loc[result.orders.execution_date.eq(entry)]
        self.assertEqual(first.loc[first.symbol.eq("A"), "status"].item(), "cancelled_missing_entry")
        self.assertFalse(result.trades.loc[result.trades.date.eq(entry), "symbol"].eq("A").any())

    def test_without_the_rule_quotes_are_untouched(self):
        result = self.run_case()
        self.assertTrue(result.suspect_quotes.empty)


if __name__ == "__main__":
    unittest.main()
