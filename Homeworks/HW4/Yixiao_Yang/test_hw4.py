"""Hand-calculated accounting and synthetic information-timing checks.

These 56 tests check implementation correctness, not market performance.
Run: python -m unittest test_hw4 -v
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backtester import (
    DataQualityError, benchmark_history, performance_metrics, run_backtest,
)
from data import (
    FEATURE_COLUMNS, FINANCIAL_FEATURES, FINANCIAL_INPUTS, PRICE_FEATURES,
    apply_dated_membership, join_financial_events, make_financial_events,
    make_holding_labels, make_price_features, monthly_schedule, prepare_data,
)
from hw4 import (
    generate_walk_forward_scores, make_model, mature_training_rows,
    select_alpha, run_experiment,
)


# Portfolio accounting


def score_frame(calendar, entries, symbols=("A",), scores=None):
    rows = []
    for entry in entries:
        entry = pd.Timestamp(entry)
        signal = calendar[calendar.get_loc(entry) - 1]
        for i, symbol in enumerate(symbols):
            rows.append({
                "signal_date": signal, "entry_date": entry, "symbol": symbol,
                "score": (scores[i] if scores is not None else 1.0 - i / 10),
            })
    return pd.DataFrame(rows)

class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2020-01-01", "2020-08-31")
        self.prices = pd.DataFrame({"A": 10.0, "B": 20.0}, index=self.calendar)

    def run_case(self, scores, end="2020-05-01", **kwargs):
        settings = dict(initial_cash=1000.0, top_n=1, cohort_fraction=1, fee_bps=0)
        settings.update(kwargs)
        return run_backtest(self.prices, scores, "2020-01-02", end, **settings)

    def test_hand_calculated_fees_and_exit(self):
        # $1,000 / 1.01 / $10 = 99.00990099 units. Sell at $12, minus 1%.
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-04-01":, "A"] = 12.0
        result = self.run_case(scores, fee_bps=100)
        units = 1000 / (10 * 1.01)
        expected_cash = units * 12 * 0.99
        self.assertAlmostEqual(result.trades.iloc[0].units, units)
        self.assertAlmostEqual(result.history.loc["2020-01-02", "nav"], 1000 / 1.01)
        self.assertAlmostEqual(result.history.iloc[-1].cash, expected_cash)
        self.assertEqual(result.trades.iloc[1].date, pd.Timestamp("2020-04-01"))
        self.assertEqual(result.history.iloc[-1].n_open_lots, 0)
        self.assertAlmostEqual(result.diagnostics["pnl_reconciliation_error"], 0)

    def test_weekend_maturity_next_market_session(self):
        # Jan 6 + 90 calendar days = Sunday Apr 5; sell Monday Apr 6.
        scores = score_frame(self.calendar, ["2020-01-06"])
        result = self.run_case(scores)
        sale = result.trades.loc[result.trades.side.eq("SELL")].iloc[0]
        self.assertEqual(sale.maturity_date, pd.Timestamp("2020-04-05"))
        self.assertEqual(sale.date, pd.Timestamp("2020-04-06"))

    def test_fixed_budget_not_changed_by_execution_price(self):
        scores = score_frame(self.calendar, ["2020-01-02"], ("A", "B"))
        normal = self.run_case(scores, end="2020-01-03", top_n=2, cohort_fraction=0.5)
        self.prices.loc["2020-01-02":, "A"] = 100.0
        changed = self.run_case(scores, end="2020-01-03", top_n=2, cohort_fraction=0.5)
        pd.testing.assert_frame_equal(normal.orders, changed.orders)
        self.assertAlmostEqual(changed.trades.iloc[0].units, normal.trades.iloc[0].units / 10)
        self.assertAlmostEqual(changed.history.iloc[-1].cash, 500)

    def test_today_sale_proceeds_do_not_fund_yesterday_orders(self):
        # First cohort spends all cash; no April 1 buy was possible March 31.
        scores = score_frame(self.calendar, ["2020-01-02", "2020-04-01"])
        result = self.run_case(scores)
        self.assertEqual((result.trades.side == "BUY").sum(), 1)
        self.assertEqual((result.trades.side == "SELL").sum(), 1)
        self.assertAlmostEqual(result.history.loc["2020-04-01", "cash"], 1000)

    def test_multiple_lots_keep_distinct_maturities(self):
        scores = score_frame(self.calendar, ["2020-01-02", "2020-02-03"])
        result = self.run_case(scores, end="2020-04-15", cohort_fraction=0.5)
        self.assertEqual((result.trades.side == "BUY").sum(), 2)
        self.assertEqual((result.trades.side == "SELL").sum(), 1)
        self.assertEqual(len(result.open_positions), 1)
        self.assertEqual(result.open_positions.iloc[0].entry_date, pd.Timestamp("2020-02-03"))
        self.assertAlmostEqual(result.history.iloc[-1].nav, 1000)

    def test_missing_entry_cancelled_without_reranking(self):
        scores = score_frame(self.calendar, ["2020-01-02"], ("A", "B"))
        self.prices.loc["2020-01-02", "A"] = np.nan
        result = self.run_case(scores, end="2020-01-03")
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.orders.iloc[0].symbol, "A")
        self.assertEqual(result.orders.iloc[0].status, "cancelled_missing_quote")
        self.assertAlmostEqual(result.history.iloc[-1].cash, 1000)

    def test_missing_maturity_quote_stops_at_valid_prefix(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-04-01", "A"] = np.nan
        with self.assertRaises(DataQualityError) as raised:
            self.run_case(scores)
        error = raised.exception
        self.assertEqual(error.date, pd.Timestamp("2020-04-01"))
        self.assertEqual(error.partial_result.history.index[-1], pd.Timestamp("2020-03-31"))
        self.assertEqual((error.partial_result.trades.side == "SELL").sum(), 0)
        self.assertFalse(error.partial_result.diagnostics["complete"])

    def test_stale_marks_then_stop_on_sixth_session(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        missing = pd.bdate_range("2020-01-03", periods=6)
        self.prices.loc[missing, "A"] = np.nan
        with self.assertRaises(DataQualityError) as raised:
            self.run_case(scores, end="2020-01-20")
        result = raised.exception.partial_result
        self.assertEqual(raised.exception.date, missing[-1])
        self.assertEqual(result.history.iloc[-1].stale_lots, 1)
        self.assertEqual(result.open_positions.iloc[0].stale_sessions, 5)
        self.assertAlmostEqual(result.history.iloc[-1].nav, 1000)

    def test_terminal_open_positions_and_after_end_prices(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-01":, "A"] = 11.0
        result = self.run_case(scores, end="2020-02-28")
        self.prices.loc["2020-03-02":, "A"] = 1e6
        changed = self.run_case(scores, end="2020-02-28")
        pd.testing.assert_frame_equal(result.history, changed.history)
        self.assertEqual(len(result.open_positions), 1)
        self.assertAlmostEqual(result.history.iloc[-1].nav, 1100)
        self.assertEqual((result.trades.side == "SELL").sum(), 0)

    def test_fewer_candidates_leave_unused_slots_in_cash(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        result = self.run_case(scores, end="2020-01-03", top_n=2)
        self.assertAlmostEqual(result.orders.iloc[0].budget, 500)
        self.assertAlmostEqual(result.history.iloc[-1].cash, 500)

    def test_same_day_signal_rejected(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        scores["signal_date"] = scores.entry_date
        with self.assertRaises(ValueError):
            self.run_case(scores)

    def test_independent_cash_reconstruction(self):
        scores = score_frame(self.calendar, ["2020-01-02", "2020-02-03", "2020-03-02"])
        result = self.run_case(scores, fee_bps=10, cohort_fraction=1 / 3)
        changes = result.trades.groupby("date").cash_change.sum().reindex(result.history.index, fill_value=0)
        np.testing.assert_allclose(1000 + changes.cumsum(), result.history.cash, atol=1e-9)
        np.testing.assert_allclose(result.history.cash + result.history.holdings_value, result.history.nav, atol=1e-9)

class PerformanceTests(unittest.TestCase):
    def test_entry_fee_is_first_return_and_drawdown(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        nav = pd.Series([1000, 990, 990], index=dates)
        result = performance_metrics(nav, 1000, "2020-01-02", "2020-01-03")
        expected_returns = np.array([-0.01, 0.0])
        rf = 1.02 ** (1 / 252) - 1
        expected_sharpe = np.sqrt(252) * (expected_returns - rf).mean() / expected_returns.std(ddof=1)
        self.assertAlmostEqual(result["max_drawdown"], -0.01)
        self.assertAlmostEqual(result["sharpe"], expected_sharpe)

    def test_benchmark_invests_at_first_close_not_previous_close(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        raw = pd.Series([50, 100, 110], index=dates)
        nav = benchmark_history(raw, "2020-01-02", "2020-01-03", 1000)
        np.testing.assert_allclose(nav, [1000, 1000, 1100])

class CashMergerTests(unittest.TestCase):
    """Cash actions are dated legal entitlements, not fabricated market fills."""

    def setUp(self):
        self.calendar = pd.bdate_range("2020-01-01", "2020-08-31")
        self.prices = pd.DataFrame({"A": 10.0, "B": 20.0}, index=self.calendar)
        self.action = {
            "action_id": "A_CASH_MERGER_20200203", "symbol": "A",
            "effective_date": "2020-02-03", "cash_per_adjusted_unit": 12.0,
            "source_url": "https://example.org/verified-merger-document",
        }

    def run_case(self, scores, end="2020-05-06", **kwargs):
        settings = dict(initial_cash=1000.0, top_n=1, cohort_fraction=1,
                        fee_bps=0, cash_actions=[self.action])
        settings.update(kwargs)
        return run_backtest(self.prices, scores, "2020-01-02", end, **settings)

    def test_conversion_is_ring_fenced_and_release_charges_one_exit_fee(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03":, "A"] = np.nan
        result = self.run_case(scores, fee_bps=100)
        units = 1000 / (10 * 1.01)
        claim = units * 12
        self.assertEqual(result.trades.side.tolist(),
                         ["BUY", "CASH_MERGER_CONVERSION", "CASH_CLAIM_RELEASE"])
        conversion = result.trades.iloc[1]
        release = result.trades.iloc[2]
        self.assertEqual(conversion.date, pd.Timestamp("2020-02-03"))
        self.assertEqual(conversion.cash_change, 0)
        self.assertEqual(conversion.fee, 0)
        self.assertEqual(conversion.realized_pnl, 0)
        self.assertAlmostEqual(conversion.gross_value, claim)
        self.assertEqual(release.date, pd.Timestamp("2020-04-01"))
        self.assertAlmostEqual(release.fee, claim * 0.01)
        self.assertAlmostEqual(release.realized_pnl, claim * 0.99 - 1000)
        self.assertAlmostEqual(result.history.loc["2020-02-03", "stock_value"], 0)
        self.assertAlmostEqual(result.history.loc["2020-03-31", "cash"], 0)
        self.assertAlmostEqual(result.history.loc["2020-03-31", "cash_claim_value"], claim)
        self.assertAlmostEqual(result.history.loc["2020-04-01", "cash"], claim * 0.99)
        self.assertAlmostEqual(result.history.loc["2020-04-01", "cash_claim_value"], 0)
        self.assertAlmostEqual(result.diagnostics["pnl_reconciliation_error"], 0)
        self.assertAlmostEqual(result.diagnostics["gross_trading_volume"], units * 10)
        self.assertEqual(result.diagnostics["stale_lot_session_marks"], 0)

    def test_claim_and_same_day_release_cannot_fund_prior_close_budgets(self):
        scores = pd.concat([
            score_frame(self.calendar, ["2020-01-02"]),
            score_frame(self.calendar, ["2020-03-02", "2020-04-01", "2020-04-02"], ("B",)),
        ], ignore_index=True)
        self.prices.loc["2020-02-03":, "A"] = np.nan
        result = self.run_case(scores, end="2020-04-03")
        buy_dates = result.trades.loc[result.trades.side.eq("BUY"), "date"].tolist()
        self.assertEqual(buy_dates, [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-04-02")])
        second = result.orders.loc[result.orders.symbol.eq("B")].iloc[0]
        self.assertEqual(second.signal_date, pd.Timestamp("2020-04-01"))
        self.assertAlmostEqual(second.budget, 1200)

    def test_separate_claim_lots_keep_their_own_maturity(self):
        self.action["effective_date"] = "2020-02-10"
        scores = score_frame(self.calendar, ["2020-01-02", "2020-02-03"])
        self.prices.loc["2020-02-10":, "A"] = np.nan
        result = self.run_case(scores, end="2020-04-15", cohort_fraction=0.5)
        converted = result.trades.loc[result.trades.side.eq("CASH_MERGER_CONVERSION")]
        released = result.trades.loc[result.trades.side.eq("CASH_CLAIM_RELEASE")]
        self.assertEqual(len(converted), 2)
        self.assertEqual(len(released), 1)
        self.assertEqual(result.open_positions.iloc[0].position_type, "cash_claim")
        self.assertEqual(result.open_positions.iloc[0].maturity_date, pd.Timestamp("2020-05-03"))
        self.assertTrue(pd.isna(result.open_positions.iloc[0].scheduled_exit))
        self.assertAlmostEqual(result.history.iloc[-1].cash, 600)
        self.assertAlmostEqual(result.history.iloc[-1].cash_claim_value, 600)
        self.assertAlmostEqual(result.history.iloc[-1].nav, 1200)
        self.assertAlmostEqual(result.diagnostics["pnl_reconciliation_error"], 0)

        extended = self.run_case(scores, cohort_fraction=0.5)
        releases = extended.trades.loc[extended.trades.side.eq("CASH_CLAIM_RELEASE")]
        self.assertEqual(releases.date.tolist(), [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-04")])

    def test_open_claim_marks_ignore_later_quotes_and_future_release(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03":, "A"] = 999.0
        result = self.run_case(scores, end="2020-02-28")
        self.assertAlmostEqual(result.history.iloc[-1].nav, 1200)
        self.assertAlmostEqual(result.history.iloc[-1].cash, 0)
        self.assertEqual(len(result.open_positions), 1)
        self.assertEqual(result.open_positions.iloc[0].position_type, "cash_claim")
        self.assertTrue(pd.isna(result.open_positions.iloc[0].scheduled_exit))
        self.assertEqual(result.diagnostics["cash_claim_release_count"], 0)
        self.assertEqual(result.diagnostics["terminal_cash_claim_lots"], 1)

    def test_action_after_cutoff_does_not_change_earlier_path(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        actual = self.run_case(scores, end="2020-01-31")
        original = self.run_case(scores, end="2020-01-31", cash_actions=None)
        pd.testing.assert_frame_equal(actual.history, original.history)
        pd.testing.assert_frame_equal(actual.orders, original.orders)
        pd.testing.assert_frame_equal(actual.trades, original.trades)

    def test_future_action_does_not_rescue_a_gap_before_effective_date(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-01-03":, "A"] = np.nan
        with self.assertRaises(DataQualityError) as raised:
            self.run_case(scores)
        self.assertEqual(raised.exception.date, pd.Timestamp("2020-01-10"))
        self.assertEqual(raised.exception.partial_result.diagnostics["cash_merger_conversion_count"], 0)

    def test_conversion_on_maturity_precedes_missing_quote_failure(self):
        self.action["effective_date"] = "2020-04-01"
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-04-01":, "A"] = np.nan
        result = self.run_case(scores)
        events = result.trades.loc[result.trades.date.eq("2020-04-01")]
        self.assertEqual(events.side.tolist(), ["CASH_MERGER_CONVERSION", "CASH_CLAIM_RELEASE"])
        self.assertAlmostEqual(result.history.iloc[-1].cash, 1200)

    def test_nonmarket_effective_date_observed_on_next_session(self):
        self.action["effective_date"] = "2020-02-01"  # Saturday
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03":, "A"] = np.nan
        result = self.run_case(scores)
        event = result.trades.loc[result.trades.side.eq("CASH_MERGER_CONVERSION")].iloc[0]
        self.assertEqual(event.date, pd.Timestamp("2020-02-03"))
        self.assertEqual(event.action_effective_date, pd.Timestamp("2020-02-01"))

    def test_action_is_not_reapplied_to_later_reused_ticker(self):
        scores = score_frame(self.calendar, ["2020-01-02", "2020-04-02"])
        self.prices.loc["2020-02-03":"2020-03-31", "A"] = np.nan
        result = self.run_case(scores, end="2020-04-03")
        self.assertEqual(result.diagnostics["cash_merger_conversion_count"], 1)
        self.assertEqual(result.open_positions.iloc[0].position_type, "stock")
        self.assertEqual(result.open_positions.iloc[0].entry_date, pd.Timestamp("2020-04-02"))

    def test_preconverted_adjusted_units_are_not_converted_twice(self):
        # Caller verifies raw=20, adjusted=10: each adjusted-price unit carries
        # 0.5 actual shares. $24/actual share -> $12/adjusted-price unit.
        self.action["cash_per_adjusted_unit"] = 24 * (10 / 20)
        scores = score_frame(self.calendar, ["2020-01-02"])
        result = self.run_case(scores, end="2020-02-28")
        self.assertAlmostEqual(result.history.iloc[-1].cash_claim_value, 100 * 12)

    def test_invalid_or_duplicate_actions_rejected(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        invalid = [
            [self.action, self.action],
            [{**self.action, "cash_per_adjusted_unit": np.inf}],
            [{**self.action, "cash_per_adjusted_unit": -1}],
            [{**self.action, "source_url": ""}],
            [{**self.action, "effective_date": "2020-02-03T12:00:00"}],
        ]
        for actions in invalid:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.run_case(scores, cash_actions=actions)

    def test_pending_buy_on_conversion_session_is_cancelled(self):
        scores = score_frame(self.calendar, ["2020-02-03"])
        result = self.run_case(scores)
        self.assertEqual(result.orders.iloc[0].status, "cancelled_cash_action")
        self.assertTrue(result.trades.empty)
        self.assertAlmostEqual(result.history.iloc[-1].cash, 1000)

    def test_explicit_sourced_extinguishment_can_have_zero_recovery(self):
        self.action["cash_per_adjusted_unit"] = 0.0
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03":, "A"] = np.nan
        result = self.run_case(scores)
        self.assertAlmostEqual(result.history.iloc[-1].nav, 0)
        self.assertEqual(result.diagnostics["cash_merger_conversion_count"], 1)
        self.assertEqual(result.diagnostics["cash_claim_release_count"], 1)
        self.assertAlmostEqual(result.diagnostics["realized_pnl"], -1000)
        self.assertAlmostEqual(result.diagnostics["pnl_reconciliation_error"], 0)
        metrics = performance_metrics(result.history.nav, 1000, "2020-01-02", "2020-05-06")
        self.assertAlmostEqual(metrics["cagr"], -1)
        self.assertAlmostEqual(metrics["max_drawdown"], -1)
        self.assertTrue(np.isfinite(metrics["sharpe"]))

class SecurityActionTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2020-01-01", "2020-08-31")
        self.prices = pd.DataFrame({"A": 10.0, "B": 20.0, "C": 10.0}, index=self.calendar)
        self.action = {
            "action_id": "A_TO_B_20200203", "symbol": "A",
            "effective_date": "2020-02-03", "successor_symbol": "B",
            "units_per_adjusted_unit": 0.5,
            "source_url": "https://example.org/verified-share-merger-document",
        }

    def run_case(self, scores, end="2020-05-06", **kwargs):
        settings = dict(initial_cash=1000.0, top_n=1, cohort_fraction=1,
                        fee_bps=0, security_actions=[self.action])
        settings.update(kwargs)
        return run_backtest(self.prices, scores, "2020-01-02", end, **settings)

    def test_share_merger_preserves_cost_maturity_and_has_no_conversion_fee(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03":, "A"] = np.nan
        self.prices.loc["2020-04-01":, "B"] = 24.0
        result = self.run_case(scores, fee_bps=100)
        old_units = 1000 / (10 * 1.01)
        new_units = old_units * 0.5
        self.assertEqual(result.trades.side.tolist(), ["BUY", "SECURITY_CONVERSION", "SELL"])
        conversion, sale = result.trades.iloc[1], result.trades.iloc[2]
        self.assertEqual(conversion.original_symbol, "A")
        self.assertEqual(conversion.from_symbol, "A")
        self.assertEqual(conversion.symbol, "B")
        self.assertAlmostEqual(conversion.from_units, old_units)
        self.assertAlmostEqual(conversion.units, new_units)
        self.assertEqual(conversion.fee, 0)
        self.assertEqual(conversion.cash_change, 0)
        self.assertEqual(conversion.realized_pnl, 0)
        self.assertEqual(sale.date, pd.Timestamp("2020-04-01"))
        self.assertEqual(sale.symbol, "B")
        self.assertAlmostEqual(sale.units, new_units)
        self.assertAlmostEqual(result.history.iloc[-1].cash, new_units * 24 * 0.99)
        self.assertAlmostEqual(result.diagnostics["pnl_reconciliation_error"], 0)

    def test_terminal_successor_lot_ignores_old_symbol_prices(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03":, "A"] = 100000
        self.prices.loc["2020-02-03":, "B"] = 22
        result = self.run_case(scores, end="2020-02-28")
        position = result.open_positions.iloc[0]
        self.assertEqual(position.original_symbol, "A")
        self.assertEqual(position.symbol, "B")
        self.assertAlmostEqual(position.entry_units, 100)
        self.assertAlmostEqual(position.units, 50)
        self.assertAlmostEqual(position.entry_budget, 1000)
        self.assertEqual(position.maturity_date, pd.Timestamp("2020-04-01"))
        self.assertAlmostEqual(result.history.iloc[-1].nav, 1100)

    def test_successor_quote_required_on_effective_session(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-02-03", "B"] = np.nan
        with self.assertRaises(DataQualityError) as raised:
            self.run_case(scores)
        error = raised.exception
        self.assertEqual(error.date, pd.Timestamp("2020-02-03"))
        self.assertIn("successor", error.cause)
        self.assertEqual(error.partial_result.history.index[-1], pd.Timestamp("2020-01-31"))
        self.assertEqual(error.partial_result.diagnostics["security_conversion_count"], 0)

    def test_separate_successor_lots_keep_original_maturities(self):
        self.action["effective_date"] = "2020-02-10"
        scores = score_frame(self.calendar, ["2020-01-02", "2020-02-03"])
        self.prices.loc["2020-02-10":, "A"] = np.nan
        result = self.run_case(scores, end="2020-04-15", cohort_fraction=0.5)
        self.assertEqual(result.diagnostics["security_conversion_count"], 2)
        self.assertEqual(result.diagnostics["sell_count"], 1)
        self.assertEqual(result.open_positions.iloc[0].entry_date, pd.Timestamp("2020-02-03"))
        self.assertEqual(result.open_positions.iloc[0].maturity_date, pd.Timestamp("2020-05-03"))
        self.assertAlmostEqual(result.history.iloc[-1].cash, 500)
        self.assertAlmostEqual(result.history.iloc[-1].stock_value, 500)

    def test_future_security_action_does_not_change_earlier_path(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        actual = self.run_case(scores, end="2020-01-31")
        original = self.run_case(scores, end="2020-01-31", security_actions=None)
        pd.testing.assert_frame_equal(actual.history, original.history)
        pd.testing.assert_frame_equal(actual.trades, original.trades)
        pd.testing.assert_frame_equal(actual.orders, original.orders)

    def test_conversion_at_maturity_uses_successor_quote_for_sale(self):
        self.action["effective_date"] = "2020-04-01"
        scores = score_frame(self.calendar, ["2020-01-02"])
        self.prices.loc["2020-04-01":, "A"] = np.nan
        result = self.run_case(scores)
        self.assertEqual(result.trades.side.tolist(), ["BUY", "SECURITY_CONVERSION", "SELL"])
        self.assertEqual(result.trades.iloc[-1].date, pd.Timestamp("2020-04-01"))
        self.assertEqual(result.trades.iloc[-1].symbol, "B")
        self.assertAlmostEqual(result.history.iloc[-1].cash, 1000)

    def test_successive_security_and_cash_events_preserve_original_lineage(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        second = {
            **self.action, "action_id": "B_TO_C_20200302", "symbol": "B",
            "successor_symbol": "C", "effective_date": "2020-03-02",
            "units_per_adjusted_unit": 2.0,
        }
        cash = {
            "action_id": "C_CASH_20200320", "symbol": "C", "effective_date": "2020-03-20",
            "cash_per_adjusted_unit": 12.0, "source_url": "https://example.org/verified-cash-merger",
        }
        self.prices.loc["2020-02-03":, "A"] = np.nan
        self.prices.loc["2020-03-02":, "B"] = np.nan
        self.prices.loc["2020-03-20":, "C"] = np.nan
        result = self.run_case(scores, security_actions=[self.action, second], cash_actions=[cash])
        self.assertEqual(result.trades.side.tolist(), ["BUY", "SECURITY_CONVERSION", "SECURITY_CONVERSION",
                                                      "CASH_MERGER_CONVERSION", "CASH_CLAIM_RELEASE"])
        self.assertTrue(result.trades.original_symbol.eq("A").all())
        self.assertEqual(result.trades.iloc[-1].date, pd.Timestamp("2020-04-01"))
        self.assertAlmostEqual(result.history.iloc[-1].cash, 1200)
        self.assertAlmostEqual(result.diagnostics["pnl_reconciliation_error"], 0)

    def test_invalid_ratios_and_conflicting_actions_rejected(self):
        scores = score_frame(self.calendar, ["2020-01-02"])
        for ratio in [0, -1, np.inf]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.run_case(scores, security_actions=[{**self.action, "units_per_adjusted_unit": ratio}])
        cash = {
            "action_id": "A_CASH", "symbol": "A", "effective_date": "2020-02-03",
            "cash_per_adjusted_unit": 12, "source_url": "https://example.org/verified-merger",
        }
        with self.assertRaises(ValueError):
            self.run_case(scores, cash_actions=[cash])

# Data availability and feature timing


def financial_row(fiscal, filing, revenue=100.0, net_income=10.0, **extra):
    row = {
        "symbol": "AAA", "fiscal_date": fiscal, "filing_date": filing,
        "period": "quarter", "reported_currency": "USD", "revenue": revenue,
        "net_income": net_income, "operating_income": 12.0, "total_assets": 200.0,
        "total_debt": 50.0, "cash_and_equivalents": 25.0, "free_cash_flow": 8.0,
        "operating_cash_flow": 11.0,
    }
    row.update(extra)
    return row

class DataTimingTests(unittest.TestCase):
    def test_monthly_entry_uses_previous_session_and_excludes_buffer(self):
        calendar = pd.bdate_range("2009-12-01", "2026-09-30")
        schedule = monthly_schedule(calendar)
        first = schedule.iloc[0]
        self.assertEqual(first.entry_date, pd.Timestamp("2010-01-01"))
        self.assertEqual(first.signal_date, pd.Timestamp("2009-12-31"))
        self.assertFalse(schedule.entry_date.between("2016-01-01", "2016-03-31").any())
        april = schedule.loc[schedule.entry_date.eq("2016-04-01")].iloc[0]
        self.assertEqual(april.signal_date, pd.Timestamp("2016-03-31"))
        self.assertEqual(len(schedule), 72 + 126)

    def test_price_features_do_not_change_when_future_prices_change(self):
        calendar = pd.bdate_range("2008-01-01", periods=400)
        prices = pd.Series(100.0 * np.exp(np.arange(400) * 0.001), index=calendar)
        original = make_price_features(prices, calendar)
        prices.iloc[301:] *= 20
        perturbed = make_price_features(prices, calendar)
        pd.testing.assert_frame_equal(original.iloc[:301], perturbed.iloc[:301])

    def test_price_gap_is_not_forward_filled_into_eligibility(self):
        calendar = pd.bdate_range("2012-01-02", periods=160)
        prices = pd.Series(np.arange(160) + 100.0, index=calendar)
        prices.iloc[70] = np.nan
        features = make_price_features(prices, calendar)
        self.assertFalse(features.eligible.iloc[62])
        self.assertTrue(features.eligible.iloc[63])
        self.assertFalse(features.eligible.iloc[70])
        self.assertFalse(features.eligible.iloc[133])
        self.assertTrue(features.eligible.iloc[134])
        self.assertTrue(features.drawdown_252.isna().all())
        self.assertTrue(features.momentum_252_21.isna().all())

    def test_membership_uses_entry_quarter_not_signal_quarter(self):
        signals = pd.DataFrame({
            "symbol": ["AAA", "BBB"],
            "signal_date": pd.to_datetime(["2016-03-31"] * 2),
            "entry_date": pd.to_datetime(["2016-04-01"] * 2),
            "eligible": [True, True],
        })
        membership = pd.DataFrame({
            "symbol": ["AAA", "BBB"], "quarter_start": ["2016-04-01", "2016-01-01"],
            "selection_date": ["2016-03-31", "2015-12-31"], "universe_rank": [1, 2],
        })
        result = apply_dated_membership(signals, membership)
        self.assertEqual(result.eligible.tolist(), [True, False])
        membership.loc[0, "selection_date"] = "2016-04-01"
        with self.assertRaisesRegex(ValueError, "not available"):
            apply_dated_membership(signals, membership)

    def test_exact_90_day_label_rolls_weekend_to_market_session(self):
        calendar = pd.bdate_range("2020-12-30", "2021-06-01")
        # Entry Jan 4 + 90 days = Sunday Apr 4; exit Monday Apr 5.
        quotes = pd.Series(100.0, index=calendar)
        quotes.loc["2021-04-05"] = 110.0
        signals = pd.DataFrame({
            "signal_date": pd.to_datetime(["2021-01-01"]),
            "entry_date": pd.to_datetime(["2021-01-04"]), "eligible": [True],
        })
        result = make_holding_labels(signals, quotes, calendar).iloc[0]
        self.assertEqual(result.scheduled_exit, pd.Timestamp("2021-04-04"))
        self.assertEqual(result.label_end, pd.Timestamp("2021-04-05"))
        self.assertAlmostEqual(result.target, 0.10)
        self.assertTrue(result.eligible)

    def test_missing_exit_is_censored_without_searching_for_later_quote(self):
        calendar = pd.bdate_range("2020-12-30", "2021-06-01")
        quotes = pd.Series(100.0, index=calendar)
        quotes.loc["2021-04-05"] = np.nan
        quotes.loc["2021-04-06"] = 120.0
        signals = pd.DataFrame({
            "signal_date": pd.to_datetime(["2021-01-01"]),
            "entry_date": pd.to_datetime(["2021-01-04"]), "eligible": [True],
        })
        result = make_holding_labels(signals, quotes, calendar).iloc[0]
        self.assertTrue(pd.isna(result.target))
        self.assertTrue(pd.isna(result.label_end))
        self.assertEqual(result.exit_session, pd.Timestamp("2021-04-05"))
        self.assertTrue(result.eligible)

    def test_final_open_trade_label_is_unknown_and_still_eligible(self):
        calendar = pd.bdate_range("2026-08-01", "2026-09-30")
        quotes = pd.Series(100.0, index=calendar)
        signals = pd.DataFrame({
            "signal_date": pd.to_datetime(["2026-08-31"]),
            "entry_date": pd.to_datetime(["2026-09-01"]), "eligible": [True],
        })
        result = make_holding_labels(signals, quotes, calendar).iloc[0]
        self.assertTrue(pd.isna(result.target))
        self.assertTrue(pd.isna(result.exit_session))
        self.assertTrue(result.eligible)

    def test_filing_only_available_next_day_and_late_old_period_does_not_replace(self):
        fundamentals = pd.DataFrame([
            financial_row("2012-03-31", "2012-05-01", net_income=10.0),
            financial_row("2012-06-30", "2012-08-01", net_income=20.0),
            financial_row("2011-12-31", "2012-09-01", net_income=99.0),
            financial_row("2012-09-30", "2012-09-30", net_income=88.0),
        ])
        signals = pd.DataFrame({
            "symbol": ["AAA"] * 5,
            "signal_date": pd.to_datetime(["2012-05-01", "2012-05-02", "2012-08-01", "2012-08-02", "2012-10-01"]),
        })
        events = make_financial_events(fundamentals)
        self.assertEqual(len(events), 2)
        result = join_financial_events(signals, events)
        self.assertTrue(pd.isna(result.net_margin.iloc[0]))
        np.testing.assert_allclose(result.net_margin.iloc[1:].to_numpy(), [0.1, 0.1, 0.2, 0.2])
        self.assertEqual(result.filing_age_days.iloc[1], 0)

    def test_financial_staleness_does_not_turn_into_a_future_backfill(self):
        fundamentals = pd.DataFrame([
            financial_row("2010-01-01", "2010-02-01"),
            financial_row("2012-01-01", "2012-02-01", net_income=50.0),
        ])
        signals = pd.DataFrame({"symbol": ["AAA"], "signal_date": pd.to_datetime(["2011-12-30"])})
        result = join_financial_events(signals, make_financial_events(fundamentals))
        self.assertTrue(result[FINANCIAL_FEATURES].isna().all(axis=None))
        self.assertEqual(result.fiscal_date.iloc[0], pd.Timestamp("2010-01-01"))

    def test_no_financial_reports_leave_features_missing(self):
        empty = pd.DataFrame(columns=FINANCIAL_INPUTS)
        signals = pd.DataFrame({"symbol": ["AAA"], "signal_date": pd.to_datetime(["2015-12-31"])})
        events = make_financial_events(empty)
        result = join_financial_events(signals, events)
        self.assertTrue(result[FINANCIAL_FEATURES].isna().all(axis=None))
        self.assertTrue(result.filing_age_days.isna().all())

    def test_same_close_entry_is_rejected(self):
        calendar = pd.bdate_range("2021-01-01", "2021-06-01")
        quotes = pd.Series(100.0, index=calendar)
        signals = pd.DataFrame({"signal_date": pd.to_datetime(["2021-01-04"]), "entry_date": pd.to_datetime(["2021-01-04"])})
        with self.assertRaisesRegex(ValueError, "next market session"):
            make_holding_labels(signals, quotes, calendar)

    @unittest.skipUnless(importlib.util.find_spec("duckdb"), "DuckDB is not installed")
    def test_read_only_sql_pipeline_on_temporary_synthetic_database(self):
        """Exercise SQL/schema/batching end to end, without a real-market claim."""
        import duckdb

        calendar = pd.bdate_range("2008-01-01", "2026-09-30")
        calendar_frame = pd.DataFrame({"date": calendar})
        universe = pd.DataFrame({"symbol": ["AAA", "BBB"], "is_active_snapshot": [True, False]})
        prices = pd.concat([
            pd.DataFrame({"symbol": symbol, "date": calendar, "adj_close": 100.0 + np.arange(len(calendar)) * 0.01})
            for symbol in universe.symbol
        ], ignore_index=True)
        prices.loc[0, "adj_close"] = 0.0
        prices = pd.concat([prices, pd.DataFrame({"symbol": ["AAA"], "date": [pd.Timestamp("2011-01-01")], "adj_close": [100.0]})], ignore_index=True)
        benchmark = pd.DataFrame({"symbol": "^SP500TR", "date": calendar, "adj_close": 1000.0 + np.arange(len(calendar)) * 0.1})
        membership_rows = []
        for quarter in pd.date_range("2010-01-01", "2026-07-01", freq="QS"):
            for rank, symbol in enumerate(universe.symbol, start=1):
                membership_rows.append({
                    "symbol": symbol, "quarter_start": quarter,
                    "selection_date": calendar[calendar < quarter][-1],
                    "universe_rank": rank, "selection_close": 100.0,
                    "shares_outstanding": 20_000_000_000.0 if symbol == "BBB" else 1000.0,
                })
        membership = pd.DataFrame(membership_rows)
        fundamentals = pd.DataFrame([
            financial_row("2009-09-30", "2009-11-01", symbol=symbol)
            for symbol in universe.symbol
        ])
        for col in ("fiscal_date", "filing_date"):
            fundamentals[col] = pd.to_datetime(fundamentals[col])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic_loader_test.duckdb"
            with duckdb.connect(str(path)) as con:
                frames = {
                    "calendar": calendar_frame, "universe": universe,
                    "prices": prices, "benchmark": benchmark,
                    "universe_membership": membership, "fundamentals": fundamentals,
                }
                for table, frame in frames.items():
                    con.register("synthetic_input", frame)
                    con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM synthetic_input')
                    con.unregister("synthetic_input")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prepared = prepare_data(path, batch_size=1)
            self.assertEqual(prepared.audit["securities_in_union"], 2)
            self.assertEqual(prepared.audit["currently_inactive_audit_only"], 1)
            self.assertEqual(prepared.audit["quarantined_invalid_prices_loaded_window"], 1)
            self.assertEqual(prepared.audit["quarantined_off_calendar_prices_loaded_window"], 1)
            self.assertEqual(prepared.audit["ranking_inputs"]["flagged_rows"], len(membership) // 2)
            self.assertEqual(len(prepared.panel), 2 * 198)
            self.assertTrue(prepared.panel.loc[prepared.panel.symbol.eq("BBB"), "eligible"].any())
            self.assertTrue(pd.isna(prepared.prices.iloc[0].AAA))
            last = prepared.panel.loc[prepared.panel.entry_date.eq("2026-09-01")]
            self.assertTrue(last.eligible.all())
            self.assertTrue(last.target.isna().all())
            self.assertEqual(prepared.calendar[-1], pd.Timestamp("2026-09-30"))

# Training and research timing


def monthly_panel():
    """Create independent arithmetic fixtures, not simulated market returns."""
    calendar = pd.bdate_range("2009-01-01", "2018-06-30")
    rows = []
    for month_number, month in enumerate(pd.date_range("2010-01-01", "2017-12-01", freq="MS")):
        entry_index = calendar.searchsorted(month)
        entry = calendar[entry_index]
        if pd.Timestamp("2016-01-01") <= entry <= pd.Timestamp("2016-03-31"):
            continue
        signal = calendar[entry_index - 1]
        maturity = entry + pd.Timedelta(days=90)
        label_end = calendar[calendar.searchsorted(maturity)]
        for stock in range(12):
            row = {
                "signal_date": signal,
                "entry_date": entry,
                "scheduled_exit": maturity,
                "label_end": label_end,
                "symbol": f"S{stock:02d}",
                "eligible": True,
                "target": 0.02 + 0.03 * np.sin(month_number / 5) + 0.001 * stock,
            }
            for j, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = np.sin((month_number + j) / 7) + stock / (20 + j)
            rows.append(row)
    return pd.DataFrame(rows)

class ResearchTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = monthly_panel()

    def test_training_requires_strictly_completed_finite_outcomes(self):
        base = self.panel.iloc[0].copy()
        base["signal_date"] = pd.Timestamp("2010-01-29")
        base["entry_date"] = pd.Timestamp("2010-02-01")
        base["label_end"] = pd.Timestamp("2010-05-03")
        specifications = [
            ("GOOD", {}),
            ("ENDS_ON_FIT", {"label_end": pd.Timestamp("2010-06-01")}),
            ("FUTURE", {"label_end": pd.Timestamp("2010-06-02")}),
            ("UNKNOWN", {"label_end": pd.NaT}),
            ("NAN", {"target": np.nan}),
            ("INF", {"target": np.inf}),
            ("INELIGIBLE", {"eligible": False}),
            ("SAME_SIGNAL", {"signal_date": pd.Timestamp("2010-06-01")}),
            ("PRE_DEVELOPMENT", {"entry_date": pd.Timestamp("2009-12-01")}),
        ]
        rows = []
        for symbol, changes in specifications:
            row = base.copy()
            row["symbol"] = symbol
            for key, value in changes.items():
                row[key] = value
            rows.append(row)
        chosen = mature_training_rows(pd.DataFrame(rows), pd.Timestamp("2010-06-01"))
        self.assertEqual(chosen.symbol.tolist(), ["GOOD"])

    def test_censored_eligible_rows_still_receive_forecasts(self):
        panel = self.panel.copy()
        row = panel.entry_date.eq("2016-04-01") & panel.symbol.eq("S00")
        self.assertEqual(int(row.sum()), 1)
        panel.loc[row, "target"] = np.nan
        panel.loc[row, "label_end"] = pd.NaT
        scores, _ = generate_walk_forward_scores(panel, 100.0, "2016-04-01", "2016-06-30")
        target_score = scores.loc[scores.entry_date.eq("2016-04-01") & scores.symbol.eq("S00")]
        self.assertEqual(len(target_score), 1)
        self.assertTrue(np.isfinite(target_score.score.iloc[0]))

    def test_unknown_future_outcomes_cannot_change_earlier_scores(self):
        original, _ = generate_walk_forward_scores(self.panel, 100.0, "2016-04-01", "2016-06-30")
        changed = self.panel.copy()
        future = changed.label_end.ge("2016-03-31")
        changed.loc[future, "target"] += 1000.0
        perturbed, _ = generate_walk_forward_scores(changed, 100.0, "2016-04-01", "2016-06-30")
        assert_frame_equal(original.reset_index(drop=True), perturbed.reset_index(drop=True))

    def test_final_test_targets_cannot_choose_development_alpha(self):
        best, table = select_alpha(self.panel, alphas=(10.0, 100.0), validation_years=(2013, 2014, 2015))
        changed = self.panel.copy()
        changed.loc[changed.entry_date.ge("2016-04-01"), "target"] += 1000.0
        changed_best, changed_table = select_alpha(changed, alphas=(10.0, 100.0), validation_years=(2013, 2014, 2015))
        self.assertEqual(best, changed_best)
        assert_frame_equal(table.reset_index(drop=True), changed_table.reset_index(drop=True))

    def test_january_entry_is_in_its_entry_year_validation_fold(self):
        _, original = select_alpha(self.panel, alphas=(100.0,), validation_years=(2013,))
        changed = self.panel.copy()
        january = changed.entry_date.eq("2013-01-01")
        self.assertEqual(int(january.sum()), 12)
        self.assertTrue(changed.loc[january, "signal_date"].dt.year.eq(2012).all())
        changed.loc[january, "target"] += 100.0
        _, perturbed = select_alpha(changed, alphas=(100.0,), validation_years=(2013,))
        # A 2013 fold must evaluate this January outcome even though its signal
        # was on the last session of 2012. It cannot alter the pre-2013 fit.
        self.assertFalse(original.select_dtypes(include="number").equals(
            perturbed.select_dtypes(include="number")
        ))

    def test_prediction_does_not_refit_preprocessing(self):
        train = self.panel.iloc[:120].copy()
        features = list(FEATURE_COLUMNS)
        train[features[-1]] = np.nan
        model = make_model(100.0)
        model.fit(train[features], train.target)
        expected = model.predict(train[features])
        held_out = train[features].copy()
        held_out[features[0]] = 1e12
        held_out[features[-1]] = 123.0
        self.assertTrue(np.isfinite(model.predict(held_out)).all())
        np.testing.assert_array_equal(expected, model.predict(train[features]))

    def test_unverified_input_stops_and_removes_old_result_files(self):
        """The final runner must not pass a fixture off as the official input."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "synthetic_not_official.duckdb"
            database.write_bytes(b"Not a real course database")
            output = root / "outputs"
            output.mkdir()
            for name in ["metrics.csv", "REPORT.md", "daily_nav.csv", "INCOMPLETE_trades.csv"]:
                (output / name).write_text("old generated output", encoding="utf-8")
            fixture = SimpleNamespace(audit={"matches_published_sha256": False, "fixture": True})
            with patch("hw4.prepare_data", return_value=fixture), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "published.*SHA-256"):
                    run_experiment(database, output)
            status = json.loads((output / "RUN_STATUS.json").read_text())
            self.assertFalse(status["complete"])
            self.assertEqual(status["stage"], "data preparation")
            self.assertTrue((output / "INCOMPLETE.txt").is_file())
            for name in ["metrics.csv", "REPORT.md", "daily_nav.csv", "INCOMPLETE_trades.csv"]:
                self.assertFalse((output / name).exists())

    def test_report_pipeline_with_explicitly_mocked_data_then_cleans_failure_marker(self):
        """Exercise output plumbing, not a market experiment or a file-hash test.

        prepare_data is deliberately mocked. The separate test above verifies
        that an unverified real input is rejected. Every artifact here lives
        in an automatically deleted temporary directory.
        """
        dates = pd.bdate_range("2016-03-31", "2026-09-30")
        symbols = sorted(self.panel.symbol.unique())
        prices = pd.DataFrame({
            symbol: 50 + number + np.arange(len(dates)) * 0.01
            for number, symbol in enumerate(symbols)
        }, index=dates)
        fixture = SimpleNamespace(
            panel=self.panel.copy(), prices=prices,
            benchmark=pd.Series(1000 + np.arange(len(dates)) * 0.2, index=dates),
            audit={"matches_published_sha256": True,
                   "sha256": "EXPLICITLY MOCKED SYNTHETIC TEST; NOT A DATABASE FINGERPRINT"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "synthetic_plumbing_fixture.bin"
            database.write_bytes(b"The loader is mocked in this test")
            output = root / "outputs"
            output.mkdir()
            (output / "INCOMPLETE.txt").write_text("old failure", encoding="utf-8")
            def synthetic_report(output_dir, *unused):
                (output_dir / "REPORT.md").write_text(
                    "Synthetic output-plumbing fixture; not investment results.\n"
                )
            with patch("hw4.prepare_data", return_value=fixture), \
                 patch("hw4.load_registry", return_value={"events": [], "file_sha256": "synthetic"}), \
                 patch("hw4.overlay_prices", side_effect=lambda px, *args: (px, pd.DataFrame())), \
                 patch("hw4.audit_data_issues"), \
                 patch("hw4.write_full_report", side_effect=synthetic_report), \
                 contextlib.redirect_stdout(io.StringIO()):
                status = run_experiment(database, output)
            self.assertTrue(status["complete"])
            self.assertFalse((output / "INCOMPLETE.txt").exists())
            for name in ["metrics.csv", "REPORT.md", "cumulative_returns.png", "comparison.csv"]:
                self.assertGreater((output / name).stat().st_size, 0)
            metrics = pd.read_csv(output / "metrics.csv")
            self.assertEqual(set(metrics.portfolio), {"strategy", "benchmark"})

    def test_data_gap_orchestration_preserves_prefix_and_returns_incomplete_status(self):
        """A documented data stop must not masquerade as full-period success."""
        partial = SimpleNamespace(
            history=pd.DataFrame({"nav": [1_000_000.0]}, index=pd.to_datetime(["2016-03-31"])),
            trades=pd.DataFrame(), orders=pd.DataFrame(), open_positions=pd.DataFrame(),
            diagnostics={"complete": False},
        )
        fixture = SimpleNamespace(
            panel=self.panel.copy(), prices=pd.DataFrame(), benchmark=pd.Series(dtype=float),
            audit={"matches_published_sha256": True, "fixture": "mocked orchestration only"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "mocked_input.bin"
            database.write_bytes(b"Synthetic orchestration test; loader is mocked")
            output = root / "outputs"
            with patch("hw4.prepare_data", return_value=fixture), \
                 patch("hw4.load_registry", return_value={"events": [], "file_sha256": "synthetic"}), \
                 patch("hw4.overlay_prices", side_effect=lambda px, *args: (px, pd.DataFrame())), \
                 patch("hw4.run_backtest", side_effect=DataQualityError("2016-04-01", "TEST", "synthetic gap", partial)), \
                 patch("hw4.write_full_report") as report_mock, \
                 contextlib.redirect_stdout(io.StringIO()):
                status = run_experiment(database, output)
            self.assertFalse(status["complete"])
            self.assertTrue(status["analysis_report_complete"])
            self.assertTrue((output / "INCOMPLETE_daily_nav.csv").is_file())
            self.assertFalse((output / "daily_nav.csv").exists())
            report_mock.assert_not_called()  # A data gap must not produce a completed-results report.
            self.assertIn("TEST", (output / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
