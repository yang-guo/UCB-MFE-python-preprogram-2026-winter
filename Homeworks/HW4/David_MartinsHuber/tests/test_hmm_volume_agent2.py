"""Adversarial cash, order-sizing and 90-day-lot audit for ``strategies.hmm_volume``.

Round 1 of this audit attacked causality; this module attacks the money side:

* budgets versus available (not total) cash, zero/NaN/inf/negative budgets,
  duplicate/non-string symbols, tiny and huge balances, ``min_order_notional``
  rounding, equal weights and full deployment;
* missing entry quotes, rejected orders, partial baskets, several lots per
  symbol, maturity sales, delayed maturity and any sale before 90 calendar days;
* cash monotonicity, NAV reconciliation, fees, reservations and the final
  cash residue;
* empty membership, one-name universes, all-cash regimes and repeated buying
  while every existing lot is still locked;
* the engine's ``prepare``/``advance`` barrier and the strategy's use of
  ``context.portfolio.available_cash``.

A red test here is a defect report.  Nothing is relaxed to accommodate the
implementation: independently derived expectations (buffer size, equal weights,
first session on or after ``entry_date + 90 days``, the cash ledger closed from
fills alone) are asserted directly.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backtesting import Backtester, DecisionContext, Phase, Portfolio
from strategies.hmm_volume import HMMVolumeStrategy, _scale_to_cash


# The strategy keeps at least this much cash uninvested after a full basket
# (``_scale_to_cash``: ``max(1e-6, abs(cash) * 1e-9)``).
def buffer_for(cash):
    return max(1e-6, abs(cash) * 1e-9)


# --------------------------------------------------------------------------
# synthetic market and fixtures
# --------------------------------------------------------------------------
def synthetic_market(periods=420, symbols=("S0", "S1", "S2"), levels=None,
                     seed=0, start="2015-01-01"):
    """Benchmark/volume plus flat stock prices on a business-day calendar."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15, 0.3, periods), index=dates)
    rows = []
    for position, symbol in enumerate(symbols):
        level = 10.0 * 10 ** position if levels is None else float(levels[position])
        rows.append(pd.DataFrame({"date": dates, "symbol": symbol,
                                  "adj_close": np.full(periods, level)}))
    prices = pd.concat(rows, ignore_index=True)
    benchmark_frame = pd.DataFrame({"date": dates, "adj_close": benchmark})
    quarters = pd.date_range(dates[0], dates[-1], freq="QS")
    membership = pd.concat(
        [pd.DataFrame({"symbol": list(symbols), "quarter_start": quarter,
                       "selection_date": quarter - pd.Timedelta(days=1)})
         for quarter in quarters],
        ignore_index=True,
    )
    return (dates, pd.Series(benchmark, index=dates), volume, prices,
            benchmark_frame, membership)


def price_series(dates, level, change_at=None):
    """Flat level, optionally switching to ``change_at[1]`` from ``change_at[0]``."""
    values = np.full(len(dates), float(level))
    if change_at is not None:
        start, new_level = change_at
        values[dates.searchsorted(pd.Timestamp(start)):] = float(new_level)
    return values


def punch_gaps(prices, dates, gaps):
    """Remove quotes for ``symbol`` between the two date indices (inclusive)."""
    wide = prices.pivot(index="date", columns="symbol", values="adj_close").copy()
    for symbol, (first, last) in gaps.items():
        wide.loc[dates[first]:dates[last], symbol] = np.nan
    long = wide.reset_index().melt(id_vars="date", var_name="symbol",
                                   value_name="adj_close")
    return long.dropna(subset=["adj_close"])


OPTIONS = dict(median_window=20, bucket_multiple=5.0, n_states=3,
               min_train_bars=25, refit_every_bars=10, n_init=1, n_iter=50,
               seed=7, top_n=None, risk_on_states=(1, 2))


class ForcedRegime(HMMVolumeStrategy):
    """The production allocation code with a pinned HMM regime.

    ``needs_refit`` is disabled and the fit is replaced by a sentinel, so
    ``propose_buys`` exercises the real eligibility, ranking, weighting and
    scaling path without paying for a walk-forward HMM fit.
    """

    def __init__(self, benchmark, volume, *, state=2, **overrides):
        options = dict(OPTIONS)
        options.update(overrides)
        super().__init__(benchmark, volume, **options)
        self._fit = object()
        self._state = int(state)
        self._state_bar_count = 10 ** 9

    def needs_refit(self, context):
        return False


def forced(benchmark, volume, *, state=2, **overrides):
    return ForcedRegime(benchmark, volume, state=state, **overrides)


def planning_context(model, market, signal, *, cash, available=None,
                     volume=None, symbols=None):
    """A ``DecisionContext`` shaped like the engine's planning view."""
    dates, _, _, prices, _, membership = market
    signal = pd.Timestamp(signal)
    wide = prices.pivot(index="date", columns="symbol",
                        values="adj_close").loc[:signal]
    names = list(symbols if symbols is not None else
                 sorted(membership.symbol.unique()))
    quarter = (signal + pd.Timedelta(days=1)).to_period("Q").start_time
    members = pd.DataFrame({"symbol": names, "quarter_start": quarter,
                            "selection_date": signal})
    available = cash if available is None else available
    return DecisionContext(
        signal, signal + pd.Timedelta(days=1), wide, members, pd.DataFrame(),
        Portfolio(cash, available, cash, {}), 0,
        pd.DataFrame() if volume is None else volume.loc[:signal].copy(),
        pd.DataFrame(),
    )


def run(market, strategy, start_index, end_index, *, initial_cash=100_000.0,
        fee_bps=5.0, **engine_options):
    dates, _, _, prices, benchmark_frame, membership = market
    engine = Backtester(prices, benchmark_frame, membership, strategy,
                        initial_cash=initial_cash, fee_bps=fee_bps,
                        **engine_options)
    result = engine.run(dates[start_index],
                        dates[end_index] if end_index is not None else dates[-1])
    return engine, result


class LedgerAssertions:
    """Independent accounting invariants shared by every integration test."""

    def assert_ledger(self, result, initial_cash, *, fee_tolerance=None):
        history, fills = result.history, result.fills
        # 1. cash is a pure ledger: it only moves on fills.
        self.assertGreaterEqual(history.cash.min(), 0.0)
        self.assertTrue(np.isfinite(history.cash).all())
        flows = (fills.groupby("date").cash_change.sum()
                 .reindex(history.index, fill_value=0.0))
        np.testing.assert_allclose(history.cash.to_numpy(),
                                   initial_cash + flows.cumsum().to_numpy(),
                                   rtol=1e-12, atol=1e-9 * max(1.0, initial_cash))
        # 2. NAV is cash plus marked holdings, and PnL closes the loop.
        np.testing.assert_allclose(
            history.nav.to_numpy(),
            (history.cash + history.holdings_value).to_numpy(),
            rtol=1e-12, atol=1e-9 * max(1.0, initial_cash))
        self.assertTrue(np.isfinite(history.nav).all())
        self.assertGreater(history.nav.min(), 0.0)
        self.assertAlmostEqual(result.pnl.total,
                               history.nav.iloc[-1] - initial_cash,
                               delta=1e-9 * max(1.0, initial_cash))
        self.assertAlmostEqual(result.pnl.realized + result.pnl.unrealized,
                               result.pnl.total,
                               delta=1e-7 * max(1.0, initial_cash))
        # 3. the fee column is the sum of that session's fill fees.
        tolerance = fee_tolerance if fee_tolerance is not None else 1e-9 * max(1.0, initial_cash)
        self.assertAlmostEqual(history.fees.sum(), fills.fee.sum(), delta=tolerance)
        # 4. every buy stayed inside its fee-inclusive budget.
        if not fills.empty:
            merged = fills.merge(result.orders[["order_id", "budget"]],
                                 on="order_id", how="left")
            buys = merged.loc[merged.side.eq("BUY")]
            self.assertTrue((buys.notional + buys.fee
                             <= buys.budget * (1 + 1e-12) + 1e-12).all())
            self.assertTrue((buys.units > 0).all())
            self.assertTrue((buys.notional > 0).all())
            self.assertTrue((buys.fee >= 0).all())
            self.assertTrue((merged.loc[merged.side.eq("SELL"), "cash_change"]
                             >= 0).all())


class AllocationUnitChecks(unittest.TestCase):
    """Direct attacks on ``propose_buys`` sizing and ``_scale_to_cash``."""

    def setUp(self):
        self.market = synthetic_market()
        self.dates, self.benchmark, self.volume, self.prices, _, _ = self.market
        self.model = forced(self.benchmark, self.volume)

    def test_budgets_are_sized_from_available_cash_not_total_cash(self):
        context = planning_context(self.model, self.market, self.dates[300],
                                   cash=1_000.0, available=100.0)
        budgets = self.model.propose_buys(context)
        self.assertEqual(set(budgets), {"S0", "S1", "S2"})
        weights = np.array(list(budgets.values()))
        # equal weights, within one ulp of each other
        self.assertLessEqual(weights.max() - weights.min(), 1e-15 * 100.0)
        # full deployment of the *available* cash, not of the 1,000 total
        self.assertAlmostEqual(sum(budgets.values()), 100.0 - buffer_for(100.0),
                               places=9)
        self.assertLess(sum(budgets.values()), 100.0)
        self.assertLess(sum(budgets.values()), 1_000.0)

    def test_zero_available_cash_blocks_buys_despite_total_cash(self):
        context = planning_context(self.model, self.market, self.dates[300],
                                   cash=1_000_000.0, available=0.0)
        self.assertEqual(self.model.propose_buys(context), {})

    def test_extreme_balances_never_overspend_or_emit_nonfinite_budgets(self):
        # A zero floor isolates the cash-scaling rule from the min-notional one.
        model = forced(self.benchmark, self.volume, min_order_notional=0.0)
        for cash in (1e-9, 1e-6, 1e-3, 1.0, 3.0, 3.2, 12.5, 1e6, 1e12, 1e300, 1e308):
            for symbol_count in (1, 3, 5):
                with self.subTest(cash=cash, symbols=symbol_count):
                    names = [f"S{i}" for i in range(symbol_count)]
                    context = planning_context(
                        model, self.market, self.dates[300], cash=cash,
                        symbols=names)
                    budgets = model.propose_buys(context)
                    total = sum(budgets.values())
                    self.assertLessEqual(set(budgets), set(names))
                    self.assertTrue(all(np.isfinite(v) and v > 0
                                        for v in budgets.values()))
                    self.assertLessEqual(total, cash * (1 + 1e-12) + 1e-12)
                    if cash > 1e-6:
                        # near-100% deployment: only the disclosed buffer stays
                        self.assertGreaterEqual(
                            total, cash - buffer_for(cash) - 1e-6 * max(1.0, cash))
                    else:
                        self.assertEqual(total, 0.0)

    def test_min_order_notional_floor_is_never_violated(self):
        # 2.9 and 3.0 are boundary cases: 2.9/3 < 1.0, and 3.0/3 is scaled to
        # 0.99999966 by the cash buffer, so the floor drops the whole basket.
        # That is the conservative side of the contract (no order is created
        # below the floor); it is reported as an observation, not overspending.
        for cash, expected in ((2.9, 0), (3.0, 0), (3.2, 3), (10.0, 3), (30.0, 3)):
            with self.subTest(cash=cash):
                context = planning_context(self.model, self.market,
                                           self.dates[300], cash=cash)
                budgets = self.model.propose_buys(context)
                self.assertEqual(len(budgets), expected)
                self.assertTrue(all(v >= self.model.min_order_notional
                                    for v in budgets.values()))
                self.assertLessEqual(sum(budgets.values()), cash)
        # A name below the floor is dropped; the survivor keeps its scaled size.
        scaled = _scale_to_cash({"A": 0.5, "B": 5.0}, 5.5, min_notional=1.0)
        self.assertEqual(list(scaled), ["B"])
        self.assertGreater(scaled["B"], 0.0)
        self.assertLessEqual(scaled["B"], 5.5)
        # Every surviving budget clears the floor for every floor size.
        for floor in (0.0, 0.5, 1.0, 2.0, 7.0):
            with self.subTest(floor=floor):
                out = _scale_to_cash({"A": 1.0, "B": 2.0, "C": 4.0}, 6.0,
                                     min_notional=floor)
                self.assertTrue(all(v >= floor for v in out.values()))
                self.assertLessEqual(sum(out.values()), 6.0)

    def test_scale_to_cash_fails_closed_on_hostile_budgets(self):
        cases = {
            "empty": {},
            "nan_only": {"A": np.nan},
            "nan_and_finite": {"A": np.nan, "B": 5.0},
            "zero_and_negative": {"A": 0.0, "B": -1.0},
            "infinite": {"A": np.inf},
            "everything_infinite": {"A": np.inf, "B": np.inf},
        }
        for label, budgets in cases.items():
            with self.subTest(label=label):
                original = dict(budgets)
                out = _scale_to_cash(budgets, 100.0, min_notional=1.0)
                self.assertEqual(budgets, original, "input dict was mutated")
                self.assertTrue(all(np.isfinite(v) and v > 0
                                    for v in out.values()))
                self.assertLessEqual(sum(out.values()), 100.0)
        # NaN is filtered before scaling, so a finite budget survives intact.
        self.assertEqual(_scale_to_cash({"A": np.nan, "B": 5.0}, 100.0),
                         {"B": 5.0})
        self.assertEqual(_scale_to_cash({"A": 0.0, "B": -1.0}, 100.0), {})

    def test_degenerate_cash_amounts_produce_no_buys(self):
        for cash in (0.0, -1.0, -1e-9, 1e-12, 1e-9, 1e-6):
            with self.subTest(cash=cash):
                self.assertEqual(
                    _scale_to_cash({"A": 1.0, "B": 1.0}, cash, min_notional=0.0),
                    {})
                context = planning_context(self.model, self.market,
                                           self.dates[300], cash=cash)
                self.assertEqual(self.model.propose_buys(context), {})

    def test_scale_to_cash_random_sweep_keeps_the_budget_cap(self):
        rng = np.random.default_rng(20261112)
        pool = [0.0, -3.0, np.nan, np.inf, 1e-300, 1e-9, 0.25, 1.5, 7.0,
                1e299, 1e308]
        for _ in range(300):
            size = int(rng.integers(1, 7))
            cash = float(rng.choice([1e-9, 1e-6, 1e-3, 1.0, 3.0, 100.0,
                                     1e6, 1e12, 1e300]))
            budgets = {f"S{i}": float(rng.choice(pool)) for i in range(size)}
            original = dict(budgets)
            out = _scale_to_cash(budgets, cash, min_notional=1.0)
            self.assertEqual(budgets, original)
            self.assertTrue(all(np.isfinite(v) and v > 0 for v in out.values()))
            self.assertLessEqual(sum(out.values()), cash * (1 + 1e-12) + 1e-12)
            self.assertTrue(set(out) <= set(budgets))

    def test_proposals_never_leave_the_membership_or_use_bad_quotes(self):
        dates, _, _, prices, _, membership = self.market
        wide = prices.pivot(index="date", columns="symbol",
                            values="adj_close").loc[:dates[300]].copy()
        wide["INTRUDER"] = 50.0
        wide.loc[wide.index[-1], "S1"] = np.nan      # no signal-close quote
        signal = wide.index[-1]
        members = pd.DataFrame({"symbol": ["S0", "S1", "S2"],
                                "quarter_start": signal.to_period("Q").start_time,
                                "selection_date": signal - pd.Timedelta(days=1)})
        context = DecisionContext(signal, signal + pd.Timedelta(days=1), wide,
                                  members, pd.DataFrame(),
                                  Portfolio(50_000.0, 50_000.0, 50_000.0, {}),
                                  0, pd.DataFrame(), pd.DataFrame())
        frame_before = context.prices.copy()
        members_before = context.membership.copy()
        budgets = self.model.propose_buys(context)
        self.assertEqual(set(budgets), {"S0", "S2"})
        self.assertLessEqual(sum(budgets.values()), 50_000.0)
        assert_frame_equal(context.prices, frame_before)
        assert_frame_equal(context.membership, members_before)

    def test_top_n_ranking_still_deploys_cash_equally(self):
        dates, benchmark, volume, prices, _, _ = self.market
        dates = self.market[0]
        model = forced(benchmark, volume, top_n=2)
        wide = prices.pivot(index="date", columns="symbol",
                            values="adj_close").loc[:dates[300]]
        panel = pd.DataFrame(1.0, index=wide.index,
                             columns=list(wide.columns))
        panel["S2"] = 1000.0
        panel["S1"] = 500.0
        signal = wide.index[-1]
        members = pd.DataFrame({"symbol": list(wide.columns),
                                "quarter_start": signal.to_period("Q").start_time,
                                "selection_date": signal - pd.Timedelta(days=1)})
        context = DecisionContext(
            signal, signal + pd.Timedelta(days=1), wide, members,
            pd.DataFrame(), Portfolio(10_000.0, 10_000.0, 10_000.0, {}), 0,
            panel, pd.DataFrame())
        budgets = model.propose_buys(context)
        self.assertEqual(set(budgets), {"S1", "S2"})
        expected = (10_000.0 - buffer_for(10_000.0)) / 2
        np.testing.assert_allclose(list(budgets.values()), [expected, expected],
                                   rtol=1e-12)
        self.assertAlmostEqual(sum(budgets.values()),
                               10_000.0 - buffer_for(10_000.0), places=9)


class BacktestAccountingChecks(unittest.TestCase, LedgerAssertions):
    """End-to-end runs through the real engine and OMS."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(periods=420, seed=0)
        cls.dates = cls.market[0]
        cls.engine, cls.result = run(cls.market, forced(cls.market[1], cls.market[2]),
                                     start_index=200, end_index=None,
                                     initial_cash=100_000.0, fee_bps=5.0)

    def test_run_is_a_closed_cash_ledger(self):
        self.assertFalse(self.result.fills.empty)
        self.assert_ledger(self.result, 100_000.0)
        # nothing is left pending and reservations are fully released
        self.assertTrue(self.result.orders.status.eq("filled").all())
        self.assertEqual(self.engine.oms.available_cash, self.engine.oms.cash)
        self.assertEqual(self.result.phases.phase.iloc[-1], Phase.COMPLETE.value)

    def test_every_sale_is_a_maturity_sale_after_90_calendar_days(self):
        fills = self.result.fills
        sells = fills.loc[fills.side.eq("SELL")]
        self.assertFalse(sells.empty)
        self.assertTrue(sells.trigger.eq("maturity").all())
        self.assertTrue(sells.lot_id.is_unique, "a lot was sold twice")
        self.assertTrue((sells.date >= sells.due_date).all())
        entries = fills.loc[fills.side.eq("BUY")].set_index("lot_id").date
        entry_dates = entries.reindex(sells.lot_id).to_numpy()
        holds = (sells.due_date.to_numpy() - entry_dates).astype("timedelta64[D]")
        self.assertTrue((holds.astype(int) == 90).all(),
                        "a lot did not carry exactly a 90-calendar-day due date")
        # each sale happens on the first session on or after maturity
        calendar = self.result.history.index
        for row in sells.itertuples():
            expected = calendar[calendar.searchsorted(row.due_date)]
            self.assertEqual(row.date, expected)

    def test_equal_weight_orders_and_full_deployment_per_signal_date(self):
        orders = self.result.orders.loc[self.result.orders.side.eq("BUY")]
        self.assertFalse(orders.empty)
        for signal_date, group in orders.groupby("signal_date"):
            with self.subTest(signal_date=signal_date):
                spread = group.budget.max() - group.budget.min()
                self.assertLessEqual(spread, 1e-12 * group.budget.max())
        history = self.result.history
        fills = self.result.fills
        for execution_date, group in orders.groupby("execution_date"):
            if not group.status.eq("filled").all():
                continue  # a rejected leg makes the basket partial by design
            cash_before = history.cash.iloc[history.index.get_loc(execution_date) - 1]
            spent = (fills.loc[fills.date.eq(execution_date)
                               & fills.side.eq("BUY"), ["notional", "fee"]]
                     .sum().sum())
            leftover = cash_before - spent
            self.assertGreaterEqual(leftover, 0.0)
            self.assertLessEqual(leftover, buffer_for(cash_before) + 1e-9 * cash_before)
            self.assertAlmostEqual(
                group.budget.sum(), cash_before - buffer_for(cash_before),
                delta=1e-9 * cash_before)

    def test_open_lots_agree_with_fills_and_the_final_nav(self):
        result, fills = self.result, self.result.fills
        lots = result.open_lots
        self.assertEqual(len(lots), int(result.history.open_lots.iloc[-1]))
        self.assertTrue(lots.lot_id.is_unique)
        buys = fills.loc[fills.side.eq("BUY")].set_index("lot_id")
        for row in lots.itertuples():
            self.assertIn(row.lot_id, buys.index)
            self.assertEqual(buys.loc[row.lot_id, "units"], row.units)
            self.assertEqual(buys.loc[row.lot_id, "price"], row.entry_price)
            self.assertEqual(row.due_date, row.entry_date + pd.Timedelta(days=90))
        sold_orders = set(result.orders.loc[result.orders.side.eq("SELL"), "order_id"])
        self.assertFalse(sold_orders & set(lots.lot_id))
        wide = (self.market[3].pivot(index="date", columns="symbol",
                                     values="adj_close")
                .loc[:result.history.index[-1]])
        marks = wide.ffill().iloc[-1]
        expected = float((lots.units * lots.symbol.map(marks)).sum())
        self.assertAlmostEqual(result.history.holdings_value.iloc[-1], expected,
                               delta=1e-9 * max(1.0, expected))

    def test_fee_extremes_preserve_the_ledger(self):
        for fee_bps in (0.0, 5.0, 10_000.0):
            with self.subTest(fee_bps=fee_bps):
                _, result = run(self.market,
                                forced(self.market[1], self.market[2]),
                                start_index=200, end_index=None,
                                initial_cash=100_000.0, fee_bps=fee_bps)
                self.assert_ledger(result, 100_000.0)
                if fee_bps == 10_000.0:
                    sells = result.fills.loc[result.fills.side.eq("SELL")]
                    if not sells.empty:
                        np.testing.assert_allclose(sells.cash_change, 0.0,
                                                   atol=1e-9)


class MissingQuoteChecks(unittest.TestCase, LedgerAssertions):
    """Rejected entries, partial baskets, late maturity, delisting exits."""

    def test_missing_entry_quote_rejects_and_releases_the_reservation(self):
        market = synthetic_market(periods=420, seed=2)
        dates = market[0]
        prices = punch_gaps(market[3], dates, {"S1": (200, 200)})
        market = (dates, market[1], market[2], prices, market[4], market[5])
        engine, result = run(market, forced(market[1], market[2]),
                             start_index=200, end_index=None,
                             initial_cash=30_000.0, fee_bps=5.0)

        self.assert_ledger(result, 30_000.0)
        rejected = result.orders.loc[result.orders.status.ne("filled")]
        self.assertEqual(len(rejected), 1)
        self.assertEqual((rejected.symbol.iloc[0], rejected.status.iloc[0]),
                         ("S1", "rejected_missing_quote"))
        day_one = result.fills.loc[result.fills.date.eq(dates[200])]
        self.assertEqual(set(day_one.symbol), {"S0", "S2"})
        self.assertFalse(rejected.order_id.isin(result.fills.order_id).any())
        # The rejected 1/3 reservation must be spendable on the next decision.
        day_two = result.orders.loc[result.orders.execution_date.eq(dates[201])]
        self.assertFalse(day_two.empty)
        cash_before = result.history.cash.loc[dates[200]]
        self.assertAlmostEqual(day_two.budget.sum(),
                               cash_before - buffer_for(cash_before),
                               delta=1e-9 * cash_before)
        spent_two = (result.fills.loc[result.fills.date.eq(dates[201])
                                      & result.fills.side.eq("BUY"),
                                      ["notional", "fee"]].sum().sum())
        self.assertGreater(spent_two, 0.9 * cash_before)
        self.assertGreater(engine.oms.available_cash, 0.0)
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)

    def test_partial_basket_buys_only_the_quoting_name(self):
        market = synthetic_market(periods=420, seed=2)
        dates = market[0]
        prices = punch_gaps(market[3], dates,
                            {"S0": (200, len(dates) - 1),
                             "S1": (200, len(dates) - 1)})
        market = (dates, market[1], market[2], prices, market[4], market[5])
        engine, result = run(market, forced(market[1], market[2]),
                             start_index=200, end_index=None,
                             initial_cash=9_000.0, fee_bps=5.0)

        self.assert_ledger(result, 9_000.0)
        self.assertFalse(result.fills.empty)
        self.assertEqual(set(result.fills.symbol), {"S2"})
        day_one = result.orders.loc[result.orders.execution_date.eq(dates[200])]
        self.assertEqual(set(day_one.symbol), {"S0", "S1", "S2"})
        # one third each on the signal date, but only the quoted name fills
        self.assertEqual(set(day_one.loc[day_one.status.eq("filled"), "symbol"]),
                         {"S2"})
        self.assertEqual(len(day_one.loc[day_one.status.eq("rejected_missing_quote")]), 2)
        per_name = (9_000.0 - buffer_for(9_000.0)) / 3
        self.assertAlmostEqual(day_one.budget.max(), per_name, delta=1e-9 * 9_000.0)
        cash_after_one = result.history.cash.loc[dates[200]]
        self.assertAlmostEqual(cash_after_one, 9_000.0 - per_name,
                               delta=1e-9 * 9_000.0)
        # the untouched two thirds are fully redeployed on the next decision
        day_two = result.orders.loc[result.orders.execution_date.eq(dates[201])]
        self.assertEqual(set(day_two.symbol), {"S2"})
        self.assertAlmostEqual(day_two.budget.sum(),
                               cash_after_one - buffer_for(cash_after_one),
                               delta=1e-9 * 9_000.0)
        self.assertLessEqual(result.history.cash.iloc[-1], buffer_for(9_000.0))
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)

    def test_maturity_waits_for_a_quote_and_uses_the_returning_price(self):
        dates = pd.bdate_range("2015-01-01", periods=520)
        market = synthetic_market(periods=520, symbols=("ONLY",),
                                  seed=3)
        prices = market[3].copy()
        prices["adj_close"] = price_series(dates, 10.0,
                                           change_at=(dates[300], 12.0))
        gap_prices = punch_gaps(prices, dates, {"ONLY": (210, 400)})
        market = (dates, market[1], market[2], gap_prices, market[4], market[5])
        model = forced(market[1], market[2])
        engine, result = run(market, model, start_index=200, end_index=None,
                             initial_cash=50_000.0, fee_bps=5.0,
                             max_stale_sessions=200)

        self.assert_ledger(result, 50_000.0)
        buys = result.fills.loc[result.fills.side.eq("BUY")]
        sells = result.fills.loc[result.fills.side.eq("SELL")]
        bought = buys.iloc[0]
        self.assertEqual(bought.date, dates[200])
        self.assertEqual(bought.price, 10.0)
        first_sale = sells.iloc[0]
        self.assertEqual(first_sale.lot_id, bought.lot_id)
        self.assertGreaterEqual(first_sale.date, bought.due_date)
        self.assertEqual(first_sale.date, dates[401])
        # the sale is priced at the observed quote, never at the carried mark
        self.assertEqual(first_sale.price, 12.0)
        self.assertIn("maturity_exit_delayed", set(result.diagnostics.event))
        self.assertIn("stale_mark", set(result.diagnostics.event))
        # valuation during the gap is flat at the last observed mark
        gap = result.history.loc[dates[215]:dates[395]]
        self.assertTrue(gap.holdings_value.nunique() == 1)
        self.assertTrue(gap.cash.nunique() == 1)
        self.assertTrue(gap.stale_marks.gt(0).all())
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)

    def test_delisting_last_price_is_not_a_supported_execution_policy(self):
        market = synthetic_market(periods=520, symbols=("ONLY",), seed=4)
        with self.assertRaisesRegex(ValueError, "stale mark"):
            run(market, forced(market[1], market[2]),
                start_index=200, end_index=None, delisting="last_price")

    def test_universe_removal_does_not_liquidate_a_locked_lot(self):
        market = synthetic_market(periods=420, seed=12)
        dates = market[0]
        membership = market[5]
        # S1 leaves the universe after the quarter in which it is bought.
        membership = membership.loc[~((membership.symbol == "S1")
                                      & (membership.quarter_start > dates[200]))]
        market = (dates, market[1], market[2], market[3], market[4], membership)
        engine, result = run(market, forced(market[1], market[2]),
                             start_index=200, end_index=None,
                             initial_cash=100_000.0, fee_bps=5.0)

        self.assert_ledger(result, 100_000.0)
        s1 = result.fills.loc[result.fills.symbol.eq("S1")]
        self.assertFalse(s1.empty, "S1 was never bought; the test is vacuous")
        s1_buys = s1.loc[s1.side.eq("BUY")]
        s1_sells = s1.loc[s1.side.eq("SELL")]
        self.assertEqual(len(s1_sells), len(s1_buys))
        self.assertTrue((s1_sells.date >= s1_sells.due_date).all())
        # after the removal the strategy stops adding S1 but never sells early
        first_entry = s1_buys.date.min()
        quarter_start = (first_entry + pd.Timedelta(days=1)).to_period("Q").start_time
        later_buys = s1_buys.loc[s1_buys.date >= quarter_start + pd.Timedelta(days=92)]
        self.assertTrue(later_buys.empty)
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)


class DegenerateUniverseChecks(unittest.TestCase, LedgerAssertions):
    """All-cash regimes, empty membership, one name, repeated lots, lifecycle."""

    def test_risk_off_run_holds_cash_only(self):
        market = synthetic_market(periods=420, seed=5)
        engine, result = run(market, forced(market[1], market[2], state=0),
                             start_index=200, end_index=None,
                             initial_cash=12_345.67, fee_bps=5.0)
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.orders.empty)
        self.assertTrue(result.open_lots.empty)
        self.assertTrue(result.history.cash.eq(12_345.67).all())
        self.assertTrue(result.history.nav.eq(12_345.67).all())
        self.assertEqual(result.pnl.total, 0.0)
        self.assertEqual(engine.oms.available_cash, 12_345.67)

    def test_empty_membership_holds_cash_only(self):
        market = synthetic_market(periods=420, seed=5)
        empty = market[5].iloc[0:0].copy()
        market = (market[0], market[1], market[2], market[3], market[4], empty)
        engine, result = run(market, forced(market[1], market[2]),
                             start_index=200, end_index=None,
                             initial_cash=1_000.0, fee_bps=5.0)
        self.assertTrue(result.fills.empty)
        self.assertTrue(result.orders.empty)
        self.assertTrue(result.history.cash.eq(1_000.0).all())
        self.assertTrue(result.history.nav.eq(1_000.0).all())

    def test_one_name_universe_leaves_exactly_the_buffer(self):
        market = synthetic_market(periods=420, symbols=("ONLY",), seed=6)
        engine, result = run(market, forced(market[1], market[2]),
                             start_index=200, end_index=201,
                             initial_cash=1_000_000.0, fee_bps=5.0)
        self.assert_ledger(result, 1_000_000.0)
        self.assertEqual(len(result.orders), 1)
        order = result.orders.iloc[0]
        expected = 1_000_000.0 - buffer_for(1_000_000.0)
        self.assertAlmostEqual(order.budget, expected, delta=1e-9 * 1_000_000.0)
        fill = result.fills.iloc[0]
        self.assertAlmostEqual(fill.notional + fill.fee, order.budget,
                               delta=1e-9 * 1_000_000.0)
        self.assertAlmostEqual(result.history.cash.iloc[-1], buffer_for(1_000_000.0),
                               delta=1e-9 * 1_000_000.0)
        self.assertEqual(result.history.open_lots.iloc[-1], 1)

    def test_staggered_entries_create_independent_locked_lots(self):
        dates = synthetic_market(periods=420, seed=7)[0]
        market = synthetic_market(periods=420, seed=7)
        prices = punch_gaps(market[3], dates, {"S1": (200, 200)})
        market = (dates, market[1], market[2], prices, market[4], market[5])
        engine, result = run(market, forced(market[1], market[2]),
                             start_index=200, end_index=None,
                             initial_cash=300_000.0, fee_bps=5.0)

        self.assert_ledger(result, 300_000.0)
        lots = result.open_lots
        self.assertGreaterEqual(lots.symbol.value_counts().max(), 2)
        self.assertTrue(lots.lot_id.is_unique)
        # lots are never merged: each holds its own entry price and maturity
        self.assertTrue((lots.due_date == lots.entry_date
                         + pd.Timedelta(days=90)).all())
        buys = result.fills.loc[result.fills.side.eq("BUY")]
        self.assertGreater(buys.lot_id.nunique(), buys.symbol.nunique())
        # the same symbol can be bought again while its first lot is locked
        for symbol, group in buys.groupby("symbol"):
            if len(group) < 2:
                continue
            first_due = group.due_date.iloc[0]
            second_entry = group.date.iloc[1]
            self.assertLess(second_entry, first_due)
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)

    def test_prepare_reserves_without_debiting_and_advance_books_fills(self):
        market = synthetic_market(periods=420, seed=8)
        dates = market[0]
        engine = Backtester(market[3], market[4], market[5],
                            forced(market[1], market[2]),
                            initial_cash=100_000.0, fee_bps=5.0)
        engine.begin(dates[200], dates[-1])
        self.assertEqual(engine.phase, Phase.READY)
        self.assertEqual(engine.oms.cash, 100_000.0)
        with self.assertRaises(ValueError):
            engine.advance()          # cannot execute before a decision

        engine.prepare()
        self.assertEqual(engine.phase, Phase.DECIDING)
        self.assertEqual(engine.oms.cash, 100_000.0, "reservations must not debit")
        self.assertEqual(len(engine.oms.orders), 3)
        self.assertTrue(all(order.status == "pending"
                            for order in engine.oms.orders.values()))
        self.assertAlmostEqual(engine.oms.available_cash, buffer_for(100_000.0),
                               delta=1e-9 * 100_000.0)
        # pending orders create no returns and no history row
        self.assertEqual(len(engine.history_so_far), 1)
        self.assertEqual(engine.history_so_far.nav.iloc[-1], 100_000.0)
        with self.assertRaises(ValueError):
            engine.prepare()          # one decision per session

        engine.advance()
        self.assertEqual(len(engine.oms.fills), 3)
        self.assertEqual(len(engine.oms.lots), 3)
        self.assertLess(engine.oms.cash, 100_000.0)
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)
        self.assertEqual(engine.next_session, dates[201])
        row = engine.history_so_far.iloc[-1]
        self.assertAlmostEqual(row.nav, row.cash + row.holdings_value, places=8)


class EngineBoundaryChecks(unittest.TestCase):
    """Contract guards the HMM strategy must never trip, plus rollback."""

    class Hostile(HMMVolumeStrategy):
        def __init__(self, benchmark, volume, payload):
            options = dict(OPTIONS)
            super().__init__(benchmark, volume, **options)
            self._fit = object()
            self._state = 2
            self._state_bar_count = 10 ** 9
            self.payload = payload

        def needs_refit(self, context):
            return False

        def propose_buys(self, context):
            return self.payload

    def test_out_of_contract_budgets_fail_atomically(self):
        market = synthetic_market(periods=420, seed=9)
        dates = market[0]
        cases = {
            "above_cash": {"S0": 200_000.0},
            "nan": {"S0": np.nan},
            "inf": {"S0": np.inf},
            "negative": {"S0": -1.0},
            "outside_universe": {"NOPE": 10.0},
            "non_string_symbol": {7: 10.0},
            "zero": {"S0": 0.0},
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                engine = Backtester(market[3], market[4], market[5],
                                    self.Hostile(market[1], market[2], payload),
                                    initial_cash=1_000.0, fee_bps=5.0)
                if label == "zero":
                    result = engine.run(dates[200], dates[202])
                    self.assertTrue(result.fills.empty)
                    self.assertTrue(result.orders.empty)
                    self.assertEqual(engine.oms.cash, 1_000.0)
                    continue
                with self.assertRaises(ValueError):
                    engine.run(dates[200], dates[202])
                self.assertEqual(engine.phase, Phase.FAILED)
                self.assertEqual(engine.oms.cash, 1_000.0)
                self.assertEqual(engine.oms.orders, {})
                self.assertEqual(engine.oms.fills, [])
                self.assertEqual(engine.oms.available_cash, 1_000.0)

    def test_str_subclass_symbols_are_rejected(self):
        market = synthetic_market(periods=420, seed=9)
        dates = market[0]

        class Sneaky(str):
            pass

        engine = Backtester(market[3], market[4], market[5],
                            self.Hostile(market[1], market[2],
                                         {Sneaky("S0"): 10.0}),
                            initial_cash=1_000.0, fee_bps=5.0)
        with self.assertRaises(ValueError):
            engine.run(dates[200], dates[202])
        self.assertEqual(engine.oms.cash, 1_000.0)
        self.assertEqual(engine.oms.orders, {})


class RealModelChecks(unittest.TestCase, LedgerAssertions):
    """One genuine walk-forward HMM run, audited with the same ledger rules."""

    def test_real_hmm_run_keeps_the_cash_and_holding_rules(self):
        market = synthetic_market(periods=800, symbols=("S0", "S1", "S2", "S3",
                                                        "S4", "S5", "S6", "S7"),
                                  seed=11)
        dates = market[0]
        model = HMMVolumeStrategy(
            market[1], market[2], median_window=20, min_train_bars=30,
            refit_every_bars=10, n_init=1, n_iter=50, top_n=4,
            risk_on_states=(1, 2), seed=7)
        engine, result = run(market, model, start_index=250, end_index=750,
                             initial_cash=500_000.0, fee_bps=5.0)

        self.assertFalse(result.refits.empty)
        self.assertTrue(result.refits.status.eq("success").all())
        self.assertFalse(result.fills.empty)
        self.assert_ledger(result, 500_000.0)
        sells = result.fills.loc[result.fills.side.eq("SELL")]
        self.assertTrue(sells.trigger.eq("maturity").all())
        self.assertTrue((sells.date >= sells.due_date).all())
        buys = result.orders.loc[result.orders.side.eq("BUY")]
        for _, group in buys.groupby("signal_date"):
            self.assertLessEqual(group.budget.max() - group.budget.min(),
                                 1e-12 * group.budget.max())
        self.assertEqual(engine.oms.available_cash, engine.oms.cash)


if __name__ == "__main__":
    unittest.main()
