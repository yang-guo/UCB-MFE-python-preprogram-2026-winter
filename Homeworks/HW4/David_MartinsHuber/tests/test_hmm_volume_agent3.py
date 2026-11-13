"""Adversarial tests for the volume-bar HMM lifetime: refit cadence, state.

These tests target the refit lifecycle and the state machine of
``strategies/hmm_volume.py``:

* ``needs_refit`` cadence (first fit, minimum bars, exactly
  ``refit_every_bars``, skipped bars, signal dates between bar ends),
* what ``refit`` trains on and which audit fields it stores,
* state updates between refits via ``propose_buys``,
* the engine's ``self_fitting`` deepcopy, model versions, and refit audit,
* constructor validation and parameter boundaries.

The tests inspect public behaviour (``needs_refit``, ``refit``,
``propose_buys``, ``state``, ``state_date``, ``probabilities``) *and* the
stored audit fields (``_fit_bar_count``, ``_fit_bar_end``,
``_state_bar_count``, ``_state_bar_end``, ``_fit_date``, and the engine's
``refits`` table).

Tests that currently fail document confirmed defects; each one says what
invariant is violated.
"""

from copy import deepcopy
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

import strategies.hmm_volume as hmm_volume
from backtesting import Backtester, DecisionContext, Phase, Portfolio
from research.hmm import fit_hmm, infer_state, walk_forward_states
from strategies.hmm_volume import HMMVolumeStrategy

warnings.filterwarnings("ignore", category=DeprecationWarning)


def setUpModule():
    # unittest.runner installs a "default" filter before running the module;
    # hmmlearn emits a NumPy 2.5 deprecation on every fit, which would drown
    # the test report.
    warnings.simplefilter("ignore", DeprecationWarning)

REFIT_COLUMNS = {
    "signal_date",
    "model_version",
    "rows",
    "first_date",
    "last_date",
    "latest_label_end",
    "latest_available",
    "settings",
    "status",
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def synthetic_market(periods=520, symbols=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15, 0.3, periods), index=dates)
    price_rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        prices = benchmark * np.exp(np.cumsum(rng.normal(0, 0.005, periods)))
        price_rows.append(
            pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": prices})
        )
    prices = pd.concat(price_rows, ignore_index=True)
    benchmark_frame = pd.DataFrame({"date": dates, "adj_close": benchmark})
    quarters = pd.date_range(dates[0], dates[-1], freq="QS")
    membership = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": [f"S{i}" for i in range(symbols)],
                    "quarter_start": quarter,
                    "selection_date": quarter - pd.Timedelta(days=1),
                }
            )
            for quarter in quarters
        ],
        ignore_index=True,
    )
    return (dates, pd.Series(benchmark, index=dates), volume,
            prices, benchmark_frame, membership)


DEFAULTS = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=20,
    refit_every_bars=5,
    n_init=1,
    n_iter=50,
    seed=7,
    top_n=3,
    risk_on_states=(1, 2),
)


def build(seed=0, periods=520, symbols=6, **overrides):
    dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(
        periods=periods, symbols=symbols, seed=seed
    )
    options = dict(DEFAULTS)
    options.update(overrides)
    model = HMMVolumeStrategy(benchmark, volume, **options)
    return model, dates, benchmark, volume, prices, benchmark_frame, membership


def context(model, signal, prices, membership, *, nav=100_000.0, cash=None,
            holdings=None, volume=None, model_version=0):
    """Build a DecisionContext whose price view stops at ``signal``."""
    signal = pd.Timestamp(signal)
    execution = signal + pd.Timedelta(days=1)
    cash = nav if cash is None else cash
    if "date" in prices.columns:
        price_view = (
            prices.loc[prices["date"] <= signal]
            .pivot(index="date", columns="symbol", values="adj_close")
        )
    else:
        price_view = prices.loc[:signal].copy()
    return DecisionContext(
        signal,
        execution,
        price_view,
        membership.copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, nav, dict(holdings or {})),
        model_version,
        pd.DataFrame() if volume is None else volume.loc[:signal].copy(),
        pd.DataFrame(),
    )


def bar_count(model, signal):
    return int(len(model._bars_at(pd.Timestamp(signal))))


def volume_panel(dates, symbols, fill=1.0):
    return pd.DataFrame(fill, index=dates, columns=list(symbols))


class RefitLifecycleChecks(unittest.TestCase):
    """needs_refit cadence and the first-fit boundary."""

    def test_first_fit_needs_one_more_bar_than_min_train(self):
        model, dates, _, _, prices, _, membership = build(seed=1)
        bars = model._bars
        first = model.min_train_bars
        for count in (0, 1, first - 1, first):
            with self.subTest(completed_bars=count):
                signal = bars["end"].iloc[count - 1] if count else dates[0]
                self.assertEqual(bar_count(model, signal), count)
                self.assertFalse(model.needs_refit(context(model, signal, prices, membership)))
        self.assertTrue(
            model.needs_refit(context(model, bars["end"].iloc[first], prices, membership))
        )
        self.assertEqual(bar_count(model, bars["end"].iloc[first]), first + 1)

    def test_needs_refit_waits_for_a_new_completed_bar(self):
        model, dates, _, _, prices, _, membership = build(seed=2)
        bars = model._bars
        first = model.min_train_bars
        fit_signal = bars["end"].iloc[first]
        model.refit(None, context(model, fit_signal, prices, membership))
        self.assertEqual(model._fit_bar_count, first + 1)

        # Same bar end, later signal session: no new bar, never due.
        later_sessions = dates[(dates > fit_signal) & (dates < bars["end"].iloc[first + 1])]
        self.assertGreater(len(later_sessions), 0)
        for signal in later_sessions:
            with self.subTest(signal=signal):
                self.assertEqual(bar_count(model, signal), first + 1)
                self.assertFalse(model.needs_refit(context(model, signal, prices, membership)))

        # Off-by-one below the cadence and exactly at the cadence.
        cadence = model.refit_every_bars
        for offset in range(1, cadence):
            with self.subTest(offset=offset):
                signal = bars["end"].iloc[first + offset]
                self.assertFalse(model.needs_refit(context(model, signal, prices, membership)))
        due = bars["end"].iloc[first + cadence]
        self.assertTrue(model.needs_refit(context(model, due, prices, membership)))
        model.refit(None, context(model, due, prices, membership))
        # The cadence restarts from the new fit bar, not from the old one.
        self.assertFalse(model.needs_refit(context(model, due, prices, membership)))
        self.assertFalse(
            model.needs_refit(context(model, bars["end"].iloc[first + cadence + 1],
                                     prices, membership))
        )
        self.assertTrue(
            model.needs_refit(context(model, bars["end"].iloc[first + 2 * cadence],
                                     prices, membership))
        )

    def test_needs_refit_is_not_fooled_by_a_bar_gap(self):
        """A signal date that skips several bars refits once and resets."""
        model, dates, _, _, prices, _, membership = build(seed=3)
        bars = model._bars
        first = model.min_train_bars
        cadence = model.refit_every_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))

        jumped = first + 3 * cadence  # three cadences skipped at once
        signal = bars["end"].iloc[jumped]
        self.assertTrue(model.needs_refit(context(model, signal, prices, membership)))
        model.refit(None, context(model, signal, prices, membership))
        # The count is measured from the observed bar, not from a stale plan.
        self.assertEqual(model._fit_bar_count, jumped + 1)
        self.assertEqual(model._fit_bar_end, signal)
        self.assertEqual(model.state_date, signal)
        self.assertFalse(model.needs_refit(context(model, signal, prices, membership)))
        self.assertFalse(
            model.needs_refit(context(model, bars["end"].iloc[jumped + 1], prices, membership))
        )
        self.assertTrue(
            model.needs_refit(context(model, bars["end"].iloc[jumped + cadence],
                                     prices, membership))
        )


    def test_refit_every_one_bar_refits_on_every_new_bar(self):
        model, dates, _, _, prices, _, membership = build(seed=25, refit_every_bars=1)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        for index in range(first + 1, first + 4):
            with self.subTest(bar=index):
                signal = bars["end"].iloc[index]
                ctx = context(model, signal, prices, membership)
                self.assertTrue(model.needs_refit(ctx))
                previous = model._fit
                model.refit(None, ctx)
                self.assertIsNot(model._fit, previous)
                self.assertEqual(model._fit_bar_count, index + 1)
                self.assertFalse(model.needs_refit(ctx))


class RefitFittingChecks(unittest.TestCase):
    """What refit trains on, what it scores, what it stores."""

    def test_refit_trains_on_bars_before_the_signal_bar_and_scores_the_last(self):
        model, dates, benchmark, volume, prices, _, membership = build(seed=4)
        bars = model._bars
        first = model.min_train_bars
        signal = bars["end"].iloc[first]
        ctx = context(model, signal, prices, membership)
        model.refit(None, ctx)
        returns = bars["log_return"].to_numpy(dtype=float)

        fresh = fit_hmm(returns[:first], n_states=3, n_init=1, seed=7, n_iter=50)
        np.testing.assert_allclose(model._fit.model.means_, fresh.model.means_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model._fit.model.transmat_, fresh.model.transmat_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model._fit.model.covars_, fresh.model.covars_, rtol=0, atol=1e-12)
        self.assertEqual(model._fit.mean, fresh.mean)
        self.assertEqual(model._fit.scale, fresh.scale)

        expected_state, expected_probabilities = infer_state(fresh, returns[: first + 1])
        self.assertEqual(model.state, expected_state)
        np.testing.assert_allclose(model.probabilities, expected_probabilities)

        # Stored audit fields.
        self.assertEqual(model._fit_bar_count, first + 1)
        self.assertEqual(model._state_bar_count, first + 1)
        self.assertEqual(model._fit_bar_end, signal)
        self.assertEqual(model._state_bar_end, signal)
        self.assertEqual(model.state_date, signal)
        self.assertEqual(model._fit_date, pd.Timestamp(signal))
        self.assertEqual(len(model._bar_returns), first + 1)
        self.assertEqual(model.probabilities.shape, (3,))
        self.assertAlmostEqual(float(model.probabilities.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(model.probabilities)), model.state)

    def test_scored_bar_return_never_enters_the_fit(self):
        """Changing the signal bar's own return must not change the fit."""
        model, dates, benchmark, volume, prices, _, membership = build(seed=8)
        bars = model._bars
        first = model.min_train_bars
        signal = bars["end"].iloc[first]
        ctx = context(model, signal, prices, membership)
        model.refit(None, ctx)

        bumped = benchmark.copy()
        bumped.loc[signal] = float(bumped.loc[signal]) * 1.5
        scenario = HMMVolumeStrategy(bumped, volume, **DEFAULTS)
        scenario.refit(None, context(scenario, signal, prices, membership))

        # Same bars (volume drives the boundaries), different scored return.
        self.assertEqual(scenario._bars["end"].iloc[first], bars["end"].iloc[first])
        self.assertNotAlmostEqual(
            float(scenario._bar_returns[-1]), float(model._bar_returns[-1]), places=6
        )
        # ... but the training sample is untouched.
        np.testing.assert_array_equal(
            scenario._fit.model.means_, model._fit.model.means_
        )
        np.testing.assert_array_equal(
            scenario._fit.model.transmat_, model._fit.model.transmat_
        )
        self.assertEqual(scenario._fit_bar_count, model._fit_bar_count)
        self.assertEqual(scenario._fit_bar_end, model._fit_bar_end)

    def test_refit_lifecycle_matches_research_walk_forward(self):
        model, dates, benchmark, volume, prices, _, membership = build(
            seed=5, periods=600
        )
        bars = model._bars
        returns = bars["log_return"].to_numpy(dtype=float)
        research = walk_forward_states(
            returns, bars["end"], min_train=model.min_train_bars,
            refit_every=model.refit_every_bars, n_states=model.n_states,
            n_init=model.n_init, seed=model.seed, n_iter=model.n_iter,
        )
        fit_indices = []
        for index in range(model.min_train_bars, len(bars)):
            signal = bars["end"].iloc[index]
            ctx = context(model, signal, prices, membership)
            if model.needs_refit(ctx):
                model.refit(None, ctx)
                fit_indices.append(index)
                self.assertEqual(model._fit_bar_count, index + 1)
                self.assertEqual(model._fit_bar_end, signal)
                self.assertEqual(model.state_date, signal)
            self.assertIsNotNone(model._fit)
            model._update_state(model._bars_at(signal))
            self.assertEqual(model._state_bar_count, index + 1)
            self.assertEqual(model.state_date, bars["end"].iloc[index])
            self.assertEqual(
                model.state, int(research.states.iloc[index]),
                msg=f"state diverges from research at bar {index}",
            )

        research_fits = [
            int(bars.index[bars["end"] == fit_date][0])
            for fit_date, _version, _fit in research.fits
        ]
        self.assertEqual(fit_indices, research_fits)

    def test_refit_without_enough_bars_is_refused(self):
        model, dates, _, _, prices, _, membership = build(seed=6)
        bars = model._bars
        signal = bars["end"].iloc[model.min_train_bars - 1]
        with self.assertRaises(ValueError):
            model.refit(None, context(model, signal, prices, membership))
        self.assertIsNone(model._fit)
        self.assertIsNone(model.state)
        self.assertIsNone(model.state_date)
        self.assertEqual(model._fit_bar_count, 0)
        self.assertEqual(model._state_bar_count, 0)

    def test_failed_scoring_leaves_the_active_model_untouched(self):
        model, dates, _, _, prices, _, membership = build(seed=7)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        before = dict(
            fit=model._fit,
            state=model.state,
            probabilities=model.probabilities,
            fit_count=model._fit_bar_count,
            state_count=model._state_bar_count,
            fit_end=model._fit_bar_end,
            state_end=model._state_bar_end,
            fit_date=model._fit_date,
        )
        with mock.patch.object(hmm_volume, "infer_state", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                model.refit(None, context(model, bars["end"].iloc[first + 10],
                                          prices, membership))
        self.assertIs(model._fit, before["fit"])
        self.assertEqual(model.state, before["state"])
        np.testing.assert_allclose(model.probabilities, before["probabilities"])
        self.assertEqual(model._fit_bar_count, before["fit_count"])
        self.assertEqual(model._state_bar_count, before["state_count"])
        self.assertEqual(model._fit_bar_end, before["fit_end"])
        self.assertEqual(model.state_date, before["state_end"])
        self.assertEqual(model._fit_date, before["fit_date"])


class StateUpdateChecks(unittest.TestCase):
    """State updates between refits and the risk-on gate."""

    def test_state_advances_between_refits_with_the_last_fitted_model(self):
        model, dates, benchmark, volume, prices, _, membership = build(seed=9)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        fitted = model._fit
        fit_count = model._fit_bar_count
        returns = bars["log_return"].to_numpy(dtype=float)

        for index in range(first + 1, first + model.refit_every_bars):
            signal = bars["end"].iloc[index]
            ctx = context(model, signal, prices, membership)
            self.assertFalse(model.needs_refit(ctx))
            model.propose_buys(ctx)
            self.assertIs(model._fit, fitted, msg="refit happened without a request")
            self.assertEqual(model._fit_bar_count, fit_count)
            self.assertEqual(model._state_bar_count, index + 1)
            self.assertEqual(model.state_date, signal)
            expected_state, expected_probabilities = infer_state(fitted, returns[: index + 1])
            self.assertEqual(model.state, expected_state)
            np.testing.assert_allclose(model.probabilities, expected_probabilities)

    def test_risk_on_gate_uses_the_state_updated_by_propose_buys(self):
        model, dates, benchmark, volume, prices, _, membership = build(seed=10)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))

        next_signal = bars["end"].iloc[first + 1]
        ctx = context(model, next_signal, prices, membership, nav=10_000.0)
        with mock.patch.object(
            hmm_volume, "infer_state",
            return_value=(2, np.array([0.1, 0.2, 0.7])),
        ):
            budgets = model.propose_buys(ctx)
        self.assertEqual(model.state, 2)
        self.assertEqual(model.state_date, next_signal)
        self.assertTrue(budgets)
        self.assertTrue(all(value > 0 for value in budgets.values()))

        following = bars["end"].iloc[first + 2]
        ctx = context(model, following, prices, membership, nav=10_000.0)
        with mock.patch.object(
            hmm_volume, "infer_state",
            return_value=(0, np.array([0.8, 0.15, 0.05])),
        ):
            budgets = model.propose_buys(ctx)
        self.assertEqual(budgets, {}, msg="risk-off state must block purchases")
        self.assertEqual(model.state, 0)
        self.assertEqual(model.state_date, following)

    def test_repeated_propose_buys_on_one_signal_date_is_idempotent(self):
        model, dates, benchmark, volume, prices, _, membership = build(seed=11)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        signal = bars["end"].iloc[first + 2]
        ctx = context(model, signal, prices, membership, nav=50_000.0)

        first_budgets = model.propose_buys(ctx)
        snapshot = (model.state, model.state_date, model._state_bar_count)
        for _ in range(3):
            self.assertEqual(model.propose_buys(ctx), first_budgets)
            self.assertEqual((model.state, model.state_date, model._state_bar_count), snapshot)

        # Same signal date, new cash: only the sizing changes, not the state.
        richer = context(model, signal, prices, membership, nav=200_000.0)
        budgets = model.propose_buys(richer)
        self.assertEqual(set(budgets), set(first_budgets))
        self.assertEqual((model.state, model.state_date, model._state_bar_count), snapshot)

    def test_state_does_not_advance_between_bar_ends(self):
        model, dates, benchmark, volume, prices, _, membership = build(seed=23)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        scored = bars["end"].iloc[first + 1]
        ctx = context(model, scored, prices, membership)
        model.propose_buys(ctx)
        snapshot = (model.state, model.state_date, model._state_bar_count, model.probabilities)
        budgets = model.propose_buys(ctx)

        between = dates[(dates > scored) & (dates < bars["end"].iloc[first + 2])]
        self.assertGreater(len(between), 0)
        for signal in between:
            with self.subTest(signal=signal):
                ctx = context(model, signal, prices, membership)
                self.assertEqual(model.propose_buys(ctx), budgets)
                self.assertEqual(model.state, snapshot[0])
                self.assertEqual(model.state_date, snapshot[1])
                self.assertEqual(model._state_bar_count, snapshot[2])
                np.testing.assert_allclose(model.probabilities, snapshot[3])

    def test_state_label_is_the_emission_mean_rank(self):
        """State ``k`` must be the ``k``-th lowest-mean component after a refit."""
        model, dates, benchmark, volume, prices, _, membership = build(seed=24)
        bars = model._bars
        for index in (model.min_train_bars, model.min_train_bars + 7,
                      model.min_train_bars + 15):
            with self.subTest(bar=index):
                signal = bars["end"].iloc[index]
                model.refit(None, context(model, signal, prices, membership))
                fit = model._fit
                means = fit.model.means_.ravel()
                order = np.argsort(means, kind="stable")
                np.testing.assert_array_equal(fit.state_order, order)
                history = model._bar_returns
                posterior = fit.model.predict_proba(
                    ((history - fit.mean) / fit.scale).reshape(-1, 1)
                )[-1]
                winner = int(np.argmax(posterior))
                expected = int(np.where(order == winner)[0][0])
                self.assertEqual(model.state, expected)
                self.assertEqual(int(np.argmax(model.probabilities)), expected)

    def test_proposals_are_finite_for_degenerate_cash(self):
        model, dates, benchmark, volume, prices, _, membership = build(seed=12)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        signal = bars["end"].iloc[first + 1]
        for cash in (float("nan"), 0.0, -1.0):
            with self.subTest(cash=cash):
                ctx = context(model, signal, prices, membership, cash=cash, nav=cash)
                self.assertEqual(model.propose_buys(ctx), {})
        huge = context(model, signal, prices, membership, cash=1e12, nav=1e12)
        budgets = model.propose_buys(huge)
        self.assertTrue(all(np.isfinite(value) and value > 0 for value in budgets.values()))
        self.assertLessEqual(sum(budgets.values()), 1e12)


class SelectionChecks(unittest.TestCase):
    """top_n and liquidity_window boundaries."""

    def fitted(self, **overrides):
        model, dates, benchmark, volume, prices, _, membership = build(seed=13, **overrides)
        bars = model._bars
        first = model.min_train_bars
        model.refit(None, context(model, bars["end"].iloc[first], prices, membership))
        with mock.patch.object(
            hmm_volume, "infer_state",
            return_value=(2, np.array([0.0, 0.0, 1.0])),
        ):
            model.propose_buys(context(model, bars["end"].iloc[first + 1], prices, membership))
        return model, dates, prices, membership, bars["end"].iloc[first + 2]

    def test_top_n_boundaries(self):
        model, dates, prices, membership, signal = self.fitted(top_n=None, symbols=6)
        everywhere = set(membership.symbol) & set(
            prices.loc[prices.date <= signal, "symbol"]
        )
        ctx = context(model, signal, prices, membership, nav=10_000.0)
        self.assertEqual(set(model.propose_buys(ctx)), everywhere)

        model, dates, prices, membership, signal = self.fitted(top_n=99, symbols=6)
        ctx = context(model, signal, prices, membership, nav=10_000.0)
        self.assertEqual(set(model.propose_buys(ctx)), everywhere)

        model, dates, prices, membership, signal = self.fitted(top_n=1, symbols=6)
        volume = volume_panel(dates, [f"S{i}" for i in range(6)], fill=1.0)
        volume.loc[volume.index == signal, "S4"] = 1e6
        ctx = context(model, signal, prices, membership, nav=10_000.0, volume=volume)
        budgets = model.propose_buys(ctx)
        self.assertEqual(set(budgets), {"S4"})
        self.assertAlmostEqual(sum(budgets.values()), 10_000.0, places=4)

    def test_liquidity_window_changes_the_ranking(self):
        symbols = [f"S{i}" for i in range(6)]
        model, dates, prices, membership, signal = self.fitted(
            top_n=1, liquidity_window=1
        )
        volume = volume_panel(dates, symbols, fill=0.0)
        volume.loc[volume.index < signal, "S1"] = 1e6
        volume.loc[volume.index == signal, "S2"] = 1e6
        ctx = context(model, signal, prices, membership, nav=10_000.0, volume=volume)
        self.assertEqual(set(model.propose_buys(ctx)), {"S2"})

        model, dates, prices, membership, signal = self.fitted(
            top_n=1, liquidity_window=63
        )
        volume = volume_panel(dates, symbols, fill=0.0)
        volume.loc[volume.index < signal, "S1"] = 1e6
        volume.loc[volume.index == signal, "S2"] = 1e6
        ctx = context(model, signal, prices, membership, nav=10_000.0, volume=volume)
        self.assertEqual(set(model.propose_buys(ctx)), {"S1"})

    def test_liquidity_window_longer_than_history_is_clamped(self):
        symbols = [f"S{i}" for i in range(6)]
        same = []
        for window in (10_000, 63):
            model, dates, prices, membership, signal = self.fitted(
                top_n=2, liquidity_window=window
            )
            volume = volume_panel(dates, symbols, fill=1.0)
            volume["S5"] = 50.0
            volume["S4"] = 20.0
            ctx = context(model, signal, prices, membership, nav=10_000.0, volume=volume)
            same.append(set(model.propose_buys(ctx)))
        self.assertEqual(same[0], same[1])
        self.assertEqual(same[0], {"S5", "S4"})

    def test_missing_volume_falls_back_deterministically(self):
        model, dates, prices, membership, signal = self.fitted(top_n=2)
        partial = volume_panel(dates, ["S3"], fill=1.0)
        ctx = context(model, signal, prices, membership, nav=10_000.0, volume=partial)
        first = model.propose_buys(ctx)
        second = model.propose_buys(ctx)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(set(first), {"S3", "S0"})

    def test_symbols_without_a_signal_date_price_are_skipped(self):
        model, dates, prices, membership, signal = self.fitted(top_n=None)
        wide = (
            prices.loc[prices["date"] <= signal]
            .pivot(index="date", columns="symbol", values="adj_close")
        )
        wide.iloc[-1, wide.columns.get_loc("S1")] = np.nan
        wide = wide.drop(columns=["S2"])
        ctx = DecisionContext(
            pd.Timestamp(signal),
            pd.Timestamp(signal) + pd.Timedelta(days=1),
            wide,
            membership.copy(),
            pd.DataFrame(),
            Portfolio(10_000.0, 10_000.0, 10_000.0, {}),
            0,
            pd.DataFrame(),
            pd.DataFrame(),
        )
        budgets = model.propose_buys(ctx)
        self.assertNotIn("S1", budgets)
        self.assertNotIn("S2", budgets)
        self.assertEqual(set(budgets), {"S0", "S3", "S4", "S5"})
        self.assertTrue(all(value > 0 for value in budgets.values()))


class EngineLifecycleChecks(unittest.TestCase):
    """Engine-side refit audit, model versions, and deepcopy behaviour."""

    def test_refit_cadence_and_audit_rows(self):
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=14, periods=520
        )
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=100_000, fee_bps=0)
        result = engine.run(dates[1], dates[-1])

        self.assertFalse(result.refits.empty)
        self.assertTrue(REFIT_COLUMNS <= set(result.refits.columns))
        self.assertTrue(result.refits.status.eq("success").all())
        self.assertTrue(result.refits.rows.eq(0).all())
        for column in ("first_date", "last_date", "latest_label_end", "latest_available"):
            self.assertTrue(result.refits[column].isna().all(), msg=column)

        self.assertEqual(result.refits.model_version.tolist(),
                         list(range(1, len(result.refits) + 1)))
        self.assertEqual(engine.model_version, len(result.refits))

        # Cadence measured in completed bars, from the first eligible bar.
        counts = [bar_count(model, row.signal_date) for row in result.refits.itertuples()]
        self.assertEqual(counts[0], model.min_train_bars + 1)
        self.assertEqual(np.diff(counts).tolist(),
                         [model.refit_every_bars] * (len(counts) - 1))
        for row in result.refits.itertuples():
            self.assertLessEqual(
                model._bars_at(row.signal_date)["end"].max(), row.signal_date
            )

        # The engine's refit phases line up one-for-one with the audit rows.
        refitting = result.phases.loc[result.phases.phase == "refitting", "signal_date"]
        self.assertEqual(sorted(result.refits.signal_date.tolist()), sorted(refitting.tolist()))
        self.assertEqual(list(result.phases.phase).count("refitting"), len(result.refits))

        # Settings snapshot equals the live strategy settings, and the last
        # audit row is the fit that the surviving strategy carries.
        for settings in result.refits.settings:
            self.assertEqual(settings, engine.strategy.refit_settings())
        self.assertEqual(engine.strategy._fit_date, result.refits.signal_date.iloc[-1])
        self.assertEqual(engine.strategy.state_date, engine.strategy._state_bar_end)

    def test_decisions_after_a_refit_see_the_new_model_version(self):
        recorded = []

        class Recorder(HMMVolumeStrategy):
            def propose_buys(self, ctx):
                recorded.append((pd.Timestamp(ctx.signal_date), ctx.model_version))
                return super().propose_buys(ctx)

        dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(
            periods=460, seed=15
        )
        options = dict(DEFAULTS)
        model = Recorder(benchmark, volume, **options)
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        result = engine.run(dates[1], dates[-1])
        # The engine deep-copies the strategy on every refit; the closure list
        # still receives every decision of the surviving copies.
        refit_dates = sorted(result.refits.signal_date.tolist())
        self.assertTrue(refit_dates)
        for signal, version in recorded:
            expected = sum(1 for refit_date in refit_dates if refit_date <= signal)
            self.assertEqual(version, expected, msg=str(signal))
        self.assertEqual(recorded[0][1], 0, msg="first decision must be unfitted")
        self.assertEqual(
            len({version for _signal, version in recorded}), len(refit_dates) + 1
        )

    def test_failed_refit_keeps_the_original_strategy(self):
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=16, periods=420
        )
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        with mock.patch.object(hmm_volume, "fit_hmm", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                engine.run(dates[1], dates[-1])
        self.assertIs(engine.phase, Phase.FAILED)
        self.assertIs(engine.strategy, model)
        self.assertIsNone(model._fit)
        self.assertIsNone(model.state)
        self.assertEqual(engine.model_version, 0)
        self.assertEqual(len(engine.refits), 1)
        self.assertEqual(engine.refits[0]["status"], "failed")
        self.assertEqual(engine.refits[0]["model_version"], 1)
        self.assertEqual(len(engine.oms.orders), 0)
        # The unfitted strategy is still scheduled to fit, not silently skipped.
        first = model.min_train_bars
        self.assertTrue(
            model.needs_refit(context(model, model._bars["end"].iloc[first],
                                      prices, membership))
        )

    def test_refit_cannot_change_strategy_identity(self):
        class Hijacker(HMMVolumeStrategy):
            def refit(self, data, ctx):
                super().refit(data, ctx)
                self.strategy_id = "hijacked"

        dates, benchmark, volume, prices, benchmark_frame, membership = synthetic_market(
            periods=420, seed=17
        )
        options = dict(DEFAULTS)
        model = Hijacker(benchmark, volume, **options)
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        with self.assertRaises(ValueError):
            engine.run(dates[1], dates[-1])
        self.assertIs(engine.phase, Phase.FAILED)
        self.assertIs(engine.strategy, model)
        self.assertEqual(engine.strategy.strategy_id, "hmm_volume")
        self.assertEqual(engine.model_version, 0)
        self.assertEqual(engine.refits[-1]["status"], "failed")

    def test_failed_refit_after_a_success_keeps_the_fitted_state(self):
        """A mid-run failure must not roll back or advance the active model."""
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=26, periods=460
        )
        real_fit = hmm_volume.fit_hmm
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("boom on second fit")
            return real_fit(*args, **kwargs)

        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        with mock.patch.object(hmm_volume, "fit_hmm", side_effect=flaky):
            with self.assertRaises(RuntimeError):
                engine.run(dates[1], dates[-1])

        self.assertIs(engine.phase, Phase.FAILED)
        self.assertEqual(len(engine.refits), 2)
        self.assertEqual([row["status"] for row in engine.refits],
                         ["success", "failed"])
        self.assertEqual(engine.model_version, 1)
        self.assertIsNone(model._fit, msg="the caller's object must stay untrained")
        active = engine.strategy
        self.assertIsNot(active, model)
        self.assertIsNotNone(active._fit)
        self.assertEqual(active._fit_bar_count, bar_count(model, engine.refits[0]["signal_date"]))
        self.assertEqual(active._fit_bar_end, engine.refits[0]["signal_date"])
        # The state stopped at the last bar before the failed refit: the bar that
        # triggered the failure was never scored and the model was not rolled back.
        failed_signal = engine.refits[1]["signal_date"]
        self.assertEqual(active._state_bar_count, bar_count(model, failed_signal) - 1)
        self.assertEqual(active.state_date,
                         model._bars["end"].iloc[active._state_bar_count - 1])
        self.assertLess(active.state_date, failed_signal)
        self.assertEqual([order for order in engine.oms.orders.values()
                          if order.status == "pending"], [])
        # The overdue refit is still scheduled after the failure.
        later = model._bars["end"].iloc[active._fit_bar_count + model.refit_every_bars]
        self.assertTrue(
            active.needs_refit(context(active, later, prices, membership))
        )

    def test_deepcopy_shares_inputs_but_not_model_state(self):
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=18, periods=420
        )
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        engine.run(dates[1], dates[-1])
        active = engine.strategy
        self.assertIsNot(active, model)
        self.assertIsNone(model._fit, msg="the engine must not train the caller's object")
        self.assertIs(active._bars, model._bars)
        self.assertIs(active.benchmark, model.benchmark)
        self.assertIsNotNone(active._fit)

        clone = deepcopy(active)
        self.assertIs(clone._bars, active._bars)
        self.assertIsNot(clone._fit, active._fit)
        self.assertGreater(active._fit_bar_count, 0)
        clone._state = (active.state + 1) % active.n_states
        clone._fit_bar_count = 0
        active_fit_count = active._fit_bar_count
        self.assertEqual(clone._fit_bar_count, 0)
        self.assertEqual(active._fit_bar_count, active_fit_count)
        self.assertEqual(active.state, engine.strategy.state)

    def test_training_panel_does_not_change_a_self_fitting_run(self):
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=19, periods=420
        )
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        plain = engine.run(dates[1], dates[-1])

        panel_dates = dates[:300]
        panel = pd.DataFrame(
            {
                "date": np.repeat(panel_dates, 2),
                "symbol": np.tile(["S0", "S1"], len(panel_dates)),
                "available_date": np.repeat(panel_dates, 2),
                "label_end": np.repeat(panel_dates, 2),
                "target": 0.01,
                "mom": 1.0,
            }
        )
        model2, *_ = build(seed=19, periods=420)
        engine2 = Backtester(prices, benchmark_frame, membership, model2,
                             initial_cash=50_000, fee_bps=0, training_panel=panel,
                             feature_columns=("mom",))
        panelled = engine2.run(dates[1], dates[-1])

        self.assertTrue(panelled.refits.rows.eq(0).all())
        self.assertTrue(panelled.refits.first_date.isna().all())
        self.assertEqual(panelled.refits.signal_date.tolist(),
                         plain.refits.signal_date.tolist())
        self.assertTrue(
            plain.fills.reset_index(drop=True)
            .equals(panelled.fills.reset_index(drop=True))
        )
        np.testing.assert_allclose(plain.history.nav.to_numpy(),
                                   panelled.history.nav.to_numpy())

    def test_short_run_never_fits_and_stays_in_cash(self):
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=20, periods=200, min_train_bars=400
        )
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=10_000, fee_bps=0)
        result = engine.run(dates[1], dates[120])
        self.assertTrue(result.refits.empty)
        self.assertTrue(REFIT_COLUMNS <= set(result.refits.columns))
        self.assertTrue(result.orders.empty)
        self.assertEqual(len(engine.oms.orders), 0)
        self.assertTrue((result.history.cash == 10_000).all())
        self.assertNotIn("refitting", list(result.phases.phase))
        self.assertIsNone(engine.strategy._fit)
        self.assertIsNone(engine.strategy.state)
        self.assertEqual(engine.strategy.state_date, None)
        self.assertEqual(engine.model_version, 0)

    def test_refit_settings_snapshot_is_immune_to_later_mutation(self):
        model, dates, benchmark, volume, prices, benchmark_frame, membership = build(
            seed=21, periods=420
        )
        engine = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=50_000, fee_bps=0)
        result = engine.run(dates[1], dates[-1])
        snapshot = [dict(settings) for settings in result.refits.settings]
        engine.strategy.median_window = 999
        engine.strategy.risk_on_states = (0,)
        engine.strategy.refit_every_bars = 1
        self.assertEqual([dict(settings) for settings in result.refits.settings], snapshot)
        self.assertEqual(snapshot[0]["median_window"], 20)
        self.assertEqual(snapshot[0]["risk_on_states"], [1, 2])
        self.assertEqual(snapshot[0]["refit_every_bars"], 5)
        self.assertIn("model", snapshot[0])


class ValidationBoundaryChecks(unittest.TestCase):
    """Constructor boundaries that the lifecycle depends on."""

    def test_boundary_validation_rejects_off_by_one_and_wrong_types(self):
        _, benchmark, volume, _, _, _ = synthetic_market(periods=520)
        invalid = [
            dict(min_train_bars=19),
            dict(min_train_bars=20.5),
            dict(n_states=1),
            dict(n_states=2.0),
            dict(refit_every_bars=0),
            dict(refit_every_bars=2.5),
            dict(refit_every_bars="5"),
            dict(top_n=0),
            dict(top_n=-1),
            dict(top_n=2.5),
            dict(liquidity_window=0),
            dict(liquidity_window=2.5),
            dict(median_window=1),
            dict(bucket_multiple=float("nan")),
            dict(risk_on_states=(-1,)),
            dict(risk_on_states=(0, 3)),
            dict(risk_on_states=(1, 1)),
            dict(min_order_notional=float("inf")),
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                options = dict(DEFAULTS)
                options.update(overrides)
                with self.assertRaises(ValueError):
                    HMMVolumeStrategy(benchmark, volume, **options)

    def test_min_train_bars_must_be_trainable_for_n_states(self):
        """A configuration that the constructor accepts must be able to fit.

        ``fit_hmm`` needs ``max(20, 5 * n_states)`` observations. With
        ``n_states=5`` and ``min_train_bars=20`` the first scheduled refit has
        only 20 training bars, so it raises and aborts the whole run.
        """
        dates, benchmark, volume, prices, _benchmark_frame, membership = synthetic_market(
            periods=520
        )
        options = dict(DEFAULTS)
        options.update(n_states=5, min_train_bars=20)
        try:
            model = HMMVolumeStrategy(benchmark, volume, **options)
        except ValueError:
            return  # Rejecting the combination up front is the correct fix.
        ctx = context(model, model._bars["end"].iloc[model.min_train_bars],
                      prices, membership)
        self.assertTrue(model.needs_refit(ctx))
        model.refit(None, ctx)

    def test_boolean_parameters_are_rejected(self):
        """``min_order_notional`` rejects bools; the other numerics must too."""
        _, benchmark, volume, _, _, _ = synthetic_market(periods=520)
        cases = [
            ("median_window", True),      # control: rejected today
            ("min_train_bars", True),     # control: rejected today
            ("n_states", True),           # control: rejected today
            ("min_order_notional", True),  # control: rejected today
            ("refit_every_bars", True),
            ("top_n", True),
            ("liquidity_window", True),
            ("n_init", True),
            ("n_iter", True),
            ("seed", True),
            ("bucket_multiple", True),
        ]
        for name, value in cases:
            with self.subTest(parameter=name):
                options = dict(DEFAULTS)
                options[name] = value
                with self.assertRaises(ValueError):
                    HMMVolumeStrategy(benchmark, volume, **options)

    def test_refit_must_not_rewind_to_an_earlier_signal_date(self):
        """A backward refit must be refused, not silently degrade the model."""
        model, dates, _, _, prices, _, membership = build(seed=22)
        bars = model._bars
        later = model.min_train_bars + 40
        model.refit(None, context(model, bars["end"].iloc[later], prices, membership))
        fitted = model._fit
        fit_count = model._fit_bar_count
        state_date = model.state_date
        try:
            model.refit(None, context(model, bars["end"].iloc[model.min_train_bars],
                                      prices, membership))
        except ValueError:
            pass
        self.assertIs(model._fit, fitted, msg="an earlier refit replaced a newer model")
        self.assertEqual(model._fit_bar_count, fit_count)
        self.assertEqual(model.state_date, state_date)

    def test_default_risk_on_states_avoid_only_the_lowest_mean_state(self):
        """The documented default must work for every ``n_states``."""
        _, benchmark, volume, _, _, _ = synthetic_market(periods=520)
        for n_states in (2, 3, 5):
            with self.subTest(n_states=n_states):
                options = dict(DEFAULTS)
                options.pop("risk_on_states")
                options["n_states"] = n_states
                options["min_train_bars"] = max(
                    options.get("min_train_bars", 20), 5 * n_states
                )
                model = HMMVolumeStrategy(benchmark, volume, **options)
                self.assertEqual(model.risk_on_states, tuple(range(1, n_states)))


if __name__ == "__main__":
    unittest.main()
