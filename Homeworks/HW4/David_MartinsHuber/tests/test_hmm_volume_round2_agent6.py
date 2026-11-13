"""Round-2 adversarial numerical/statistical tests for the volume-bar HMM.

The target is ``strategies/hmm_volume.py`` plus the two modules it depends on,
``research/bars.py`` and ``research/hmm.py``.  Every fixture here is synthetic
and deterministic: no database, no random stream (the one place where a real
market-like return sample is needed it is a hard-coded literal, so the
regression evidence cannot drift when numpy changes its generator).

Two defects found by this round are pinned by three tests in
:class:`DefectsRound2`; both are fixed in production and the tests remain as
regression guards:

1. ``test_fit_survives_one_large_most_recent_bar`` and
   ``test_a_walk_forward_run_survives_a_crisis_bar`` -
   ``research/hmm.py:100`` fits a ``GaussianHMM`` inside a per-init
   ``try/except``.  When the *most recent* training observation is a single
   isolated large move (>= ~10 sigma in a 30-observation sample) one component
   ends up with no observed outgoing transition, so ``model.fit`` leaves an
   all-zero ``transmat_`` row and ``model.score`` (``research/hmm.py:101``)
   raises ``ValueError: transmat_ rows must sum to 1 (got row sums of
   [1. 0. 1.])``.  The failure is a property of the data, not of the seed, so
   every restart is skipped and ``fit_hmm`` raises
   ``RuntimeError('HMM fitting failed for every initialization')``
   (``research/hmm.py:109``).  ``HMMVolumeStrategy.refit``
   (``strategies/hmm_volume.py:267``) therefore raises and
   ``Backtester._refit`` (``backtesting/engine.py:198-207``) stops the whole
   walk-forward run.  The identical sample fits when the same bar is not the
   last training observation, so one ordinary crisis bar can kill a run whose
   refit cadence happens to land on it.

2. ``test_a_boundary_basket_decision_terminates`` - the compensation loop at
   ``strategies/hmm_volume.py:82-84`` shrinks the largest budget one *ulp of
   that budget* at a time.  For a wide universe one element's ulp is far below
   one ulp of the total, the recomputed total does not fall, and
   ``propose_buys`` never returns (measured: minutes, with the excess over
   ``usable`` *growing*).
"""

import unittest
from unittest.mock import patch
import warnings
import signal as signal_module

import numpy as np
import pandas as pd

# hmmlearn triggers numpy deprecations on every fit; they are unrelated to the
# behaviour under test and would drown the run output.
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")

from backtesting import Backtester, DecisionContext, Portfolio
from research.bars import volume_bars
from research.hmm import fit_hmm, infer_state
from strategies.hmm_volume import HMMVolumeStrategy, _scale_to_cash

MEDIAN_WINDOW = 20
BUCKET_MULTIPLE = 5.0
BAR_DAYS = 5           # 5 x the median volume of a constant-volume series
START = "2010-01-04"

# A hard-coded 60-observation sample of ~0.9% bar returns.  The last value is
# used by the defect tests: a single 12% move (a 2008/2020 crisis week) placed
# as the most recent training observation.
SAMPLE = np.array([
    -0.00802, -0.01324, -0.00248, 0.00420, 0.01136, 0.00110, -0.00553,
    -0.00785, 0.00749, 0.01635, 0.00273, -0.01233, -0.00958, 0.01600,
    0.00203, -0.01732, -0.00084, -0.01163, -0.00629, -0.00488, -0.00713,
    0.00553, -0.00063, -0.00589, 0.00410, 0.00830, -0.01643, -0.00257,
    -0.00981, -0.00173, -0.01289, 0.00021, -0.00038, -0.00304, -0.01048,
    -0.00396, -0.01091, -0.01355, 0.00225, -0.01109, 0.01170, 0.00717,
    -0.01998, 0.00272, -0.01102, 0.00033, 0.00044, -0.01988, -0.00233,
    -0.00256, 0.00962, -0.01181, 0.00738, -0.01099, -0.00331, -0.00840,
    0.01449, 0.00568, 0.02432, 0.00642,
])
CRISIS = 0.12


# --------------------------------------------------------------- fixtures
def price_path(returns, start_price=100.0, median_window=MEDIAN_WINDOW):
    """A daily price series whose volume bars reproduce ``returns`` exactly.

    With a constant volume panel the bar rule makes every bucket exactly
    ``BAR_DAYS`` sessions wide, so the path is built so that
    ``log(p[end] / p[start - 1])`` equals the requested return.  The flat
    warm-up prefix fills the trailing-median window, and three trailing
    sessions are appended so the last bucket completes.
    """
    prices = [start_price] * (median_window - 1)
    price = start_price
    for value in returns:
        for step in range(1, BAR_DAYS + 1):
            prices.append(price * float(np.exp(value * step / BAR_DAYS)))
        price = prices[-1]
    prices.extend([price] * 3)
    dates = pd.bdate_range(START, periods=len(prices))
    return pd.Series(prices, index=dates, name="benchmark")


def volume_path(index, value=1e6):
    return pd.Series(value, index=index, name="volume")


def build(returns, *, volume_value=1e6, **overrides):
    """A strategy whose bars reproduce ``returns``; returns it with the data."""
    prices = price_path(np.asarray(returns, dtype=float))
    volume = volume_path(prices.index, volume_value)
    options = dict(
        median_window=MEDIAN_WINDOW,
        bucket_multiple=BUCKET_MULTIPLE,
        n_states=3,
        min_train_bars=30,
        refit_every_bars=1,
        n_init=1,
        n_iter=100,
        top_n=2,
    )
    options.update(overrides)
    model = HMMVolumeStrategy(prices, volume, **options)
    return model, prices, volume


def members(signal, symbols):
    signal = pd.Timestamp(signal)
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "quarter_start": [signal.to_period("Q").start_time] * len(symbols),
            "selection_date": [signal - pd.Timedelta(days=1)] * len(symbols),
        }
    )


def context(model, signal, prices, symbols=("AAA", "BBB"), *, nav=100_000.0,
            cash=None, price=10.0, volume=None):
    """A one-row DecisionContext as of ``signal``."""
    signal = pd.Timestamp(signal)
    if price is None:
        row = {symbol: np.nan for symbol in symbols}
    else:
        row = {symbol: float(price) for symbol in symbols}
    price_frame = pd.DataFrame([row], index=[signal])
    maintain = members(signal, symbols)
    available = nav if cash is None else cash
    return DecisionContext(
        signal,
        signal + pd.Timedelta(days=1),
        price_frame,
        maintain,
        pd.DataFrame(),
        Portfolio(nav, available, nav, {}),
        0,
        pd.DataFrame() if volume is None else volume,
        pd.DataFrame(),
    )


def stateful(model, state):
    """Pin the HMM state so propose_buys can be tested without a real fit."""
    model._fit = object()          # a fitted model has already been produced
    model._state = state
    model._state_bar_count = len(model._bars)
    model._state_bar_end = model._bars["end"].iloc[-1]
    return model


class StrategyCase(unittest.TestCase):
    """Shared helpers for proposition-level assertions."""

    def _liquidity_model(self, top_n=2, liquidity_window=63):
        model, prices, _ = build(SAMPLE, top_n=top_n,
                                 liquidity_window=liquidity_window)
        signal = model.bars["end"].iloc[-1]
        return stateful(model, 2), prices, signal

    def assert_valid_budget(self, proposals, cash, floor, selected=None):
        self.assertTrue(proposals, "expected at least one order")
        total = sum(proposals.values())
        self.assertTrue(np.isfinite(total))
        self.assertLessEqual(total, cash)
        for symbol, value in proposals.items():
            self.assertIs(type(symbol), str)
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, floor)
        if selected is not None:
            self.assertEqual(set(proposals), set(selected))

    def assert_valid_state(self, model):
        self.assertIn(model.state, range(model.n_states))
        probabilities = model.probabilities
        self.assertEqual(len(probabilities), model.n_states)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all() and (probabilities <= 1).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=9)
        self.assertEqual(int(np.argmax(probabilities)), model.state)


# ------------------------------------------------------- bar construction
class BarRuleEdges(StrategyCase):
    """The documented volume rule: 5 x the trailing median, minimal buckets."""

    def test_constant_volume_bars_are_exactly_the_ceiling_of_the_multiple(self):
        """Size buckets reach the target on the last session they add."""
        prices = price_path(np.zeros(60))
        n_sessions = len(prices)
        for multiple, days in [(1.0, 1), (2.0, 2), (2.5, 3), (3.0, 3), (5.0, 5),
                               (7.0, 7)]:
            with self.subTest(bucket_multiple=multiple):
                model = HMMVolumeStrategy(
                    prices, volume_path(prices.index),
                    median_window=MEDIAN_WINDOW, bucket_multiple=multiple,
                )
                bars = model.bars
                self.assertEqual(set(bars["days"]), {days})
                expected = (n_sessions - (MEDIAN_WINDOW - 1)) // days
                self.assertEqual(len(bars), expected)
                # The strategy's bars are the documented rule, unmodified.
                direct = volume_bars(prices, volume_path(prices.index),
                                     median_window=MEDIAN_WINDOW,
                                     bucket_multiple=multiple).bars
                self.assertTrue(bars.equals(direct))

    def test_a_bucket_below_daily_size_reproduces_the_daily_sessions(self):
        """A target under one session's volume gives one-session bars."""
        rng = np.random.default_rng(0)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
            index=pd.bdate_range(START, periods=400),
        )
        model = HMMVolumeStrategy(prices, volume_path(prices.index),
                                  median_window=MEDIAN_WINDOW,
                                  bucket_multiple=1e-12)
        bars = model.bars
        self.assertEqual(set(bars["days"]), {1})
        daily = np.log(prices / prices.shift(1))
        np.testing.assert_allclose(bars["log_return"].to_numpy(),
                                   daily.loc[bars["start"]].to_numpy(),
                                   rtol=0, atol=1e-12)

    def test_volume_spikes_and_zero_stretches_keep_every_bar_rule(self):
        """Spikes shorten a bucket, zeros leave the median at zero."""
        rng = np.random.default_rng(7)
        sessions = pd.bdate_range(START, periods=400)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=sessions
        )
        volume = pd.Series(rng.lognormal(15, 0.3, 400), index=sessions)
        volume.iloc[100] = 1e12          # a single-day volume spike
        volume.iloc[150:220] = 0.0       # a 70-session zero-volume stretch
        bars = volume_bars(prices, volume,
                           median_window=MEDIAN_WINDOW,
                           bucket_multiple=BUCKET_MULTIPLE).bars

        np.testing.assert_allclose(bars["target"].to_numpy(),
                                   5.0 * bars["median_volume"].to_numpy())
        # A bucket stops on the first session that reaches the target ...
        self.assertTrue((bars["bar_volume"] >= bars["target"] - 1e-6).all())
        last_volume = volume.loc[bars["end"]].to_numpy()
        # ... and could not have stopped one session earlier.
        self.assertTrue(
            (bars["bar_volume"].to_numpy() - last_volume
             < bars["target"].to_numpy() + 1e-6).all()
        )
        # Documented return definition: close of the bucket over the close of
        # the session before the bucket starts.
        np.testing.assert_allclose(
            bars["log_return"].to_numpy(),
            np.log(bars["close_price"].to_numpy() / bars["base_price"].to_numpy()),
        )
        self.assertTrue(pd.Series(
            [prices.iloc[pos - 1] for pos in bars["start_pos"]]
        ).eq(bars["base_price"].to_numpy()).all())
        self.assertTrue((bars["days"].to_numpy()
                         == bars["end_pos"].to_numpy()
                         - bars["start_pos"].to_numpy() + 1).all())
        # No bucket starts while the trailing median is zero (there is no
        # positive target to fill).
        self.assertTrue((bars["median_volume"] > 0).all())
        zero_median = pd.Series(0.0, index=sessions).loc[
            volume.rolling(MEDIAN_WINDOW, min_periods=MEDIAN_WINDOW).median() == 0
        ].index
        self.assertFalse(set(bars["start"]) & set(zero_median))

    def test_a_volume_spike_on_a_bucket_start_is_a_one_session_bucket(self):
        """The first session of a bucket already exceeds 5 x the median."""
        sessions = pd.bdate_range(START, periods=200)
        rng = np.random.default_rng(5)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=sessions
        )
        volume = pd.Series(1e6, index=sessions)
        volume.iloc[24] = 1e12          # session 24 starts the second bucket
        bars = volume_bars(prices, volume, median_window=MEDIAN_WINDOW,
                           bucket_multiple=BUCKET_MULTIPLE).bars
        # 1 five-session bucket (19-23), the spike bucket (24), then one
        # five-session bucket for every session from 25 on.
        self.assertEqual(len(bars), 1 + 1 + (200 - 25) // BAR_DAYS)
        spike = bars.loc[bars["start_pos"] == 24].iloc[0]
        self.assertEqual(spike["days"], 1)
        self.assertEqual(spike["end_pos"], 24)
        self.assertGreaterEqual(spike["bar_volume"], spike["target"])
        # The spike does not move the robust median target.
        self.assertAlmostEqual(spike["target"], 5e6)
        # The next bucket still starts at the following session and its base
        # price is the spike session's close.
        following = bars.loc[bars["start_pos"] == 25].iloc[0]
        self.assertEqual(following["base_pos"], 24)

    def test_zero_volume_stretch_cannot_make_a_bar_with_a_zero_target(self):
        """An all-zero volume panel has no positive target at all."""
        prices = price_path(np.zeros(60))
        with self.assertRaisesRegex(ValueError, "no completed bars"):
            HMMVolumeStrategy(prices, volume_path(prices.index, 0.0),
                              median_window=MEDIAN_WINDOW)

    def test_huge_finite_volume_stays_finite_and_overflow_fails_closed(self):
        prices = price_path(np.zeros(60))
        model = HMMVolumeStrategy(prices, volume_path(prices.index, 1e300),
                                  median_window=MEDIAN_WINDOW)
        self.assertTrue(np.isfinite(model.bars["bar_volume"]).all())
        self.assertTrue(np.isfinite(model.bars["log_return"]).all())
        # 5 * 1e300 is representable but 5 * 1.6e308 is not, so no positive
        # target can exist; the constructor must refuse instead of guessing.
        with self.assertRaisesRegex(ValueError, "no completed bars"):
            HMMVolumeStrategy(prices, volume_path(prices.index, 1.6e308),
                              median_window=MEDIAN_WINDOW)


# --------------------------------------------------- inputs and date edges
class InputEdges(StrategyCase):
    def test_disjoint_ranges_are_rejected_and_partial_overlap_is_intersected(self):
        benchmark_index = pd.bdate_range(START, periods=120)
        volume_index = pd.bdate_range("2015-01-01", periods=120)
        prices = pd.Series(100.0, index=benchmark_index)
        with self.assertRaisesRegex(ValueError, "no common dates"):
            HMMVolumeStrategy(prices, pd.Series(1e6, index=volume_index))

        # Half-overlapping ranges: only the shared sessions may be used.
        volume_index = pd.bdate_range(START, periods=120)[60:]
        model = HMMVolumeStrategy(
            prices, pd.Series(1e6, index=volume_index),
            median_window=MEDIAN_WINDOW,
        )
        shared = benchmark_index[60:]
        self.assertTrue(model.benchmark.index.equals(shared))
        self.assertTrue(model.volume.index.equals(shared))
        self.assertTrue(model.bars["end"].le(shared[-1]).all())
        self.assertTrue(model.bars["start"].ge(shared[0]).all())
        self.assertTrue(set(model.bars["start"]) <= set(shared))
        self.assertTrue(set(model.bars["end"]) <= set(shared))

    def test_missing_sessions_are_aggregated_over_supplied_sessions_only(self):
        sessions = pd.bdate_range(START, periods=300)
        rng = np.random.default_rng(3)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=sessions
        )
        volume = pd.Series(1e6, index=sessions)
        # Drop every seventh session from both inputs.
        keep = sessions[np.arange(len(sessions)) % 7 != 0]
        model = HMMVolumeStrategy(prices.loc[keep], volume.loc[keep],
                                  median_window=MEDIAN_WINDOW)
        self.assertTrue(model.benchmark.index.equals(keep))
        self.assertTrue(set(model.bars["start"]) <= set(keep))
        self.assertFalse(set(model.bars["start"]) & set(sessions.difference(keep)))

    def test_duplicate_dates_keep_the_last_row_and_are_sorted(self):
        prices = price_path(np.zeros(40))
        duplicated = pd.concat(
            [prices, pd.Series([999.0], index=[prices.index[25]])]
        )
        model = HMMVolumeStrategy(duplicated, volume_path(prices.index),
                                  median_window=MEDIAN_WINDOW)
        self.assertEqual(len(model.benchmark), len(prices))
        self.assertEqual(float(model.benchmark.loc[prices.index[25]]), 999.0)
        self.assertTrue(model.benchmark.index.is_monotonic_increasing)

    def test_reversed_and_shuffled_inputs_build_the_same_bars(self):
        rng = np.random.default_rng(11)
        sessions = pd.bdate_range(START, periods=400)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=sessions
        )
        volume = pd.Series(rng.lognormal(15, 0.4, 400), index=sessions)
        reference = HMMVolumeStrategy(prices, volume,
                                      median_window=MEDIAN_WINDOW).bars
        shuffled = np.random.default_rng(2).permutation(len(sessions))
        other = HMMVolumeStrategy(prices.iloc[shuffled], volume.iloc[shuffled],
                                  median_window=MEDIAN_WINDOW)
        self.assertTrue(reference.equals(other.bars))

    def test_hostile_date_and_value_columns_have_documented_errors(self):
        prices = price_path(np.zeros(40))
        volume = volume_path(prices.index)
        cases = {
            "timezone-aware": prices.tz_localize("UTC"),
            "nonfinite": prices.where(prices.index != prices.index[5], np.inf),
            "nonpositive": prices.where(prices.index != prices.index[5], 0.0),
        }
        messages = {
            "timezone-aware": "timezone-naive",
            "nonfinite": "nonempty finite",
            "nonpositive": "strictly positive",
        }
        for label, series in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, messages[label]):
                    HMMVolumeStrategy(series, volume, median_window=MEDIAN_WINDOW)
        bad_index = prices.index.to_numpy().astype(object)
        bad_index[3] = None
        with self.assertRaisesRegex(ValueError, "nonempty finite|nonmissing"):
            HMMVolumeStrategy(pd.Series(prices.to_numpy(), index=bad_index),
                              volume, median_window=MEDIAN_WINDOW)
        negatives = volume.copy()
        negatives.iloc[10] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            HMMVolumeStrategy(prices, negatives, median_window=MEDIAN_WINDOW)
        # Zero volume is explicitly allowed by the contract.
        zeros = volume.copy()
        zeros.iloc[10] = 0.0
        HMMVolumeStrategy(prices, zeros, median_window=MEDIAN_WINDOW)


# --------------------------------------------------------- HMM statistics
class HMMStatisticalEdges(StrategyCase):
    def test_a_zero_variance_segment_keeps_the_model_valid(self):
        """A run of identical closes yields exact zero bar returns, no NaN."""
        returns = SAMPLE.copy()
        returns[10:20] = 0.0            # ten perfectly flat buckets
        returns[40:50] = 0.0
        model, prices, _ = build(returns)
        bars = model.bars
        np.testing.assert_array_equal(bars["log_return"].to_numpy()[10:20], 0.0)
        np.testing.assert_array_equal(bars["log_return"].to_numpy()[40:50], 0.0)
        model.refit(None, context(model, bars["end"].iloc[30], prices))
        self.assert_valid_state(model)

    def test_the_model_is_invariant_to_return_scale_and_location(self):
        """``fit_hmm`` standardises with the training sample only."""
        rng = np.random.default_rng(4)
        x = rng.normal(0.001, 0.01, 260)
        base = fit_hmm(x[:-1], n_states=3, n_init=1, seed=7)
        base_state, base_probabilities = infer_state(base, x)
        for label, transformed in [
            ("x1e6", x * 1e6), ("x1e-6", x * 1e-6), ("x1e3", x * 1e3),
            ("plus 0.5", x + 0.5), ("minus one percent", x - 0.01),
        ]:
            with self.subTest(label):
                fit = fit_hmm(transformed[:-1], n_states=3, n_init=1, seed=7)
                state, probabilities = infer_state(fit, transformed)
                self.assertEqual(state, base_state)
                np.testing.assert_allclose(probabilities, base_probabilities,
                                           atol=1e-9)

    def test_state_labels_are_relabelled_by_mean_at_every_refit(self):
        """Documented: labels are comparable across refits, 0 = lowest mean."""
        model, prices, _ = build(SAMPLE)
        bars = model.bars
        for count in (31, 35, 40, 50):
            with self.subTest(bars=count):
                model.refit(None, context(model, bars["end"].iloc[count], prices))
                fit = model._fit
                means = fit.model.means_.ravel()
                order = fit.state_order
                self.assertTrue(np.all(np.diff(means[order]) >= 0))
                self.assertEqual(sorted(order.tolist()), list(range(model.n_states)))
                standardized = ((model._bar_returns - fit.mean) / fit.scale)
                raw = fit.model.predict_proba(
                    standardized.reshape(-1, 1))[-1]
                np.testing.assert_allclose(model.probabilities, raw[order])
                self.assert_valid_state(model)

    def test_states_are_relabelled_by_emission_mean(self):
        """State ``k`` is the component with the ``k``-th lowest mean."""
        rng = np.random.default_rng(6)
        returns = rng.normal(0, 0.01, 300)
        fit = fit_hmm(returns[:-1], n_states=3, n_init=1, seed=7)
        order = fit.state_order
        means = fit.model.means_.ravel()
        self.assertTrue(np.all(np.diff(means[order]) >= 0))
        state, probabilities = infer_state(fit, returns)
        standardized = ((returns - fit.mean) / fit.scale).reshape(-1, 1)
        raw = fit.model.predict_proba(standardized)[-1]
        np.testing.assert_allclose(probabilities, raw[order])
        self.assertEqual(state, int(np.argmax(raw[order])))

    def test_extreme_bars_still_produce_a_proper_distribution(self):
        """A 700-sigma bar must not yield NaN, inf, or an out-of-range state."""
        rng = np.random.default_rng(8)
        training = rng.normal(0, 0.01, 250)
        fit = fit_hmm(training, n_states=3, n_init=1, seed=7)
        for shock in [0.0, 0.05, 5.0, 700.0, -700.0, 1e6, -1e6]:
            with self.subTest(shock=shock):
                history = np.r_[training[:50], [shock]]
                state, probabilities = infer_state(fit, history)
                self.assertIn(state, range(3))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=9)

    def test_minimal_expected_maximisation_settings_are_supported(self):
        """``n_init`` and ``n_iter`` minima must still deliver a valid model."""
        model, prices, _ = build(SAMPLE, n_init=1, n_iter=1)
        bars = model.bars
        model.refit(None, context(model, bars["end"].iloc[30], prices))
        self.assert_valid_state(model)

    def test_state_count_and_minimum_sample_boundaries_agree(self):
        """A constructor-accepted pair must actually be fittable."""
        prices = price_path(SAMPLE)
        for n_states in [2, 5, 8]:
            with self.subTest(n_states=n_states):
                minimum = max(20, 5 * n_states)
                with self.assertRaises(ValueError):
                    HMMVolumeStrategy(prices, volume_path(prices.index),
                                      n_states=n_states,
                                      min_train_bars=minimum - 1)
                model, prices_n, _ = build(SAMPLE, n_states=n_states,
                                           min_train_bars=minimum)
                bars = model.bars
                model.refit(None,
                            context(model, bars["end"].iloc[minimum], prices_n))
                self.assert_valid_state(model)
                self.assertEqual(len(model.probabilities), n_states)
                self.assertEqual(model.state, int(np.argmax(model.probabilities)))

    def test_short_and_long_samples_behave_as_documented(self):
        # median_window sessions give a median but no complete bucket.
        short = price_path(np.zeros(3))[:MEDIAN_WINDOW + 2]
        with self.assertRaisesRegex(ValueError, "no completed bars"):
            HMMVolumeStrategy(pd.Series(short.to_numpy(), index=short.index),
                              volume_path(short.index),
                              median_window=MEDIAN_WINDOW)
        # The first fit needs min_train_bars + 1 completed bars.
        model, prices, _ = build(SAMPLE, min_train_bars=30)
        bars = model.bars
        self.assertFalse(model.needs_refit(
            context(model, bars["end"].iloc[29], prices)))
        self.assertTrue(model.needs_refit(
            context(model, bars["end"].iloc[30], prices)))
        with self.assertRaisesRegex(ValueError, "Not enough completed bars"):
            model.refit(None, context(model, bars["end"].iloc[29], prices))
        # A long sample: ~600 buckets still produce a valid walk-forward state.
        long_returns = np.tile(SAMPLE, 10)
        model, prices, _ = build(long_returns, min_train_bars=250,
                                 n_init=1, n_iter=50)
        self.assertEqual(len(model.bars), 600)
        model.refit(None, context(model, model.bars["end"].iloc[300], prices))
        self.assert_valid_state(model)

    def test_a_failed_refit_does_not_activate_partial_model_state(self):
        """A failed fit must leave the previous model exactly as it was."""
        returns = SAMPLE.copy()
        returns[30] = CRISIS            # scored as the 31st bucket, trained on
        model, prices, _ = build(returns)   # only at the following refit
        bars = model.bars
        model.refit(None, context(model, bars["end"].iloc[30], prices))
        self.assert_valid_state(model)
        previous = (model.state, model.probabilities.copy(), model.state_date,
                    model._fit_date, model._fit_bar_count,
                    model._bar_returns.copy(), model._fit)

        with patch("strategies.hmm_volume.fit_hmm",
                   side_effect=RuntimeError("forced failure")):
            with self.assertRaises(RuntimeError):
                model.refit(None, context(model, bars["end"].iloc[31], prices))

        self.assertEqual(model.state, previous[0])
        np.testing.assert_array_equal(model.probabilities, previous[1])
        self.assertEqual(model.state_date, previous[2])
        self.assertEqual(model._fit_date, previous[3])
        self.assertEqual(model._fit_bar_count, previous[4])
        np.testing.assert_array_equal(model._bar_returns, previous[5])
        self.assertIs(model._fit, previous[6])


# --------------------------------------------- universe, liquidity, cash
class UniverseAndCashEdges(StrategyCase):
    def test_top_n_boundaries_select_the_documented_universe(self):
        symbols = ("AAA", "BBB", "CCC", "DDD")
        expectations = {1: ["BBB"], 2: ["BBB", "DDD"],
                        4: ["AAA", "BBB", "CCC", "DDD"]}
        for top_n, expected in expectations.items():
            with self.subTest(top_n=top_n):
                model, prices, signal = self._liquidity_model(top_n=top_n)
                ranked_volume = pd.DataFrame(
                    {"AAA": [1.0], "BBB": [4.0], "CCC": [2.0], "DDD": [3.0]},
                    index=[pd.Timestamp(signal)],
                )
                proposals = model.propose_buys(
                    context(model, signal, prices, symbols, nav=1000.0,
                            volume=ranked_volume))
                self.assert_valid_budget(proposals, 1000.0,
                                         model.min_order_notional, expected)
        # top_n larger than the universe, and None, both mean "all names".
        for top_n in (9, None):
            with self.subTest(top_n=top_n):
                model, prices, signal = self._liquidity_model(top_n=top_n)
                ranked_volume = pd.DataFrame(
                    {"AAA": [1.0], "BBB": [4.0], "CCC": [2.0], "DDD": [3.0]},
                    index=[pd.Timestamp(signal)],
                )
                proposals = model.propose_buys(
                    context(model, signal, prices, symbols, nav=1000.0,
                            volume=ranked_volume))
                self.assertEqual(set(proposals), set(symbols))

    def test_missing_or_nan_volume_columns_fall_back_deterministically(self):
        symbols = ("AAA", "BBB", "CCC", "DDD")
        model, prices, signal = self._liquidity_model(top_n=2)
        session = pd.Timestamp(signal)
        all_nan = pd.DataFrame(
            {symbol: [np.nan] for symbol in symbols}, index=[session]
        )
        self.assertEqual(set(model.propose_buys(
            context(model, signal, prices, symbols, nav=100.0,
                    volume=all_nan))), {"AAA", "BBB"})
        partial = pd.DataFrame({"AAA": [1.0], "CCC": [100.0]}, index=[session])
        self.assertEqual(set(model.propose_buys(
            context(model, signal, prices, symbols, nav=100.0,
                    volume=partial))), {"CCC", "AAA"})
        self.assertEqual(set(model.propose_buys(
            context(model, signal, prices, symbols, nav=100.0))),
            {"AAA", "BBB"})

    def test_liquidity_window_boundaries(self):
        """A one-session window ranks on the signal day only, and a window
        longer than the supplied history is clamped to it."""
        symbols = ("AAA", "BBB", "CCC")

        def request(liquidity_window):
            model, prices, signal = self._liquidity_model(
                top_n=1, liquidity_window=liquidity_window)
            signal = pd.Timestamp(signal)
            index = pd.DatetimeIndex(
                [signal - pd.Timedelta(days=4), signal - pd.Timedelta(days=2),
                 signal])
            prices_frame = pd.DataFrame(
                {symbol: [10.0, 10.0, 10.0] for symbol in symbols}, index=index)
            panel = pd.DataFrame(
                {"AAA": [1.0, 1.0, 10.0], "BBB": [100.0, 100.0, 1.0],
                 "CCC": [1.0, 1.0, 5.0]}, index=index)
            ctx = DecisionContext(
                signal, signal + pd.Timedelta(days=1), prices_frame,
                members(signal, symbols), pd.DataFrame(),
                Portfolio(100.0, 100.0, 100.0, {}), 0, panel, pd.DataFrame())
            return model.propose_buys(ctx)

        self.assertEqual(set(request(1)), {"AAA"})
        self.assertEqual(set(request(3)), {"BBB"})
        self.assertEqual(request(3), request(10_000))

    def test_symbols_without_a_signal_date_price_are_dropped(self):
        model, prices, signal = self._liquidity_model(top_n=2)
        session = pd.Timestamp(signal)
        price_frame = pd.DataFrame(
            {"AAA": [10.0], "BBB": [np.nan], "CCC": [np.nan]}, index=[session]
        )
        proposals = model.propose_buys(DecisionContext(
            session, session + pd.Timedelta(days=1), price_frame,
            members(session, ("AAA", "BBB", "CCC")), pd.DataFrame(),
            Portfolio(100.0, 100.0, 100.0, {}), 0, pd.DataFrame(),
            pd.DataFrame()))
        self.assertEqual(set(proposals), {"AAA"})
        empty = model.propose_buys(DecisionContext(
            session, session + pd.Timedelta(days=1),
            pd.DataFrame({"AAA": [np.nan], "BBB": [np.nan]}, index=[session]),
            members(session, ("AAA", "BBB")), pd.DataFrame(),
            Portfolio(100.0, 100.0, 100.0, {}), 0, pd.DataFrame(),
            pd.DataFrame()))
        self.assertEqual(empty, {})

    def test_min_order_notional_boundary_is_exact(self):
        """``cash == n * floor`` is documented to buy nothing; the buffer is 1e-6."""
        model, prices, signal = self._liquidity_model(top_n=3)
        symbols = ("AAA", "BBB", "CCC")
        for cash, expected in [(3.0, 0), (3.0 + 5e-7, 0), (3.0 + 1e-6, 3),
                               (3.0 + 1e-3, 3)]:
            with self.subTest(cash=cash):
                proposals = model.propose_buys(
                    context(model, signal, prices, symbols, nav=cash))
                self.assertEqual(len(proposals), expected)
                if expected:
                    self.assert_valid_budget(proposals, cash, 1.0)

    def test_a_single_name_universe_receives_the_whole_buffer_budget(self):
        model, prices, signal = self._liquidity_model(top_n=None)
        proposals = model.propose_buys(
            context(model, signal, prices, ("ONLY",), nav=1000.0))
        self.assertEqual(list(proposals), ["ONLY"])
        self.assertEqual(model.min_order_notional, 1.0)
        self.assertLess(proposals["ONLY"], 1000.0)
        self.assertGreater(proposals["ONLY"], 999.998)
        self.assert_valid_budget(proposals, 1000.0, model.min_order_notional)

    def test_huge_and_tiny_accounts(self):
        symbols = tuple(f"T{i:04d}" for i in range(2000))
        model, prices, signal = self._liquidity_model(top_n=None)
        big = model.propose_buys(
            context(model, signal, prices, symbols, nav=1e6))
        self.assertEqual(len(big), 2000)
        self.assert_valid_budget(big, 1e6, model.min_order_notional)
        self.assertTrue(np.allclose(list(big.values()),
                                    list(big.values())[0], rtol=0, atol=0))
        # Below n * floor the default account cannot buy a single name.
        for nav in (1000.0, 2000.0):
            with self.subTest(nav=nav):
                self.assertEqual(model.propose_buys(
                    context(model, signal, prices, symbols, nav=nav)), {})
        # A boundary basket that is large enough per name is allocated in full.
        boundary = model.propose_buys(
            context(model, signal, prices, symbols, nav=2000.5))
        self.assertEqual(len(boundary), 2000)
        self.assert_valid_budget(boundary, 2000.5, model.min_order_notional)
        huge = model.propose_buys(
            context(model, signal, prices, symbols, nav=1e300))
        self.assertEqual(len(huge), 2000)
        self.assertTrue(np.isfinite(sum(huge.values())))
        self.assertLessEqual(sum(huge.values()), 1e300)

    def test_risk_on_state_defaults_and_explicit_subsets(self):
        prices = price_path(SAMPLE)
        volume = volume_path(prices.index)
        defaults = {2: (1,), 3: (1, 2), 4: (1, 2, 3), 5: (1, 2, 3, 4)}
        for n_states, expected in defaults.items():
            with self.subTest(n_states=n_states):
                model = HMMVolumeStrategy(prices, volume,
                                          n_states=n_states,
                                          min_train_bars=max(20, 5 * n_states))
                self.assertEqual(model.risk_on_states, expected)
        # An explicit crisis-only gate buys in state 0 and nowhere else.
        model, prices, signal = self._liquidity_model(top_n=2)
        model.risk_on_states = (0,)
        stateful(model, 0)
        self.assertEqual(len(model.propose_buys(
            context(model, signal, prices, ("AAA", "BBB"), nav=100.0))), 2)
        for other in (1, 2):
            with self.subTest(state=other):
                model._state = other
                self.assertEqual(model.propose_buys(
                    context(model, signal, prices, ("AAA", "BBB"),
                            nav=100.0)), {})

    def test_scale_to_cash_never_overspends_or_breaks_the_floor(self):
        rng = np.random.default_rng(13)
        values = [0.0, 5e-324, 1e-320, 1e-12, 0.5, 1.0, 3.0, 1e6, 1e150,
                  1e300, 1e308, np.finfo(float).max, np.inf, np.nan, -5.0]
        cashes = [1e-320, 1e-12, 1e-6, 1e-3, 0.5, 1.0, 2.5, 3.0, 1e6, 1e12,
                  1e300, 0.0, -1.0, np.inf, np.nan]
        for _ in range(2000):
            cash = float(cashes[int(rng.integers(len(cashes)))])
            floor = float([0.0, 1e-6, 0.5, 1.0, 1e6][int(rng.integers(5))])
            budgets = {f"S{i}": float(values[int(rng.integers(len(values)))])
                       for i in range(int(rng.integers(1, 9)))}
            proposals = _scale_to_cash(dict(budgets), cash, min_notional=floor)
            if not np.isfinite(cash) or cash <= 0:
                self.assertEqual(proposals, {})
                continue
            for symbol, value in proposals.items():
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, floor)
            self.assertLessEqual(sum(proposals.values()), cash)

    def test_decisions_are_reproducible_and_idempotent(self):
        """Documented: identical inputs and settings give identical decisions.

        The fitted parameters are reproducible but not bit-for-bit: hmmlearn's
        forward/backward and EM reductions differ in the last ~1e-16 between
        two fits of the same array in the same process (verified by diffing
        ``log_likelihood`` across repeated ``fit_hmm`` calls), so the posterior
        is compared with a tolerance while the *decisions* must match exactly.
        """
        first, prices, _ = build(SAMPLE)
        second, _, _ = build(SAMPLE)
        self.assertTrue(first.bars.equals(second.bars))
        signal = first.bars["end"].iloc[40]
        first.refit(None, context(first, signal, prices))
        second.refit(None, context(second, signal, prices))
        self.assertEqual(first.state, second.state)
        self.assertEqual(first._fit_bar_count, second._fit_bar_count)
        self.assertEqual(first.state_date, second.state_date)
        np.testing.assert_allclose(first.probabilities, second.probabilities,
                                   rtol=0, atol=1e-12)
        request = context(first, signal, prices, ("AAA", "BBB"), nav=1000.0)
        self.assertEqual(first.propose_buys(request),
                         first.propose_buys(request))
        self.assertEqual(first.propose_buys(request), second.propose_buys(request))


# -------------------------------------------------- engine-level fixtures
def market(returns, symbols=("AAA", "BBB", "CCC")):
    """A deterministic three-name market on the bar fixture's calendar."""
    benchmark = price_path(np.asarray(returns, dtype=float))
    dates = benchmark.index
    rows = []
    for index, symbol in enumerate(symbols):
        wave = 1.0 + 0.02 * np.sin(
            np.arange(len(dates)) * (0.3 + 0.1 * index) + index
        )
        rows.append(pd.DataFrame({
            "date": dates, "symbol": symbol,
            "adj_close": benchmark.to_numpy() * wave,
        }))
    prices = pd.concat(rows, ignore_index=True)
    benchmark_frame = pd.DataFrame({"date": dates,
                                    "adj_close": benchmark.to_numpy()})
    quarters = pd.date_range(dates[0], dates[-1], freq="QS")
    membership = pd.concat([
        pd.DataFrame({
            "symbol": list(symbols),
            "quarter_start": quarter,
            "selection_date": quarter - pd.Timedelta(days=1),
        })
        for quarter in quarters
    ], ignore_index=True)
    return prices, benchmark_frame, membership, benchmark, dates


class EngineNumericalEdges(StrategyCase):
    def test_flat_benchmark_run_completes_without_a_single_bar_return(self):
        prices, benchmark_frame, membership, benchmark, dates = market(
            np.zeros(60))
        model = HMMVolumeStrategy(
            benchmark, volume_path(dates), median_window=MEDIAN_WINDOW,
            min_train_bars=30, refit_every_bars=10, n_init=1, n_iter=50,
            top_n=2, risk_on_states=(0, 1, 2),
        )
        self.assertTrue((model.bars["log_return"] == 0.0).all())
        result = Backtester(prices, benchmark_frame, membership, model,
                            initial_cash=100_000, fee_bps=0).run(
                                dates[1], dates[-1])
        self.assertFalse(result.refits.empty)
        self.assertTrue(result.refits.status.eq("success").all())
        self.assertTrue(np.isfinite(result.history.nav).all())
        self.assertTrue((result.history.nav > 0).all())
        self.assertTrue(result.phases.phase.iloc[-1] == "complete")


# ------------------------------------------------------------- defects
class DefectsRound2(StrategyCase):
    """Regression evidence for defects found by this round (must fail)."""

    def test_fit_survives_one_large_most_recent_bar(self):
        """One crisis bar must not make every HMM initialisation fail.

        A 12% five-session move is a 2008/2020-style week: 14 sigma against a
        sample whose bar volatility is 0.9%.  When that bar is the most recent
        observation, one component ends up with no observed outgoing
        transition, ``model.fit`` leaves an all-zero ``transmat_`` row and
        ``model.score`` (``research/hmm.py:101``) raises ``ValueError:
        transmat_ rows must sum to 1``; ``research/hmm.py:102`` swallows it for
        every restart and ``research/hmm.py:109`` raises ``RuntimeError``.  The
        very same bar is fitted without trouble when it is not the last
        observation, so the failure is an artefact of the initialisation/EM
        path, not an impossible sample.
        """
        training = np.r_[SAMPLE[:29], [CRISIS]]
        self.assertEqual(training.size, 30)
        clean = fit_hmm(SAMPLE[:29], n_states=3, n_init=1, seed=7)
        self.assertTrue(np.isfinite(clean.log_likelihood))

        for n_states in (2, 3):
            with self.subTest(n_states=n_states):
                for n_init in (1, 3):
                    try:
                        fit = fit_hmm(training, n_states=n_states,
                                      n_init=n_init, seed=7)
                    except RuntimeError as error:
                        self.fail(
                            "a single 12% bar as the most recent training "
                            f"observation (n_states={n_states}, "
                            f"n_init={n_init}) made every HMM restart fail: "
                            f"{error} (research/hmm.py:100-109; the same bar "
                            "fits once it is not the last observation, and "
                            "strategies/hmm_volume.py:267 turns this into an "
                            "aborted walk-forward run)"
                        )
                    self.assertTrue(np.isfinite(fit.log_likelihood))
                    state, probabilities = infer_state(fit, training)
                    self.assertIn(state, range(n_states))
                    self.assertAlmostEqual(float(probabilities.sum()), 1.0,
                                           places=9)

    def test_a_boundary_basket_decision_terminates(self):
        """A per-signal decision must return within a bounded time.

        ``_scale_to_cash`` (``strategies/hmm_volume.py:82-84``) compensates for
        rounding by shrinking the largest budget one *ulp of that budget* at a
        time.  With 2000 names one element's ulp (1.1e-16) is a thousandth of
        the total's ulp (2.3e-13), so a single decrement does not change the
        recomputed total; worse, as the budgets become unequal the summation
        error *grows*: measured over 31,783 iterations the excess over
        ``usable`` rose from 2.3e-13 to 7.1e-11.  ``propose_buys`` therefore
        never returns for, e.g., 2000 eligible names and $2001.50 of cash,
        although 2000 equal $1.00075 budgets are a perfectly valid basket.
        """
        model, prices, signal = self._liquidity_model(top_n=None)
        symbols = tuple(f"T{i:04d}" for i in range(2000))
        request = context(model, signal, prices, symbols, nav=2001.5)
        if not hasattr(signal_module, "setitimer"):
            self.skipTest("SIGALRM is required to bound a non-terminating loop")

        def expire(signum, frame):
            raise TimeoutError("propose_buys did not return within 5 seconds")

        previous = signal_module.signal(signal_module.SIGALRM, expire)
        signal_module.setitimer(signal_module.ITIMER_REAL, 5.0)
        try:
            proposals = model.propose_buys(request)
        except TimeoutError:
            self.fail(
                "propose_buys did not return within 5 seconds for 2000 names "
                "and $2001.50 of cash: strategies/hmm_volume.py:82 "
                "('while sum(budgets.values()) > usable') is an unbounded "
                "compensation loop whose decrement is below the rounding "
                "granularity of the total"
            )
        finally:
            signal_module.setitimer(signal_module.ITIMER_REAL, 0.0)
            signal_module.signal(signal_module.SIGALRM, previous)
        self.assertEqual(len(proposals), 2000)
        self.assert_valid_budget(proposals, 2001.5, model.min_order_notional)

    def test_a_walk_forward_run_survives_a_crisis_bar(self):
        """A run must not abort because one refit lands on a crisis bar.

        ``Backtester._refit`` (``backtesting/engine.py:198-207``) stops the run
        when ``HMMVolumeStrategy.refit`` raises, and the refit at the bucket
        after a crisis bar trains on a window whose last observation is that
        bar.  The same fixture completes when ``refit_every_bars`` happens to
        skip that window, so a research run's viability depends on the refit
        cadence rather than on the data being usable.
        """
        returns = SAMPLE.copy()
        returns[30] = CRISIS
        prices, benchmark_frame, membership, benchmark, dates = market(returns)
        for refit_every in (1, 13):
            with self.subTest(refit_every_bars=refit_every):
                model = HMMVolumeStrategy(
                    benchmark, volume_path(dates),
                    median_window=MEDIAN_WINDOW, min_train_bars=30,
                    refit_every_bars=refit_every, n_init=3, n_iter=200,
                    top_n=2,
                )
                engine = Backtester(prices, benchmark_frame, membership, model,
                                    initial_cash=100_000, fee_bps=0)
                try:
                    result = engine.run(dates[1], dates[-1])
                except RuntimeError as error:
                    self.fail(
                        "the walk-forward run aborted at a refit whose "
                        f"training window ends on the crisis bar "
                        f"(refit_every_bars={refit_every}): {error} "
                        f"(engine phase {engine.phase.value}, audit "
                        f"{engine.refits[-1]['signal_date'] if engine.refits else None})"
                    )
                self.assertTrue(result.refits.status.eq("success").all())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()
