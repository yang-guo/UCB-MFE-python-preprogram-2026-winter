"""Round-3 adversarial numerical-robustness tests.

Scope: the numerical repairs and guards added after the first two review
rounds, attacked with hostile parameters instead of hostile APIs.

* ``research.hmm._repair_model``: degenerate transition rows / start
  probabilities / covariance matrices.  A repair must either yield a model
  that can score a finite likelihood or raise; it must not silently return an
  invalid model, and it must not perturb an already valid model beyond
  floating-point normalisation.
* ``strategies.hmm_volume._scale_to_cash``: equal and wildly unequal budgets,
  ``1e300``/``1e308``/``DBL_MAX`` values, overflowing totals, 2000 names,
  cash exactly at / just above / below ``n x min_notional``, and floors of
  ``0``/``1``/``1e6``.
* ``HMMVolumeStrategy.refit`` / ``propose_buys`` / ``needs_refit``:
  backward dates, ``NaT``, post-data-end dates, timezone-aware dates, and
  future rows in the context price/volume panels.  Every rejection must be
  atomic (no partial state) and the exact boundaries must still work.
* Context panels whose index is not a ``DatetimeIndex`` or contains ``NaT``.

Confirmed defects found by this file (the tests are kept failing on purpose):

D1. ``research/hmm.py:86-90`` -- a start probability with negative entries
    whose sum is positive (``[-0.5, 0.5, 1.0]``) passes the ``sum() <= 0``
    test, is divided by its own sum, and keeps its negative entries.
    ``model.score`` then returns ``NaN``; ``_repair_model`` returns instead of
    raising.
D2. ``research/hmm.py:86-90`` -- a start probability with finite but
    overflowing entries (``[1e308] * 3``) is divided by ``inf`` and becomes
    an all-zero vector.  ``model.score`` then raises
    ``ValueError: startprob_ must sum to 1``; ``_repair_model`` returns
    instead of raising.
D3. ``strategies/hmm_volume.py:322-330`` -- the future-panel guards only run
    for ``DatetimeIndex`` frames, so a non-datetime index (for example string
    dates) containing a row after the signal date bypasses them; a future
    volume row then changes the top-``n`` liquidity selection.
D4. ``strategies/hmm_volume.py:322-325`` -- ``DatetimeIndex.max()`` skips
    ``NaT``, so a trailing ``NaT`` row passes the guard while
    ``context.prices.iloc[-1]`` is that undated row; it is used for the
    eligibility filter and can change the basket.

``D3``/``D4`` are API-hardening defects: the engine always supplies a
``DatetimeIndex`` pivot without ``NaT``, so they cannot fire inside a
``Backtester`` run, but the guards exist precisely for hostile callers of the
public ``propose_buys`` contract.

Everything else checked here passes.
"""

import copy
import math
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

from backtesting import DecisionContext, Portfolio
from research.hmm import _repair_model, fit_hmm
from strategies import hmm_volume as hmm_volume_module
from strategies.hmm_volume import HMMVolumeStrategy, _scale_to_cash

# hmmlearn touches deprecated NumPy shape assignment on every score call.
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r"hmmlearn(\..*)?")


FAST = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=30,
    refit_every_bars=10,
    n_init=1,
    n_iter=50,
    top_n=2,
    risk_on_states=(0, 1, 2),
)

DBL_MAX = np.finfo(float).max


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def synthetic_market(periods=560, symbols=4, seed=21, start="2016-01-01"):
    """A calendar, a benchmark, a volume proxy, prices and membership."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    benchmark = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15.0, 0.3, periods), index=dates)
    rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        path = benchmark * np.exp(np.cumsum(rng.normal(0.0, 0.004, periods)))
        rows.append(
            pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": path})
        )
    prices = pd.concat(rows, ignore_index=True)
    membership = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": [f"S{i}" for i in range(symbols)],
                    "quarter_start": [quarter] * symbols,
                    "selection_date": [quarter - pd.Timedelta(days=1)] * symbols,
                }
            )
            for quarter in pd.date_range(dates[0], dates[-1], freq="QS")
        ],
        ignore_index=True,
    )
    return dict(
        dates=dates,
        benchmark=pd.Series(benchmark, index=dates),
        benchmark_frame=pd.DataFrame({"date": dates, "adj_close": benchmark}),
        volume=volume,
        prices=prices,
        membership=membership,
    )


def make_model(market, **overrides):
    options = dict(FAST)
    options.update(overrides)
    return HMMVolumeStrategy(market["benchmark"], market["volume"], **options)


def price_pivot(market, signal):
    """The engine's own point-in-time price view, cut at ``signal``."""
    frame = market["prices"]
    view = frame.loc[frame["date"] <= pd.Timestamp(signal)].pivot(
        index="date", columns="symbol", values="adj_close"
    )
    return view


def replace_signal(context, signal_date):
    return DecisionContext(
        signal_date,
        context.execution_date,
        context.prices,
        context.membership,
        context.fundamentals,
        context.portfolio,
        context.model_version,
        context.volume,
        context.financials,
    )


def decision_context(market, signal, *, cash=100_000.0, prices=None,
                     volume=None):
    signal = pd.Timestamp(signal)
    view = price_pivot(market, signal) if prices is None else prices
    return DecisionContext(
        signal,
        signal + pd.Timedelta(days=1),
        view,
        market["membership"].copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, cash, {}),
        0,
        pd.DataFrame() if volume is None else volume,
        pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# HMM parameter repair
# ---------------------------------------------------------------------------

def diagonal_of(model):
    covars = np.asarray(model.covars_, dtype=float)
    if covars.ndim == 3:
        return np.diagonal(covars, axis1=1, axis2=2)
    return covars


class RepairModelNumericAttack(unittest.TestCase):
    """``_repair_model`` must yield a scorable model or raise."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=400)
        cls.fit = fit_hmm(sample, n_states=3, n_init=1, seed=0)
        cls.z = ((sample - cls.fit.mean) / cls.fit.scale).reshape(-1, 1)
        cls.components = cls.fit.model.n_components

    def fresh(self):
        return copy.deepcopy(self.fit.model)

    # -- assertions -------------------------------------------------------

    def assert_valid_simplex(self, model, tag):
        trans = np.asarray(model.transmat_, dtype=float)
        self.assertEqual(trans.shape, (self.components, self.components))
        self.assertTrue(np.isfinite(trans).all(), msg=f"{tag}: transmat")
        self.assertTrue((trans >= 0).all(), msg=f"{tag}: negative transmat")
        np.testing.assert_allclose(
            trans.sum(axis=1), np.ones(self.components), rtol=0, atol=1e-15,
            err_msg=f"{tag}: transition rows must sum to one",
        )
        start = np.asarray(model.startprob_, dtype=float)
        self.assertEqual(start.shape, (self.components,))
        self.assertTrue(np.isfinite(start).all(), msg=f"{tag}: startprob")
        self.assertTrue((start >= 0).all(),
                        msg=f"{tag}: negative startprob entries")
        self.assertAlmostEqual(
            float(start.sum()), 1.0, places=15,
            msg=f"{tag}: startprob must sum to one, got {start!r}",
        )

    def assert_valid_covariances(self, model, tag):
        diagonal = diagonal_of(model)
        self.assertTrue(np.isfinite(diagonal).all(), msg=f"{tag}: covariance")
        self.assertTrue((diagonal > 0).all(),
                        msg=f"{tag}: nonpositive covariance {diagonal!r}")

    def assert_scores_finitely(self, model, tag):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            likelihood = model.score(self.z)
        self.assertTrue(np.isfinite(likelihood),
                        msg=f"{tag}: repaired model scored {likelihood!r}")

    def assert_valid_model(self, model, tag):
        self.assert_valid_simplex(model, tag)
        self.assert_valid_covariances(model, tag)
        self.assert_scores_finitely(model, tag)

    # -- valid models -----------------------------------------------------

    def test_valid_model_is_unchanged_and_repair_is_idempotent(self):
        """Repairing a fitted model must not move it beyond normalisation."""
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(100 + seed)
                sample = rng.normal(size=300)
                fit = fit_hmm(sample, n_states=3, n_init=1, seed=seed)
                z = ((sample - fit.mean) / fit.scale).reshape(-1, 1)
                model = fit.model
                before = (
                    np.asarray(model.transmat_, dtype=float).copy(),
                    np.asarray(model.startprob_, dtype=float).copy(),
                    np.asarray(model.means_, dtype=float).copy(),
                    np.asarray(model.covars_, dtype=float).copy(),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    before_ll = model.score(z)
                _repair_model(model)
                after = (
                    np.asarray(model.transmat_, dtype=float),
                    np.asarray(model.startprob_, dtype=float),
                    np.asarray(model.means_, dtype=float),
                    np.asarray(model.covars_, dtype=float),
                )
                # hmmlearn can leave a transition row summing to 1 - 1 ulp, so
                # a re-normalisation is expected to move entries by an ulp.
                for old, new, name in zip(
                    before, after, ("transmat", "startprob", "means", "covars")
                ):
                    np.testing.assert_allclose(
                        new, old, rtol=0, atol=1e-14,
                        err_msg=f"seed {seed}: repair moved {name}",
                    )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    after_ll = model.score(z)
                self.assertLessEqual(abs(after_ll - before_ll),
                                     1e-11 * max(1.0, abs(before_ll)),
                                     msg=f"seed {seed}: log-likelihood moved")
                # A second repair is an exact no-op.
                snapshot = (
                    np.asarray(model.transmat_).copy(),
                    np.asarray(model.startprob_).copy(),
                    np.asarray(model.covars_).copy(),
                )
                _repair_model(model)
                self.assertTrue(np.array_equal(
                    snapshot[0], np.asarray(model.transmat_)))
                self.assertTrue(np.array_equal(
                    snapshot[1], np.asarray(model.startprob_)))
                self.assertTrue(np.array_equal(
                    snapshot[2], np.asarray(model.covars_)))

    # -- transitions ------------------------------------------------------

    def test_all_zero_transition_matrix_becomes_uniform(self):
        model = self.fresh()
        model.transmat_ = np.zeros((self.components, self.components))
        _repair_model(model)
        self.assert_valid_model(model, "all-zero transitions")
        np.testing.assert_allclose(
            np.asarray(model.transmat_),
            np.full((self.components, self.components), 1.0 / self.components),
            rtol=0, atol=1e-15,
        )

    def test_hostile_transition_rows_are_repaired_in_place(self):
        base = np.asarray(self.fit.model.transmat_, dtype=float).copy()
        cases = {
            "all-nan-row": (1, np.full(self.components, np.nan)),
            "all-inf-row": (2, np.full(self.components, np.inf)),
            "negative-row": (0, np.full(self.components, -1.0)),
            "single-nan-cell": (1, "cell"),
            "overflowing-row": (2, 1e308),
        }
        for tag, (row, value) in cases.items():
            with self.subTest(case=tag):
                model = self.fresh()
                trans = base.copy()
                if isinstance(value, str):
                    trans[row, 0] = np.nan
                else:
                    trans[row] = value
                model.transmat_ = trans
                _repair_model(model)
                self.assert_valid_model(model, tag)
                repaired = np.asarray(model.transmat_, dtype=float)
                # Rows that were already a valid simplex keep their shape.
                for other in range(self.components):
                    if other == row:
                        continue
                    np.testing.assert_allclose(
                        repaired[other], base[other], rtol=0, atol=1e-15,
                        err_msg=f"{tag}: valid row {other} was modified",
                    )

    # -- start probabilities ---------------------------------------------

    def test_zero_nan_and_infinite_startprob_become_uniform(self):
        for tag, values in {
            "all-zero": np.zeros(self.components),
            "nan": np.array([np.nan, 0.5, 0.5]),
            "positive-inf": np.array([np.inf, 0.0, 1.0]),
            "all-negative": -np.ones(self.components),
        }.items():
            with self.subTest(case=tag):
                model = self.fresh()
                model.startprob_ = values
                _repair_model(model)
                self.assert_valid_model(model, tag)
                np.testing.assert_allclose(
                    np.asarray(model.startprob_),
                    np.full(self.components, 1.0 / self.components),
                    rtol=0, atol=1e-15,
                )

    def test_negative_startprob_with_positive_sum_is_repaired_or_rejected(self):
        """D1: ``[-0.5, 0.5, 1.0]`` sums to one but is not a distribution."""
        model = self.fresh()
        model.startprob_ = np.array([-0.5, 0.5, 1.0])
        try:
            _repair_model(model)
        except ValueError:
            return  # rejecting the model is acceptable
        start = np.asarray(model.startprob_, dtype=float)
        self.assertTrue(
            (start >= 0).all(),
            msg=(
                "research/hmm.py:86-90: _repair_model accepted a startprob "
                "with negative entries because only sum() <= 0 is checked; "
                f"after repair startprob_={start!r} and model.score is "
                "NaN. Reject negative entries or replace them with the "
                "uniform vector."
            ),
        )
        self.assertAlmostEqual(float(start.sum()), 1.0, places=15)
        self.assert_scores_finitely(model, "negative startprob")

    def test_overflowing_startprob_sum_is_repaired_or_rejected(self):
        """D2: finite entries whose sum overflows must not normalise to zero."""
        model = self.fresh()
        model.startprob_ = np.full(self.components, 1e308)
        try:
            _repair_model(model)
        except ValueError:
            return  # rejecting the model is acceptable
        start = np.asarray(model.startprob_, dtype=float)
        self.assertTrue(np.isfinite(start).all())
        self.assertAlmostEqual(
            float(start.sum()), 1.0, places=15,
            msg=(
                "research/hmm.py:86-90: _repair_model divides by an "
                "overflowing startprob sum, leaving startprob_="
                f"{start!r}; model.score then raises "
                "'startprob_ must sum to 1'. Check isfinite(start.sum())."
            ),
        )
        self.assert_scores_finitely(model, "overflowing startprob")

    # -- covariance -------------------------------------------------------

    def covars(self, values):
        """Write the internal diagonal covariance array of a diag model."""
        array = np.asarray(values, dtype=float).reshape(self.components, 1)
        return array

    def test_degenerate_covariance_is_floored(self):
        cases = {
            "zero": [0.0, 0.6, 0.7],
            "negative": [-1.0, 0.6, 0.7],
            "nan": [np.nan, 0.6, 0.7],
            "positive-inf": [np.inf, 0.6, 0.7],
            "all-bad": [0.0, -1.0, np.nan],
        }
        for tag, values in cases.items():
            with self.subTest(case=tag):
                model = self.fresh()
                model._covars_ = self.covars(values)
                _repair_model(model)
                self.assert_valid_model(model, tag)
                repaired = np.ravel(diagonal_of(model))
                expected = np.array(
                    [v if np.isfinite(v) and v > 0 else 1e-9 for v in values]
                )
                np.testing.assert_array_equal(repaired, expected)

    def test_impossible_covariance_shape_raises_value_error(self):
        for tag, array in {
            "too-few-components": np.ones((2, 1)),
            "too-many-features": np.ones((self.components, 2)),
        }.items():
            with self.subTest(case=tag):
                model = self.fresh()
                model._covars_ = array
                with self.assertRaises(ValueError):
                    _repair_model(model)

    def test_nonfinite_or_misshapen_means_raise_value_error(self):
        for tag, means in {
            "nan": np.array([[np.nan], [0.0], [1.0]]),
            "wrong-shape": np.zeros((2, 1)),
        }.items():
            with self.subTest(case=tag):
                model = self.fresh()
                model.means_ = means
                with self.assertRaises(ValueError):
                    _repair_model(model)


# ---------------------------------------------------------------------------
# cash scaling
# ---------------------------------------------------------------------------

class ScaleToCashNumericAttack(unittest.TestCase):
    """``_scale_to_cash`` invariants under hostile numbers."""

    def assert_contract(self, budgets, cash, floor):
        out = _scale_to_cash(dict(budgets), cash, min_notional=floor)
        self.assertIsInstance(out, dict)
        values = np.fromiter(out.values(), dtype=float) if out else np.array([])
        if values.size:
            self.assertTrue(np.isfinite(values).all(),
                            msg=f"nonfinite budget in {out!r}")
            self.assertTrue((values >= floor).all(),
                            msg=f"budget below floor {floor}: {out!r}")
            naive = float(sum(out.values()))
            exact = math.fsum(out.values())
            self.assertLessEqual(
                naive, cash,
                msg=f"overspend: sum={naive!r} > cash={cash!r}",
            )
            self.assertLessEqual(
                exact, cash,
                msg=f"overspend (fsum): sum={exact!r} > cash={cash!r}",
            )
        return out

    def test_equal_budgets_stay_bit_identical(self):
        for cash in (100.0, 5.5, 1e6, 1e300, 1e308):
            with self.subTest(cash=cash):
                budgets = {"A": cash / 3.0, "B": cash / 3.0, "C": cash / 3.0}
                out = self.assert_contract(budgets, cash, 1e-12)
                self.assertEqual(len(out), 3)
                self.assertEqual(len(set(out.values())), 1,
                                 msg=f"equal budgets diverged: {out!r}")

    def test_uniform_factor_preserves_ratios(self):
        budgets = {"A": 1.0, "B": 2.0, "C": 7.0}
        out = self.assert_contract(budgets, 5.0, 0.0)
        self.assertEqual(len(out), 3)
        for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
            self.assertAlmostEqual(
                out[left] / out[right],
                budgets[left] / budgets[right],
                places=12,
                msg="scaling must use one uniform factor",
            )

    def test_two_thousand_names_are_exact_and_bounded(self):
        try:
            import signal as signal_module
        except ImportError:  # pragma: no cover - non-POSIX
            signal_module = None
        cash = 2001.5
        budgets = {f"T{i:04d}": cash / 2000 for i in range(2000)}
        if signal_module is not None and hasattr(signal_module, "setitimer"):
            def expire(signum, frame):
                raise TimeoutError("_scale_to_cash did not return")

            previous = signal_module.signal(signal_module.SIGALRM, expire)
            signal_module.setitimer(signal_module.ITIMER_REAL, 5.0)
            try:
                out = self.assert_contract(budgets, cash, 1.0)
            finally:
                signal_module.setitimer(signal_module.ITIMER_REAL, 0.0)
                signal_module.signal(signal_module.SIGALRM, previous)
        else:  # pragma: no cover - non-POSIX
            out = self.assert_contract(budgets, cash, 1.0)
        self.assertEqual(len(out), 2000)
        values = np.fromiter(out.values(), dtype=float)
        self.assertEqual(values.max() - values.min(), 0.0)
        self.assertGreaterEqual(values.min(), 1.0)

    def test_floor_boundaries(self):
        floor = 10.0
        names = {f"S{i}": floor for i in range(10)}
        # Exactly n x floor: the documented buffer keeps the basket below the
        # floor, so the conservative answer is no order at all.
        exact = self.assert_contract(names, 10 * floor, floor)
        self.assertEqual(exact, {})
        # Just above the boundary every name is bought at or above the floor.
        above = self.assert_contract(names, 10 * floor + 0.01, floor)
        self.assertEqual(len(above), 10)
        self.assertGreaterEqual(min(above.values()), floor)
        # Below the boundary nothing can respect the floor.
        below = self.assert_contract(names, 10 * floor - 1.0, floor)
        self.assertEqual(below, {})

    def test_floor_of_zero_one_and_one_million(self):
        budgets = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        self.assertEqual(len(self.assert_contract(budgets, 1.0, 0.0)), 4)
        self.assertEqual(len(self.assert_contract(budgets, 1.0, 1.0)), 0)
        self.assertEqual(len(self.assert_contract(budgets, 1.0, 1e6)), 0)
        big = {"A": 1e6, "B": 1e6}
        self.assertEqual(len(self.assert_contract(big, 1e6, 1e6)), 0)
        # Cash exactly at 2 x floor falls just below the floor after the
        # documented buffer, so one extra unit of cash is needed.
        self.assertEqual(len(self.assert_contract(big, 2e6 + 1.0, 1e6)), 2)
        self.assertEqual(self.assert_contract(big, 1e5, 1e6), {})

    def test_extreme_magnitudes_never_overspend(self):
        # Power-of-two counts keep cash / n exact so the total cannot overflow.
        for cash, count in ((1e300, 8), (1e308, 4), (DBL_MAX, 2048)):
            with self.subTest(cash=cash):
                budgets = {f"X{i}": cash / count for i in range(count)}
                out = self.assert_contract(budgets, cash, 0.0)
                self.assertEqual(len(out), count)
                values = np.fromiter(out.values(), dtype=float)
                self.assertTrue(np.isfinite(values).all())
                self.assertEqual(values.max() - values.min(), 0.0)

    def test_overflowing_total_fails_closed(self):
        budgets = {"A": 1e308, "B": 1e308}
        self.assertEqual(_scale_to_cash(dict(budgets), 1e308, min_notional=0.0), {})
        self.assertEqual(
            _scale_to_cash({"A": DBL_MAX, "B": DBL_MAX}, DBL_MAX, 1.0), {}
        )

    def test_single_name_at_double_maximum(self):
        out = self.assert_contract({"A": DBL_MAX}, DBL_MAX, 0.0)
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isfinite(out["A"]))

    def test_invalid_cash_returns_empty(self):
        for cash in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(cash=cash):
                self.assertEqual(_scale_to_cash({"A": 1.0}, cash, 0.0), {})
        self.assertEqual(_scale_to_cash({}, 100.0, 0.0), {})
        self.assertEqual(
            _scale_to_cash({"A": np.nan, "B": -1.0, "C": 0.0}, 10.0, 0.0), {}
        )

    def test_random_sweep_keeps_the_contract(self):
        rng = np.random.default_rng(1017)
        floors = [0.0, 1e-9, 1.0, 1e6]
        for _ in range(300):
            n = int(rng.integers(1, 12))
            scale = 10.0 ** float(rng.uniform(-250, 250))
            budgets = {
                f"R{i}": float(np.exp(rng.normal(0.0, 2.0)) * scale)
                for i in range(n)
            }
            cash = float(np.exp(rng.normal(0.0, 2.0)) * scale)
            floor = float(floors[int(rng.integers(len(floors)))])
            self.assert_contract(budgets, cash, floor)


# ---------------------------------------------------------------------------
# guard atomicity and boundaries
# ---------------------------------------------------------------------------

def model_snapshot(model):
    probabilities = (
        None if model._probabilities is None
        else np.array(model._probabilities, dtype=float, copy=True)
    )
    returns = (
        None if model._bar_returns is None
        else np.array(model._bar_returns, dtype=float, copy=True)
    )
    return dict(
        fit_id=id(model._fit),
        likelihood=(
            None if model._fit is None else model._fit.log_likelihood
        ),
        state=model._state,
        probabilities=probabilities,
        fit_bar_count=model._fit_bar_count,
        fit_bar_end=model._fit_bar_end,
        state_bar_count=model._state_bar_count,
        state_bar_end=model._state_bar_end,
        fit_date=model._fit_date,
        bar_returns=returns,
        parameters=fitted_parameters(model),
    )


def fitted_parameters(model):
    fit = model._fit
    if fit is None:
        return None
    return (
        np.array(fit.model.transmat_, dtype=float, copy=True),
        np.array(fit.model.startprob_, dtype=float, copy=True),
        np.array(fit.model.means_, dtype=float, copy=True),
        np.array(fit.model.covars_, dtype=float, copy=True),
        fit.mean,
        fit.scale,
    )


class GuardAtomicity(unittest.TestCase):
    """Rejected requests must leave the fitted state exactly untouched."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market()
        cls.model = make_model(cls.market)
        cls.bars = cls.model.bars
        cls.signal = cls.bars["end"].iloc[40]
        cls.model.refit(None, decision_context(cls.market, cls.signal))

    def setUp(self):
        self.model = copy.deepcopy(self.__class__.model)
        self.bars = self.model.bars

    def assert_state_unchanged(self, before, tag):
        after = model_snapshot(self.model)
        for key in before:
            left, right = before[key], after[key]
            if key == "parameters":
                self.assertEqual(
                    left is None, right is None,
                    msg=f"{tag}: fitted model appeared or disappeared",
                )
                if left is None:
                    continue
                for old, new, name in zip(
                    left, right,
                    ("transmat", "startprob", "means", "covars", "mean",
                     "scale"),
                ):
                    if name in ("mean", "scale"):
                        self.assertEqual(
                            old, new,
                            msg=f"{tag}: fit {name} changed",
                        )
                    else:
                        self.assertTrue(
                            np.array_equal(old, new),
                            msg=f"{tag}: fitted {name} was mutated",
                        )
            elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                self.assertTrue(
                    np.array_equal(left, right),
                    msg=f"{tag}: {key} changed by a rejected call",
                )
            else:
                self.assertEqual(
                    left, right, msg=f"{tag}: {key} changed by a rejected call"
                )

    def test_backward_refit_is_rejected_atomically(self):
        before = model_snapshot(self.model)
        earlier = self.bars["end"].iloc[30]
        with self.assertRaises(ValueError):
            self.model.refit(None, decision_context(self.market, earlier))
        self.assert_state_unchanged(before, "backward refit")
        # The original decision still works afterwards.
        self.model.refit(None, decision_context(self.market, self.signal))
        self.assertEqual(self.model._fit_date, self.signal)

    def test_nat_signal_is_rejected_atomically(self):
        before = model_snapshot(self.model)
        context = replace_signal(
            decision_context(self.market, self.signal), pd.NaT
        )
        with self.assertRaises(ValueError):
            self.model.refit(None, context)
        with self.assertRaises(ValueError):
            self.model.propose_buys(context)
        with self.assertRaises(ValueError):
            self.model.needs_refit(context)
        self.assert_state_unchanged(before, "NaT signal")

    def test_timezone_aware_signal_is_rejected_atomically(self):
        before = model_snapshot(self.model)
        aware = pd.Timestamp(self.signal, tz="UTC")
        context = replace_signal(
            decision_context(self.market, self.signal), aware
        )
        with self.assertRaises(ValueError):
            self.model.refit(None, context)
        with self.assertRaises(ValueError):
            self.model.propose_buys(context)
        with self.assertRaises(ValueError):
            self.model.needs_refit(context)
        self.assert_state_unchanged(before, "timezone-aware signal")

    def test_post_data_end_signal_is_rejected_and_the_exact_end_is_allowed(self):
        end = self.market["benchmark"].index[-1]
        before = model_snapshot(self.model)
        for offset in (pd.Timedelta(days=1), pd.Timedelta(1, "ns")):
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    self.model.refit(
                        None, decision_context(self.market, end + offset)
                    )
                self.assert_state_unchanged(before, "post-end refit")
                with self.assertRaises(ValueError):
                    self.model.propose_buys(
                        decision_context(self.market, end + offset)
                    )
                self.assert_state_unchanged(before, "post-end proposal")
        # Exactly at the last supplied date the call is valid.
        self.model.refit(None, decision_context(self.market, end))
        self.assertEqual(self.model._fit_date, end)
        self.assertEqual(self.model.state_date, self.bars["end"].iloc[-1])
        probabilities = self.model.probabilities
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_refit_cannot_rewind_below_the_last_scored_bar(self):
        later = self.bars["end"].iloc[45]
        self.model.propose_buys(decision_context(self.market, later))
        self.assertEqual(self.model.state_date, later)
        before = model_snapshot(self.model)
        with self.assertRaises(ValueError):
            self.model.refit(None, decision_context(self.market, self.signal))
        self.assert_state_unchanged(before, "rewind below scored bar")
        # Exactly at the last scored bar is a valid boundary.
        self.model.refit(None, decision_context(self.market, later))
        self.assertEqual(self.model._fit_date, later)
        self.assertEqual(self.model._state_bar_count, 46)

    def test_future_price_panel_is_rejected_before_scoring(self):
        later = self.bars["end"].iloc[45]
        self.model.propose_buys(decision_context(self.market, later))
        scored = self.model._state_bar_count
        before = model_snapshot(self.model)
        view = price_pivot(self.market, later)
        future = pd.DataFrame(
            {symbol: [1.0] for symbol in view.columns},
            index=[later + pd.Timedelta(days=3)],
        )
        hostile = decision_context(
            self.market, later, prices=pd.concat([view, future])
        )
        with self.assertRaises(ValueError):
            self.model.propose_buys(hostile)
        self.assert_state_unchanged(before, "future price panel")
        self.assertEqual(self.model._state_bar_count, scored)
        # A valid call at the same date still advances the scored bar.
        self.model.propose_buys(decision_context(self.market, later))
        self.assertEqual(self.model._state_bar_count, scored)

    def test_future_volume_panel_is_rejected_before_scoring(self):
        later = self.bars["end"].iloc[45]
        before = model_snapshot(self.model)
        volume = pd.DataFrame(
            {"S0": [1.0]}, index=[later + pd.Timedelta(days=2)]
        )
        hostile = decision_context(self.market, later, volume=volume)
        with self.assertRaises(ValueError):
            self.model.propose_buys(hostile)
        self.assert_state_unchanged(before, "future volume panel")

    def test_failed_fit_does_not_commit_partial_refit_state(self):
        before = model_snapshot(self.model)
        with mock.patch.object(
            hmm_volume_module, "fit_hmm", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                self.model.refit(
                    None, decision_context(self.market, self.bars["end"].iloc[45])
                )
        self.assert_state_unchanged(before, "failed fit")

    def test_failed_inference_does_not_commit_partial_state(self):
        before = model_snapshot(self.model)
        with mock.patch.object(
            hmm_volume_module,
            "infer_state",
            side_effect=ValueError("bad history"),
        ):
            with self.assertRaises(ValueError):
                self.model.refit(
                    None, decision_context(self.market, self.bars["end"].iloc[45])
                )
            self.assert_state_unchanged(before, "failed refit inference")
            with self.assertRaises(ValueError):
                self.model.propose_buys(
                    decision_context(self.market, self.bars["end"].iloc[45])
                )
        self.assert_state_unchanged(before, "failed proposal inference")

    def test_repeated_refit_at_the_same_signal_is_allowed(self):
        first = model_snapshot(self.model)
        self.model.refit(None, decision_context(self.market, self.signal))
        self.assertEqual(self.model._fit_date, self.signal)
        self.assertEqual(self.model._state_bar_count, first["state_bar_count"])
        self.assertEqual(self.model.state_date, first["state_bar_end"])
        self.assertIn(self.model.state, range(self.model.n_states))


# ---------------------------------------------------------------------------
# context panel causality
# ---------------------------------------------------------------------------

class ContextPanelCausality(unittest.TestCase):
    """A row the strategy cannot date must never influence the basket."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market()
        cls.model = make_model(cls.market)
        cls.bars = cls.model.bars
        cls.signal = cls.bars["end"].iloc[40]
        cls.symbols = ["S0", "S1", "S2", "S3"]
        cls.model.refit(None, decision_context(cls.market, cls.signal))

    def setUp(self):
        self.model = copy.deepcopy(self.__class__.model)

    def one_row(self, volumes):
        index = pd.DatetimeIndex([self.signal])
        prices = pd.DataFrame({s: [10.0] for s in self.symbols}, index=index)
        volume = pd.DataFrame(
            {s: [volumes[s]] for s in self.symbols}, index=index
        )
        return prices, volume

    def clean_basket(self, volumes=None):
        volumes = volumes or {
            "S0": 100.0, "S1": 200.0, "S2": 300.0, "S3": 400.0
        }
        prices, volume = self.one_row(volumes)
        return self.model.propose_buys(
            decision_context(self.market, self.signal,
                             prices=prices, volume=volume)
        )

    def test_valid_datetime_panels_select_the_most_liquid_names(self):
        basket = self.clean_basket()
        self.assertEqual(sorted(basket), ["S2", "S3"])
        self.assertLessEqual(sum(basket.values()), 100_000.0)

    def test_empty_volume_panel_is_accepted(self):
        prices, _ = self.one_row({s: 1.0 for s in self.symbols})
        basket = self.model.propose_buys(
            decision_context(self.market, self.signal, prices=prices,
                             volume=pd.DataFrame())
        )
        self.assertEqual(len(basket), self.model.top_n)
        self.assertLessEqual(sum(basket.values()), 100_000.0)

    def test_future_row_without_datetime_index_cannot_change_the_basket(self):
        """D3: string dates bypass the guard at strategies/hmm_volume.py:322."""
        clean = self.clean_basket()
        signal_text = str(pd.Timestamp(self.signal).date())
        index = [signal_text, "2030-01-01"]
        prices = pd.DataFrame(
            {s: [10.0, 10.0] for s in self.symbols}, index=index
        )
        volume = pd.DataFrame(
            {
                "S0": [100.0, 1e12],
                "S1": [200.0, 1e12],
                "S2": [300.0, 300.0],
                "S3": [400.0, 400.0],
            },
            index=index,
        )
        context = decision_context(
            self.market, self.signal, prices=prices, volume=volume
        )
        try:
            basket = self.model.propose_buys(context)
        except ValueError:
            return  # failing closed is the other acceptable behaviour
        self.assertEqual(
            basket, clean,
            msg=(
                "strategies/hmm_volume.py:322-330: a non-DatetimeIndex panel "
                "skips the future-data guards, so the row dated 2030-01-01 "
                "re-ranks liquidity and buys "
                f"{sorted(basket)} instead of {sorted(clean)}"
            ),
        )

    def test_undated_nat_row_cannot_change_the_basket(self):
        """D4: DatetimeIndex.max() skips NaT, so a NaT row slips past the guard."""
        symbols = ["S0", "S1"]
        index = pd.DatetimeIndex([self.signal])
        clean_prices = pd.DataFrame({s: [10.0] for s in symbols}, index=index)
        clean_volume = pd.DataFrame({s: [1.0] for s in symbols}, index=index)
        clean = self.model.propose_buys(
            decision_context(self.market, self.signal,
                             prices=clean_prices, volume=clean_volume)
        )
        hostile_index = pd.DatetimeIndex([self.signal, pd.NaT])
        prices = pd.DataFrame(
            {"S0": [10.0, 0.0], "S1": [10.0, 10.0]}, index=hostile_index
        )
        volume = pd.DataFrame(
            {"S0": [1.0, 1.0], "S1": [1.0, 1.0]}, index=hostile_index
        )
        context = decision_context(
            self.market, self.signal, prices=prices, volume=volume
        )
        try:
            basket = self.model.propose_buys(context)
        except ValueError:
            return  # failing closed is the other acceptable behaviour
        self.assertEqual(
            basket, clean,
            msg=(
                "strategies/hmm_volume.py:322-325: DatetimeIndex.max() skips "
                "NaT, so a trailing NaT row passes the 'no data after the "
                "signal' guard and context.prices.iloc[-1] (an undated row) "
                f"is used for eligibility, producing {basket!r} instead of "
                f"{clean!r}"
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
