"""Round-3 adversarial *mutation* audit of the round-2 fixes (agent 9).

This module attacks the test suite, not the strategy: it injects deliberate
bugs into temporary copies of the production source and asks whether the
existing ``tests/test_hmm_volume*.py`` modules go red.  No production file and
no existing test is modified.

Fixes under audit
-----------------
* ``research/hmm.py::_repair_model`` -- repairs degenerate HMM transition,
  start and covariance parameters and is called after ``model.fit``.
* ``strategies/hmm_volume.py::_scale_to_cash`` -- a single uniform scaling
  factor instead of a per-ulp compensation loop.
* ``HMMVolumeStrategy.refit`` -- rejects a signal earlier than the last fitted
  *or* last scored bar.
* ``HMMVolumeStrategy._check_signal`` -- rejects ``NaT``/NaN, timezone-aware
  and post-data-end signal dates.
* ``HMMVolumeStrategy.propose_buys`` -- rejects context price/volume panels
  with rows after the signal date.

Method
------
21 mutants were injected one at a time into a temporary copy of the
submission (``research/hmm.py`` and ``strategies/hmm_volume.py``), and the
whole pre-round-3 suite -- ``test_hmm_volume``, ``agent1..4``,
``round2_agent5..8`` (9 modules) -- plus the concurrently written round-3 peer
``round3_agent10`` was run against each copy.  The unmutated control copy is
green on the 9 pre-round-3 modules, so any red module there is caused by the
mutation.  ``round3_agent10`` has four pre-existing failures (its documented
defects D1-D4), which are subtracted before a mutant is credited to it.

Results
-------
killed by the pre-round-3 suite (14)
    R1  ``_repair_model`` disabled ................ round2_agent6 crisis tests
    R2  a valid model re-repaired (uniform transmat) ...... round2_agent7
    R4  transitions left unnormalised ............. round2_agent6 crisis tests
    R5  raise instead of repairing ................ round2_agent6 crisis tests
    S1b per-ulp loop (builtin ``sum``) ............ round2_agent6 termination
    S2  spend 1.01 x cash ............................ 9 modules
    S3  ignore ``min_notional`` ...................... 5 modules
    S4  drop the cash buffer ......................... 3 modules
    C2  remove the NaT check ......................... round2_agent8
    C3  remove the data-end check ................ agent4 + round2_agent8
    C5  remove ``propose_buys`` backward guards ... agent5 + round2_agent8
    C6  remove the future price guard ............. agent5 + round2_agent8
    C7  remove the future volume guard ............ agent5 + round2_agent8
    C8  make the future guards over-strict (``>=``) ... every module

survived the pre-round-3 suite, killed only by the round-3 peer (5)
    R3  repair transitions but not covariances .... agent10 covariance test
    R6  repair transitions/covariances not start .. agent10 startprob test
    R7  covariance floor ``1e-9`` -> ``0`` ........ agent10 covariance test
    S1  per-ulp loop with ``fsum`` ............... agent10 2000-name equality
    C4  remove the timezone guard ................. agent10 tz test

survived every existing module (2)
    C1  remove ``refit``'s ``_state_bar_end`` guard.  The bar-count backstop
        (``len(bars) < _state_bar_count``) rejects every reachable backward
        signal, so only the reported *reason* changes.  Killed below at that
        diagnostic level; ``LifecycleGuardChecks`` first proves the backstop is
        a different error, so the assertion measures the date guard.
    C9  remove ``refit``'s bar-count guard.  Equivalent mutant: for a monotone
        bar series every signal that satisfies the ``_state_bar_end`` guard has
        at least ``_state_bar_count`` bars, so the guard is unreachable.  No
        behavioural test can kill it; it stays as defence in depth.

The surviving gaps are closed by ``RepairModelInvariantChecks`` (R3/R6/R7),
``ScaleToCashChecks`` (S1/S1b), ``SignalContractChecks`` (C4) and
``LifecycleGuardChecks`` (C1).  ``MutationHarnessChecks`` proves each new test
is not vacuous by replaying its scenario against a deliberately broken
implementation, and that the broken implementation escapes.
"""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

import research.hmm as hmm_module
from backtesting import DecisionContext, Portfolio
from strategies.hmm_volume import HMMVolumeStrategy, _scale_to_cash

# hmmlearn emits a numpy deprecation on every fit; unrelated to the audit.
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")

ORIGINAL_REPAIR = hmm_module._repair_model


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def synthetic_market(periods=600, symbols=3, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15, 0.3, periods), index=dates)
    rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        path = benchmark * np.exp(np.cumsum(rng.normal(0, 0.004, periods)))
        rows.append(pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": path}))
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
        volume=volume,
        prices=prices,
        membership=membership,
    )


MARKET = synthetic_market()
OPTIONS = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=30,
    refit_every_bars=10,
    n_init=1,
    n_iter=20,
    seed=3,
    top_n=2,
    risk_on_states=(0, 1, 2),
)


def build(cls=HMMVolumeStrategy, **overrides):
    options = dict(OPTIONS)
    options.update(overrides)
    return cls(MARKET["benchmark"], MARKET["volume"], **options)


def price_panel(signal):
    signal = pd.Timestamp(signal)
    frame = MARKET["prices"]
    return frame.loc[frame["date"] <= signal].pivot(
        index="date", columns="symbol", values="adj_close"
    )


def context(signal, *, prices=None, cash=100_000.0):
    signal = pd.Timestamp(signal)
    return DecisionContext(
        signal,
        signal + pd.Timedelta(days=1),
        price_panel(signal) if prices is None else prices,
        MARKET["membership"].copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, cash, {}),
        0,
        pd.DataFrame(),
        pd.DataFrame(),
    )


def healthy_model(seed=1, n=400):
    """A GaussianHMM fit on a well-separated two-regime sample."""
    rng = np.random.default_rng(seed)
    sample = np.r_[rng.normal(-1.5, 0.5, n // 2), rng.normal(1.5, 0.5, n // 2)]
    X = sample.reshape(-1, 1)
    model = GaussianHMM(
        n_components=2, covariance_type="diag", n_iter=100, random_state=seed
    )
    model.fit(X)
    return model, X


# --------------------------------------------------------------------------- #
# scenarios shared by the regression tests and the mutation harness
# --------------------------------------------------------------------------- #
def repair_identity_outcome(module):
    """Apply the repair to an already valid fit; nothing may change."""
    model, X = healthy_model()
    before = dict(
        transmat=np.array(model.transmat_, dtype=float, copy=True),
        startprob=np.array(model.startprob_, dtype=float, copy=True),
        means=np.array(model.means_, dtype=float, copy=True),
        covars=np.array(model.covars_, dtype=float, copy=True),
        score=float(model.score(X)),
    )
    module._repair_model(model)
    after = dict(
        transmat=np.array(model.transmat_, dtype=float, copy=True),
        startprob=np.array(model.startprob_, dtype=float, copy=True),
        means=np.array(model.means_, dtype=float, copy=True),
        covars=np.array(model.covars_, dtype=float, copy=True),
        score=float(model.score(X)),
    )
    unchanged = all(
        np.allclose(before[key], after[key], rtol=1e-12, atol=1e-15)
        for key in ("transmat", "startprob", "means", "covars")
    ) and abs(before["score"] - after["score"]) <= 1e-9 * max(1.0, abs(before["score"]))
    return dict(before=before, after=after, unchanged=unchanged)


def degenerate_covar_model():
    """An unfitted model whose variances are zero/NaN (property bypassed)."""
    model = GaussianHMM(n_components=3, covariance_type="diag")
    model.n_features = 1          # normally set by fit
    model.means_ = np.array([[0.0], [1.0], [2.0]])
    model._covars_ = np.array([[0.5], [0.0], [np.nan]])  # setter validates
    model.transmat_ = np.full((3, 3), 1.0 / 3.0)
    model.startprob_ = np.full(3, 1.0 / 3.0)
    return model


def repair_covar_outcome(module):
    model = degenerate_covar_model()
    module._repair_model(model)
    covars = np.asarray(model.covars_, dtype=float)
    diagonal = np.diagonal(covars, axis1=1, axis2=2) if covars.ndim == 3 else covars
    return dict(diagonal=diagonal.ravel(), covars=covars)


def degenerate_start_model():
    model = GaussianHMM(n_components=3, covariance_type="diag")
    model.n_features = 1
    model.means_ = np.array([[0.0], [1.0], [2.0]])
    model._covars_ = np.array([[0.5], [0.6], [0.7]])
    model.transmat_ = np.full((3, 3), 1.0 / 3.0)
    model.startprob_ = np.zeros(3)
    return model


def repair_start_outcome(module):
    model = degenerate_start_model()
    module._repair_model(model)
    return dict(startprob=np.asarray(model.startprob_, dtype=float).copy())


def scale_equal_outcome(scale):
    """Equal budgets that need scaling must stay bitwise equal.

    Three equal $10 budgets sum to $30, which is a rounding step above the
    ``$30 - buffer`` cap, so the scaling branch runs.  Every output must then be
    the *same* float: one uniform factor preserves equality.  The pre-fix code
    reproduced the cap by nudging one budget down an ulp at a time, which equal
    inputs do not survive (measured: two iterations, two distinct values).
    """
    n = 3
    cash = 30.0
    budgets = {f"S{i}": 10.0 for i in range(n)}
    out = scale(dict(budgets), cash, min_notional=1.0)
    values = list(out.values())
    return dict(
        budgets=budgets,
        out=out,
        count=len(out),
        distinct=len(set(values)),
        total=float(sum(values)),
    )


def tz_signal_outcome(cls):
    """Every public entry point must reject a timezone-aware signal date."""
    model = build(cls)
    signal = model.bars["end"].iloc[35]
    aware = pd.Timestamp(signal).tz_localize("UTC")
    view = context(signal)
    request = DecisionContext(
        aware,
        view.execution_date,
        view.prices,
        view.membership,
        view.fundamentals,
        view.portfolio,
        view.model_version,
        view.volume,
        view.financials,
    )
    calls = {
        "needs_refit": lambda: model.needs_refit(request),
        "refit": lambda: model.refit(None, request),
        "propose_buys": lambda: model.propose_buys(request),
    }
    results = {}
    for name, call in calls.items():
        try:
            call()
        except Exception as error:  # noqa: BLE001 - the type is the assertion
            results[name] = type(error).__name__
        else:
            results[name] = None
    return dict(aware=aware, results=results)


# --------------------------------------------------------------------------- #
# regression tests: kill the surviving mutants
# --------------------------------------------------------------------------- #
class RepairModelInvariantChecks(unittest.TestCase):
    """A repair may fix a broken model but must not rewrite a healthy one."""

    def test_repair_leaves_a_valid_fitted_model_untouched(self):
        outcome = repair_identity_outcome(hmm_module)
        before, after = outcome["before"], outcome["after"]
        self.assertTrue(
            outcome["unchanged"],
            msg=(
                "research/hmm.py::_repair_model changed a model whose "
                "parameters were already valid: transmat "
                f"{before['transmat'].round(4).tolist()} -> "
                f"{after['transmat'].round(4).tolist()}, log-likelihood "
                f"{before['score']:.6f} -> {after['score']:.6f}. A repair must "
                "be the identity on a healthy fit, otherwise it silently "
                "discards the fitted state persistence."
            ),
        )
        np.testing.assert_allclose(after["means"], before["means"], rtol=0, atol=0)
        np.testing.assert_allclose(
            after["transmat"], before["transmat"], rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(
            after["covars"], before["covars"], rtol=1e-12, atol=1e-15
        )

    def test_repair_returns_a_positive_finite_covariance_diagonal(self):
        outcome = repair_covar_outcome(hmm_module)
        diagonal = outcome["diagonal"]
        self.assertTrue(
            np.isfinite(diagonal).all() and (diagonal > 0).all(),
            msg=(
                "research/hmm.py::_repair_model left a non-positive or "
                f"non-finite variance after repair: {diagonal.tolist()}. "
                "Transitions alone are not enough: hmmlearn also rejects a "
                "zero or NaN covariance."
            ),
        )

    def test_repair_replaces_an_all_zero_start_probability(self):
        outcome = repair_start_outcome(hmm_module)
        start = outcome["startprob"]
        self.assertAlmostEqual(float(start.sum()), 1.0, places=12)
        self.assertTrue((start >= 0).all(), msg=f"startprob={start.tolist()}")
        self.assertTrue(
            np.allclose(start, 1.0 / 3.0),
            msg=(
                "research/hmm.py::_repair_model accepted an all-zero start "
                f"probability: startprob_={start.tolist()}; model.score then "
                "rejects the model."
            ),
        )


class ScaleToCashChecks(unittest.TestCase):
    def test_equal_budgets_stay_bitwise_equal_after_scaling(self):
        outcome = scale_equal_outcome(_scale_to_cash)
        self.assertEqual(outcome["count"], 3)
        self.assertEqual(
            outcome["distinct"],
            1,
            msg=(
                "equal budgets scaled to "
                f"{sorted(outcome['out'])} ({outcome['distinct']} distinct "
                "floats): strategies/hmm_volume.py::_scale_to_cash must apply "
                "one uniform factor, not nudge individual budgets until the "
                "rounding cap is met"
            ),
        )
        self.assertLessEqual(outcome["total"], 30.0)
        self.assertTrue(all(v >= 1.0 for v in outcome["out"].values()))


class LifecycleGuardChecks(unittest.TestCase):
    def test_refit_rejects_a_previously_scored_signal_as_a_backward_date(self):
        """The refusal must be the date guard, not the bar-count backstop.

        ``refit`` has two backstops for a rewind: the ``_state_bar_end``
        comparison and ``len(bars) < self._state_bar_count``.  The second fires
        for every reachable backward signal, so a test that only demands
        ``ValueError`` cannot see whether the date guard exists.  The two guards
        say different things, and a caller must be told that the *date* moved
        backwards.
        """
        model = build()
        bars = model.bars
        model.refit(None, context(bars["end"].iloc[35]))
        model.propose_buys(context(bars["end"].iloc[45]))
        before = (model._fit_date, model._state_bar_end, model._state_bar_count,
                  model.state, model._fit)
        with self.assertRaisesRegex(
            ValueError, "Refit signal date cannot move backwards"
        ):
            model.refit(None, context(bars["end"].iloc[40]))
        self.assertEqual(
            (model._fit_date, model._state_bar_end, model._state_bar_count,
             model.state),
            (before[0], before[1], before[2], before[3]),
        )
        self.assertIs(model._fit, before[4])

        # Non-vacuity: with only the date guard neutralised, the same scenario
        # is refused by the bar-count backstop with a different message.
        backstop = build()
        backstop.refit(None, context(bars["end"].iloc[35]))
        backstop.propose_buys(context(bars["end"].iloc[45]))
        backstop._state_bar_end = None
        with self.assertRaisesRegex(
            ValueError, "Refit cannot discard previously scored bars"
        ):
            backstop.refit(None, context(bars["end"].iloc[40]))


class SignalContractChecks(unittest.TestCase):
    def test_timezone_aware_signal_is_rejected_with_value_error(self):
        outcome = tz_signal_outcome(HMMVolumeStrategy)
        for name, error in outcome["results"].items():
            with self.subTest(entry_point=name):
                self.assertEqual(
                    error,
                    "ValueError",
                    msg=(
                        f"{name} answered signal_date={outcome['aware']!r}: "
                        f"raised {error!r}. A timezone-aware signal must be "
                        "rejected with the documented ValueError before any "
                        "date is compared with the naive history index."
                    ),
                )


# --------------------------------------------------------------------------- #
# mutation harness: prove the tests above are not vacuous
# --------------------------------------------------------------------------- #
def uniform_transmat_repair(model):
    """Emulates the 'repair valid models too' mutant."""
    ORIGINAL_REPAIR(model)
    components = model.n_components
    model.transmat_ = np.full((components, components), 1.0 / components)


def no_covar_repair(model):
    """Emulates 'repair transitions but not covariances'."""
    before = np.array(model._covars_, dtype=float, copy=True)
    ORIGINAL_REPAIR(model)
    model._covars_ = before


def no_start_repair(model):
    """Emulates 'repair transitions/covariances but not start probabilities'."""
    before = np.array(model.startprob_, dtype=float, copy=True)
    ORIGINAL_REPAIR(model)
    model.startprob_ = before


def old_per_ulp_scale(budgets, cash, min_notional=1.0):
    """The pre-fix compensation loop, verbatim in behaviour."""
    import math

    if not np.isfinite(cash) or cash <= 0:
        return {}
    budgets = {
        symbol: float(value)
        for symbol, value in budgets.items()
        if np.isfinite(value) and value > 0
    }
    usable = max(0.0, cash - max(1e-6, abs(cash) * 1e-9))
    if not budgets or usable <= 0:
        return {}
    total = math.fsum(budgets.values())
    if not np.isfinite(total) or total <= 0:
        return {}
    if total > usable:
        factor = usable / total
        budgets = {symbol: value * factor for symbol, value in budgets.items()}
    while math.fsum(budgets.values()) > usable:
        largest = max(budgets, key=budgets.get)
        budgets[largest] = np.nextafter(budgets[largest], 0.0)
    return {symbol: value for symbol, value in budgets.items()
            if value >= min_notional}


class NoTimezoneGuard(HMMVolumeStrategy):
    """The timezone guard removed and the date silently re-interpreted."""

    def _check_signal(self, signal_date):
        if signal_date is None or pd.isna(signal_date):
            raise ValueError("Signal date must be a valid timestamp")
        signal = pd.Timestamp(signal_date)
        if signal.tz is not None:
            signal = signal.tz_localize(None)   # the mutant's assumption
        if signal > self.benchmark.index[-1]:
            raise ValueError("data ended")
        return signal


class MutationHarnessChecks(unittest.TestCase):
    """Each mutant must escape the new test, i.e. the test must go red."""

    def test_uniform_transmat_mutant_escapes_the_identity_check(self):
        with mock.patch.object(hmm_module, "_repair_model", uniform_transmat_repair):
            outcome = repair_identity_outcome(hmm_module)
        self.assertFalse(
            outcome["unchanged"],
            "the mutant must change a valid model; otherwise "
            "test_repair_leaves_a_valid_fitted_model_untouched is vacuous",
        )
        np.testing.assert_allclose(
            outcome["after"]["transmat"],
            np.full((2, 2), 0.5),
            err_msg="the emulated mutant must flatten the transition matrix",
        )

    def test_missing_covar_repair_escapes_the_covariance_check(self):
        with mock.patch.object(hmm_module, "_repair_model", no_covar_repair):
            outcome = repair_covar_outcome(hmm_module)
        diagonal = outcome["diagonal"]
        self.assertFalse(
            np.isfinite(diagonal).all() and (diagonal > 0).all(),
            "the mutant must leave the degenerate diagonal in place; otherwise "
            "test_repair_returns_a_positive_finite_covariance_diagonal is "
            "vacuous",
        )

    def test_missing_start_repair_escapes_the_startprob_check(self):
        with mock.patch.object(hmm_module, "_repair_model", no_start_repair):
            outcome = repair_start_outcome(hmm_module)
        self.assertAlmostEqual(float(outcome["startprob"].sum()), 0.0)
        self.assertFalse(
            np.allclose(outcome["startprob"], 1.0 / 3.0),
            "the mutant must keep the all-zero start probability",
        )

    def test_per_ulp_loop_escapes_the_bitwise_equality_check(self):
        outcome = scale_equal_outcome(old_per_ulp_scale)
        self.assertGreater(
            outcome["distinct"],
            1,
            "the old loop must make equal budgets unequal; otherwise "
            "test_equal_budgets_stay_bitwise_equal_after_scaling is vacuous",
        )

    def test_missing_timezone_guard_escapes_the_signal_contract(self):
        outcome = tz_signal_outcome(NoTimezoneGuard)
        self.assertEqual(
            set(outcome["results"].values()),
            {None},
            "the mutant must answer tz-aware dates; otherwise "
            "test_timezone_aware_signal_is_rejected_with_value_error is vacuous",
        )


if __name__ == "__main__":
    unittest.main()
