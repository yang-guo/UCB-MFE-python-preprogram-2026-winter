"""Round-2 adversarial *mutation* audit for ``strategies.hmm_volume``.

Round 1 (``test_hmm_volume_agent1..4``) attacked causality, cash and the refit
lifecycle.  This module attacks the *test suite itself*: it injects deliberate
bugs into a temporary copy of the production source and asks whether the
existing tests go red.  No production file and no existing test is modified.

Audit result (33 mutants, full ``test_hmm_volume*`` suite after each one):

killed
    1  ``refit`` fits on ``returns`` instead of ``returns[:-1]``  (in-sample leak)
    2  ``_bars_at`` uses ``end < signal``                        (drops the signal bar)
    3  ``_bars_at`` uses ``start <= signal``                     (straddling/future bar)
    4  ``_check_signal`` guard removed                           (1 test kills it)
    5  state scored from bars dated far after the signal         (future state)
    6  state update skipped between refits
    7  state scored on the previous bar                          (one-bar lag)
    8  refit cadence ``>=`` -> ``>``                             (one bar late)
    9  refit cadence ``>=`` -> ``>= cadence - 1``                (one bar early)
    10 ``_scale_to_cash`` spends ``1.02 x cash``                 (overspend)
    11 ``_scale_to_cash`` ignores ``min_notional``
    12 sizing uses ``portfolio.cash`` instead of ``available_cash``
    13 default ``risk_on_states`` becomes ``range(0, n_states)``
    14 ``risk_on_states`` validation removed
    15 the risk-on gate ignored
    16 ``top_n`` basket under-deploys (``cash / (n + 1)``)
    17 liquidity ranking sorted ascending
    18 ``liquidity_window`` ignored and the full panel used
    19 ``_fit_bar_count`` off by one
    20 fit shifted to ``returns[1:]``
    21 ``deepcopy`` shares the fitted model object
    22 bar target uses a full-sample median                     (future volume)
    23 ``_bars_at`` returns every bar, future included
    24 ``min_train_bars`` validation weakened
    25 research reference ``walk_forward_states`` trains on ``values[:t+1]``
    26 research reference returns unranked posteriors
    27 research state labels reversed
    28 research refit cadence off by one

SURVIVED (the gaps this module closes)
    A  ``refit``'s backward-date guard removed
    B  ``propose_buys``' backward-date guards removed
    C  ``propose_buys``' future ``context.prices`` guard removed
    D  ``propose_buys``' future ``context.volume`` guard removed
    E  ``refit``'s minimum-bar guard weakened (``min_train_bars + 1`` ->
       ``min_train_bars - 1``); only reachable with ``min_train_bars > 20``

Every surviving mutant slips past the whole existing suite.  Two of them
(``propose_buys`` backward dates, future price/volume panels) are silent
lookaheads: the strategy answers a decision dated in the past with a state or a
quote from the future.  The other two (backward ``refit``, weak minimum-bar
guard) are contract violations that let the model be fitted at a date that has
already been answered, or on one observation too few.

The tests below pin both the documented contract (``ValueError``) and the
underlying invariant (no decision may rest on data dated after
``context.signal_date``; a fit always has one scored bar plus a nonempty
training sample).  ``MutationHarnessChecks`` proves they are not vacuous by
replaying each scenario against a mutated in-memory copy of the module and
asserting the mutant *escapes* (i.e. the regression test would go red).

With this module added, all 33 mutants kill the suite; the only failures come
from the tests in this file.  Note that the harness tests load the production
source, so when the production tree itself is mutated they fail too - that is
expected and is still a red suite.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import strategies.hmm_volume as hmm_volume
from backtesting import DecisionContext, Portfolio
from strategies.hmm_volume import HMMVolumeStrategy


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def synthetic_market(periods=600, symbols=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    benchmark = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, periods)))
    volume = pd.Series(rng.lognormal(15, 0.3, periods), index=dates)
    rows = []
    for symbol in [f"S{i}" for i in range(symbols)]:
        path = benchmark * np.exp(np.cumsum(rng.normal(0, 0.005, periods)))
        rows.append(pd.DataFrame({"date": dates, "symbol": symbol, "adj_close": path}))
    prices = pd.concat(rows, ignore_index=True)
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
    return dict(
        dates=dates,
        benchmark=pd.Series(benchmark, index=dates),
        volume=volume,
        prices=prices,
        membership=membership,
    )


MARKET = synthetic_market()
DATES = MARKET["dates"]
BENCHMARK = MARKET["benchmark"]
VOLUME = MARKET["volume"]
PRICES = MARKET["prices"]
MEMBERSHIP = MARKET["membership"]

OPTIONS = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=20,
    refit_every_bars=5,
    n_init=1,
    n_iter=20,
    seed=3,
    top_n=2,
    risk_on_states=(1, 2),
)


def build(cls=HMMVolumeStrategy, **overrides):
    options = dict(OPTIONS)
    options.update(overrides)
    return cls(BENCHMARK, VOLUME, **options)


def price_panel(signal, *, extra_session=None):
    """Wide price view through ``signal`` (optionally one session beyond)."""
    signal = pd.Timestamp(signal)
    last = pd.Timestamp(extra_session) if extra_session is not None else signal
    view = PRICES.loc[PRICES["date"] <= last].pivot(
        index="date", columns="symbol", values="adj_close"
    )
    return view.loc[:last] if extra_session is not None else view.loc[:signal]


def context(signal, *, prices=None, volume=None, cash=100_000.0):
    signal = pd.Timestamp(signal)
    if prices is None:
        prices = price_panel(signal)
    return DecisionContext(
        signal,
        signal + pd.Timedelta(days=1),
        prices,
        MEMBERSHIP.copy(),
        pd.DataFrame(),
        Portfolio(cash, cash, cash, {}),
        0,
        pd.DataFrame() if volume is None else volume,
        pd.DataFrame(),
    )


def next_session(signal):
    position = DATES.get_loc(pd.Timestamp(signal))
    return DATES[position + 1]


def mid_bar_session(bars, index):
    """A session strictly inside bar ``index + 1``: after bar ``index`` ends."""
    span = DATES[
        (DATES > bars["end"].iloc[index]) & (DATES < bars["end"].iloc[index + 1])
    ]
    if not len(span):  # pragma: no cover - fixture guard
        raise AssertionError("the fixture needs a multi-session bar")
    return span[0]


def forced(model, *, bar_index=60, state=2):
    """Pin the regime so the allocation path runs without a walk-forward fit."""
    bars = model._bars
    model._fit = object()
    model._state = int(state)
    model._state_bar_count = bar_index + 1
    model._state_bar_end = bars["end"].iloc[bar_index]
    return model


# --------------------------------------------------------------------------- #
# scenarios, shared by the regression tests and the mutation harness
# --------------------------------------------------------------------------- #
def backward_refit_outcome(cls):
    """Refit at a mid-bar session, then refit one bar back with the same bars."""
    model = build(cls)
    bars = model._bars
    index = model.min_train_bars + 3
    later = mid_bar_session(bars, index)
    earlier = bars["end"].iloc[index]
    model.refit(None, context(later))
    before = dict(
        fit=model._fit,
        fit_date=model._fit_date,
        fit_bar_count=model._fit_bar_count,
        state=model.state,
        state_date=model.state_date,
        probabilities=model.probabilities,
    )
    raised = False
    try:
        model.refit(None, context(earlier))
    except ValueError:
        raised = True
    return dict(
        model=model,
        raised=raised,
        earlier=earlier,
        later=later,
        # The later signal sits inside the next bar, so both dates see the
        # *same* bars: the bar-count guard cannot fire, only the date guard can.
        same_bar_set=len(model._bars_at(earlier)) == len(model._bars_at(later)),
        before=before,
        after=dict(
            fit=model._fit,
            fit_date=model._fit_date,
            fit_bar_count=model._fit_bar_count,
            state=model.state,
            state_date=model.state_date,
            probabilities=model.probabilities,
        ),
    )


def backward_propose_outcome(cls, guard):
    """Propose on a signal before the fitted date, or before the state date."""
    bars = build(cls)._bars
    if guard == "fit_date":
        # A real refit whose signal sits inside the next bar: the fit date is
        # after the bar end it scored, so the state itself is legitimate but
        # the decision would be dated before the fit.
        model = build(cls)
        index = model.min_train_bars + 3
        later = mid_bar_session(model._bars, index)
        model.refit(None, context(later))
        model._state = 2          # pin the regime so a mutant would trade
        signal = model._bars["end"].iloc[index]
    elif guard == "state_date":
        model = forced(build(cls), bar_index=60)
        model._fit_date = bars["end"].iloc[55]
        signal = bars["end"].iloc[58]
    else:  # pragma: no cover - programming error
        raise AssertionError(guard)
    raised, budgets = False, {}
    try:
        budgets = model.propose_buys(context(signal))
    except ValueError:
        raised = True
    return dict(
        model=model,
        raised=raised,
        budgets=budgets,
        signal=pd.Timestamp(signal),
        fit_date=model._fit_date,
        state_date=model.state_date,
        state=model.state,
        state_bar_count=model._state_bar_count,
    )


def future_price_outcome(cls):
    """Hand over a price panel with one row after the signal date."""
    model = forced(build(cls, top_n=None), bar_index=80)
    signal = model._bars["end"].iloc[80]
    later = next_session(signal)
    panel = price_panel(later)          # deliberately one session too long
    # S3 has no quote at the signal but does have one the next session; without
    # the guard it silently becomes eligible.
    panel.loc[:, "S3"] = np.nan
    panel.loc[later, "S3"] = float(panel["S0"].iloc[-1])
    raised, budgets = False, {}
    try:
        budgets = model.propose_buys(context(signal, prices=panel))
    except ValueError:
        raised = True
    return dict(
        model=model, raised=raised, budgets=budgets, signal=signal, later=later
    )


def future_volume_outcome(cls):
    """Hand over a volume panel with one row after the signal date."""
    model = forced(build(cls, top_n=1), bar_index=80)
    signal = model._bars["end"].iloc[80]
    later = next_session(signal)
    panel = pd.DataFrame(1.0, index=price_panel(later).index, columns=["S0", "S1", "S2", "S3"])
    panel.loc[later, "S3"] = 1e9      # only the future row makes S3 liquid
    raised, budgets = False, {}
    try:
        budgets = model.propose_buys(context(signal, prices=price_panel(signal), volume=panel))
    except ValueError:
        raised = True
    return dict(
        model=model, raised=raised, budgets=budgets, signal=signal, later=later
    )


def refit_min_bars_outcome(cls):
    """Refit with exactly ``min_train_bars`` bars: one train bar is missing."""
    model = build(cls, min_train_bars=25)
    bars = model._bars
    signal = bars["end"].iloc[24]                 # 25 completed bars
    raised = False
    try:
        model.refit(None, context(signal))
    except ValueError:
        raised = True
    return dict(
        model=model,
        raised=raised,
        signal=signal,
        bar_count=len(model._bars_at(signal)),
        fitted=model._fit is not None,
    )


# --------------------------------------------------------------------------- #
# regression tests: kill the surviving mutants
# --------------------------------------------------------------------------- #
class BackwardTimeGuardChecks(unittest.TestCase):
    """A decision may never be answered with data dated after its signal."""

    def test_refit_refuses_an_earlier_signal_date_even_with_the_same_bars(self):
        outcome = backward_refit_outcome(HMMVolumeStrategy)
        self.assertTrue(
            outcome["same_bar_set"],
            "the scenario must not be vacuous: both dates have to see the same "
            "bars, otherwise the bar-count guard, not the date guard, fires",
        )
        self.assertTrue(outcome["raised"], msg="a backward refit was accepted")
        before, after = outcome["before"], outcome["after"]
        self.assertIs(after["fit"], before["fit"], "the fit was replaced")
        self.assertEqual(after["fit_date"], before["fit_date"])
        self.assertEqual(after["fit_bar_count"], before["fit_bar_count"])
        self.assertEqual(after["state"], before["state"])
        self.assertEqual(after["state_date"], before["state_date"])
        np.testing.assert_array_equal(after["probabilities"], before["probabilities"])

    def test_propose_buys_refuses_a_signal_before_the_fitted_date(self):
        outcome = backward_propose_outcome(HMMVolumeStrategy, "fit_date")
        self.assertLess(outcome["signal"], outcome["model"]._fit_date)
        self.assertTrue(outcome["raised"], msg="a pre-fit signal was answered")
        self.assertEqual(outcome["budgets"], {})

    def test_propose_buys_refuses_a_signal_before_the_state_date(self):
        outcome = backward_propose_outcome(HMMVolumeStrategy, "state_date")
        # Guard 1 is satisfied: the signal is after the fitted date.  Only the
        # state-date guard can stop the stale (i.e. future) state from trading.
        self.assertGreaterEqual(outcome["signal"], outcome["model"]._fit_date)
        self.assertLess(outcome["signal"], outcome["state_date"])
        self.assertTrue(outcome["raised"], msg="a pre-state signal was answered")
        self.assertEqual(outcome["budgets"], {})

    def test_no_decision_rests_on_a_state_from_after_the_signal(self):
        """Implementation-independent form of both ``propose_buys`` guards."""
        model = forced(build(), bar_index=60)
        bars = model._bars
        model._fit_date = bars["end"].iloc[55]
        state_date, state, state_bar_count = model.state_date, model.state, model._state_bar_count
        for offset in range(1, 11):
            signal = bars["end"].iloc[60 - offset]
            with self.subTest(offset=offset, signal=signal):
                raised, budgets = False, {}
                try:
                    budgets = model.propose_buys(context(signal))
                except ValueError:
                    raised = True
                if not raised:
                    self.assertEqual(
                        budgets, {},
                        "a signal dated before the last scored bar received "
                        "buy orders sized off a state from the future",
                    )
                    self.assertLessEqual(
                        model.state_date, signal,
                        "the state advanced past the signal date",
                    )
                self.assertEqual(model.state_date, state_date)
                self.assertEqual(model.state, state)
                self.assertEqual(model._state_bar_count, state_bar_count)

    def test_propose_buys_refuses_a_price_panel_reaching_past_the_signal(self):
        outcome = future_price_outcome(HMMVolumeStrategy)
        self.assertTrue(
            outcome["raised"],
            "the context.prices panel carried a session after the signal and "
            "the strategy answered anyway",
        )
        self.assertEqual(outcome["budgets"], {})

    def test_propose_buys_refuses_a_volume_panel_reaching_past_the_signal(self):
        outcome = future_volume_outcome(HMMVolumeStrategy)
        self.assertTrue(
            outcome["raised"],
            "the context.volume panel carried a session after the signal and "
            "the strategy ranked on it",
        )
        self.assertEqual(outcome["budgets"], {})

    def test_refit_needs_one_more_bar_than_min_train_bars(self):
        """``refit`` scores one bar and trains on the rest, always."""
        outcome = refit_min_bars_outcome(HMMVolumeStrategy)
        self.assertEqual(outcome["bar_count"], 25)
        self.assertTrue(
            outcome["raised"],
            "refit accepted min_train_bars bars, so it would train on "
            "min_train_bars - 1 observations and still score a bar",
        )
        self.assertFalse(outcome["fitted"], "a refused refit left a fitted model")

        # The boundary is exact: one more bar fits, and the training sample is
        # exactly the bars before the scored one.
        model = build(min_train_bars=25)
        bars = model._bars
        signal = bars["end"].iloc[25]
        model.refit(None, context(signal))
        returns = model._bars_at(signal)["log_return"].to_numpy(dtype=float)
        self.assertEqual(model._bar_returns.size, returns.size)
        self.assertEqual(returns.size, 26)
        reference = hmm_volume.fit_hmm(
            returns[:-1], n_states=3, n_init=1, seed=3, n_iter=20
        )
        np.testing.assert_allclose(model._fit.model.means_, reference.model.means_, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# mutation harness: prove the tests above are not vacuous
# --------------------------------------------------------------------------- #
BACKWARD_REFIT_GUARD = (
    "        if self._fit_date is not None and signal < self._fit_date:\n"
    '            raise ValueError("Refit signal date cannot move backwards")\n',
    "",
)
BACKWARD_PROPOSE_FIT_DATE_GUARD = (
    "        if self._fit_date is not None and signal < self._fit_date:\n"
    '            raise ValueError("Signal date moved backwards")\n',
    "",
)
BACKWARD_PROPOSE_STATE_DATE_GUARD = (
    "        if self._state_bar_end is not None and signal < self._state_bar_end:\n"
    '            raise ValueError("Signal date moved backwards")\n',
    "",
)
FUTURE_PRICE_GUARD = (
    "        if index.max() > signal:\n"
    '            raise ValueError(f"context.{name} contains data after the signal date")\n',
    "",
)
FUTURE_VOLUME_GUARD = FUTURE_PRICE_GUARD
MIN_BARS_GUARD = (
    "        if len(bars) < self.min_train_bars + 1:\n"
    '            raise ValueError("Not enough completed bars to fit the HMM")',
    "        if len(bars) < self.min_train_bars - 1:\n"
    '            raise ValueError("Not enough completed bars to fit the HMM")',
)


class MutationHarnessChecks(unittest.TestCase):
    """Replay each scenario against a mutated in-memory copy of the module.

    A harness test passes when the mutant *escapes* the guard, i.e. when the
    corresponding regression test above would go red.  That is the evidence the
    new test measures the guard and not something else.
    """

    def setUp(self):
        self._tmpdirs = []

    def tearDown(self):
        for path in self._tmpdirs:
            shutil.rmtree(path, ignore_errors=True)

    def load(self, patches):
        """Return a module object built from a patched copy of the real source."""
        source_path = hmm_volume.__file__
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        for old, new in patches:
            count = source.count(old)
            self.assertEqual(
                count, 1,
                msg=f"mutation pattern occurs {count}x; update the harness:\n{old}",
            )
            source = source.replace(old, new)
        directory = tempfile.mkdtemp(prefix="hmm_volume_mutant_")
        self._tmpdirs.append(directory)
        path = os.path.join(directory, "mutant_hmm_volume.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        name = "mutant_hmm_volume_" + str(abs(hash(source)))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_control_unmutated_copy_behaves_like_production(self):
        """The loader itself must not change behaviour (non-vacuity control)."""
        module = self.load(())
        self.assertIsNot(module.HMMVolumeStrategy, HMMVolumeStrategy)
        self.assertTrue(backward_refit_outcome(module.HMMVolumeStrategy)["raised"])
        self.assertTrue(
            backward_propose_outcome(module.HMMVolumeStrategy, "fit_date")["raised"]
        )
        self.assertTrue(
            backward_propose_outcome(module.HMMVolumeStrategy, "state_date")["raised"]
        )
        self.assertTrue(future_price_outcome(module.HMMVolumeStrategy)["raised"])
        self.assertTrue(future_volume_outcome(module.HMMVolumeStrategy)["raised"])
        self.assertTrue(refit_min_bars_outcome(module.HMMVolumeStrategy)["raised"])

    def test_backward_refit_guard_is_what_the_boundary_test_detects(self):
        module = self.load([BACKWARD_REFIT_GUARD])
        outcome = backward_refit_outcome(module.HMMVolumeStrategy)
        self.assertTrue(outcome["same_bar_set"])
        self.assertFalse(
            outcome["raised"],
            "the mutated module must escape the guard for this harness to prove "
            "that test_refit_refuses_an_earlier_signal_date_even_with_the_same_bars "
            "detects the removed guard",
        )
        # Observable damage: the audit date silently rewinds, so a later
        # ``propose_buys`` on that earlier date is accepted too.
        self.assertLess(outcome["after"]["fit_date"], outcome["before"]["fit_date"])
        self.assertEqual(outcome["after"]["fit_bar_count"], outcome["before"]["fit_bar_count"])

    def test_backward_propose_fit_date_guard_is_detected(self):
        module = self.load([BACKWARD_PROPOSE_FIT_DATE_GUARD])
        outcome = backward_propose_outcome(module.HMMVolumeStrategy, "fit_date")
        self.assertFalse(
            outcome["raised"],
            "the mutant must answer the pre-fit signal; otherwise "
            "test_propose_buys_refuses_a_signal_before_the_fitted_date is vacuous",
        )
        self.assertTrue(outcome["budgets"], "the mutant did not even trade")
        self.assertGreater(
            outcome["fit_date"], outcome["signal"],
            "the answered decision must precede the date the model was fitted at",
        )

    def test_backward_propose_state_date_guard_is_detected(self):
        module = self.load([BACKWARD_PROPOSE_STATE_DATE_GUARD])
        outcome = backward_propose_outcome(module.HMMVolumeStrategy, "state_date")
        self.assertFalse(
            outcome["raised"],
            "the mutant must answer the pre-state signal; otherwise "
            "test_propose_buys_refuses_a_signal_before_the_state_date is vacuous",
        )
        self.assertTrue(outcome["budgets"])
        self.assertGreater(
            outcome["state_date"], outcome["signal"],
            "the mutant traded off a state computed from a later bar",
        )

    def test_future_price_guard_is_detected(self):
        module = self.load([FUTURE_PRICE_GUARD])
        outcome = future_price_outcome(module.HMMVolumeStrategy)
        self.assertFalse(
            outcome["raised"],
            "the mutant must accept the over-long price panel; otherwise "
            "test_propose_buys_refuses_a_price_panel_reaching_past_the_signal "
            "is vacuous",
        )
        self.assertIn(
            "S3", outcome["budgets"],
            "the mutant must actually buy the name whose only quote is in the "
            "future, which is the lookahead the guard prevents",
        )

    def test_future_volume_guard_is_detected(self):
        module = self.load([FUTURE_VOLUME_GUARD])
        outcome = future_volume_outcome(module.HMMVolumeStrategy)
        self.assertFalse(
            outcome["raised"],
            "the mutant must accept the over-long volume panel; otherwise "
            "test_propose_buys_refuses_a_volume_panel_reaching_past_the_signal "
            "is vacuous",
        )
        self.assertEqual(
            set(outcome["budgets"]), {"S3"},
            "the mutant must rank on the future volume row",
        )

    def test_min_bars_guard_is_detected(self):
        module = self.load([MIN_BARS_GUARD])
        outcome = refit_min_bars_outcome(module.HMMVolumeStrategy)
        self.assertFalse(
            outcome["raised"],
            "the mutant must accept min_train_bars bars; otherwise "
            "test_refit_needs_one_more_bar_than_min_train_bars is vacuous",
        )
        self.assertTrue(
            outcome["fitted"],
            "with min_train_bars=25 the shortened sample (24) still satisfies "
            "fit_hmm, so the mutant fits quietly",
        )


if __name__ == "__main__":
    unittest.main()
