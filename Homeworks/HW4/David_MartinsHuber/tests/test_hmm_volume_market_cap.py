"""Point-in-time market-cap selection for ``HMMVolumeStrategy``.

What these tests pin down
-------------------------
When ``market_cap`` is supplied, stock selection and basket weights must come
from the *latest snapshot that is actually known at the signal date*:

1. ``top_n`` keeps the highest ``market_cap_proxy`` names and the budget of
   each name is proportional to its cap.
2. A stock volume panel that favours other names cannot change the basket or
   the weights; volume belongs to the HMM bar aggregation only.
3. A snapshot row whose ``selection_date`` is after the signal cannot be used,
   and cannot erase an older row that *was* known (causality).
4. Names absent from the latest snapshot, or with NaN/nonpositive/infinite
   caps, are excluded rather than resurrected from an older quarter.
5. ``top_n=None`` and an oversized ``top_n`` keep every eligible cap name.
6. Budgets stay within ``available_cash`` and never fall below
   ``min_order_notional``.
7. With no ``market_cap`` the legacy liquidity fallback is deterministic and
   does not crash, including with a missing/partial volume panel.
8. ``refit_settings`` reports the market-cap weighting mode and cap row count.

Fixtures
--------
Everything is synthetic and deterministic: a 300-session business-day
calendar, a GBM benchmark, and six symbols with fixed seeds.  No database is
touched.

Most tests force the strategy into a risk-on state (``_fit``/``_state``) so the
expected budgets are pure arithmetic and the HMM fit cannot make the test
flaky.  ``test_refit_and_propose_uses_caps_end_to_end`` instead runs a real
``refit`` with ``risk_on_states=(0, 1, 2)``, which makes the state irrelevant
and still exercises the production code path.

``min_order_notional`` is set to zero where a test needs every selected name
to survive, because the production contract drops budgets below the floor
(covered explicitly in ``test_budgets_respect_cash_and_min_notional``).
"""

from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from backtesting import DecisionContext, Portfolio
from strategies.hmm_volume import HMMVolumeStrategy, _market_cap_frame

SYMBOLS = tuple(f"S{i}" for i in range(6))
Q1 = pd.Timestamp("2015-01-01")
Q2 = pd.Timestamp("2015-04-01")
Q3 = pd.Timestamp("2015-07-01")
SIGNAL_Q1 = pd.Timestamp("2015-02-15")
SIGNAL_Q2 = pd.Timestamp("2015-05-15")
SIGNAL_Q3 = pd.Timestamp("2015-07-15")

STRATEGY_OPTIONS = dict(
    median_window=20,
    bucket_multiple=5.0,
    n_states=3,
    min_train_bars=30,
    refit_every_bars=10,
    n_init=1,
    n_iter=25,
    seed=7,
)


def market(periods=300, seed=11):
    """Deterministic benchmark, volume, price panel, and daily stock volume."""
    dates = pd.bdate_range(Q1, periods=periods)
    rng = np.random.default_rng(seed)
    benchmark = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, periods))),
        index=dates,
        name="benchmark",
    )
    volume = pd.Series(
        rng.lognormal(15.0, 0.25, periods), index=dates, name="volume"
    )
    prices = pd.DataFrame(
        {
            symbol: 100 * np.exp(np.cumsum(rng.normal(0.0, 0.006, periods)))
            for symbol in SYMBOLS
        },
        index=dates,
    )
    stock_volume = pd.DataFrame(
        {symbol: rng.lognormal(14.0, 0.4, periods) for symbol in SYMBOLS},
        index=dates,
    )
    return dates, benchmark, volume, prices, stock_volume


def snapshot(quarter, selection, caps, symbols=None):
    """Build a membership/market-cap snapshot from a ``symbol -> cap`` dict."""
    symbols = list(caps) if symbols is None else list(symbols)
    return pd.DataFrame(
        {
            "symbol": symbols,
            "quarter_start": quarter,
            "selection_date": pd.Timestamp(selection),
            "market_cap_proxy": [caps[symbol] for symbol in symbols],
        }
    )


def members(quarter, selection, symbols=SYMBOLS):
    """The membership rows the engine would hand the strategy."""
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "quarter_start": quarter,
            "selection_date": pd.Timestamp(selection),
        }
    )


def decision(
    signal,
    *,
    membership,
    prices,
    volume=None,
    available_cash=1000.0,
    cash=None,
):
    """A causal ``DecisionContext``; panels are cut at the signal date."""
    signal = pd.Timestamp(signal)
    nav = available_cash if cash is None else cash
    return DecisionContext(
        signal,
        signal + pd.Timedelta(days=1),
        prices.loc[:signal].copy(),
        membership.copy(),
        pd.DataFrame(),
        Portfolio(nav, available_cash, nav, {}),
        0,
        pd.DataFrame() if volume is None else volume.loc[:signal].copy(),
        pd.DataFrame(),
    )


def forced(caps=None, **overrides):
    """Strategy in a forced risk-on state, ready for deterministic proposals."""
    options = dict(STRATEGY_OPTIONS)
    options.update(overrides)
    _, benchmark, volume, _, _ = MARKET
    model = HMMVolumeStrategy(benchmark, volume, market_cap=caps, **options)
    # Only ``_fit``/``_state`` are mocked: ``propose_buys`` then runs its real
    # selection/weighting/cash code.  Setting the bar counter to the full bar
    # history stops ``_update_state`` from calling the dummy fit.
    model._fit = object()
    model._state = 2
    model._state_bar_count = len(model._bars)
    return model


MARKET = market()


class MarketCapSelectionTests(unittest.TestCase):
    """Cap ranking, PIT snapshot choice, exclusions, sizing, and settings."""

    @classmethod
    def setUpClass(cls):
        cls.dates, cls.benchmark, cls.hmm_volume, cls.prices, cls.stock_volume = MARKET
        cls.caps = dict(zip(SYMBOLS, [10.0, 20.0, 30.0, 40.0, 500.0, 1000.0]))
        cls.q1_snapshot = snapshot(Q1, "2014-12-31", cls.caps)
        cls.q1_members = members(Q1, "2014-12-31")

    # ---------------------------------------------------------------- helpers
    def assert_cap_proportional(self, budgets, caps, cash, delta=1e-4):
        """Budgets are ``cash * cap / sum(selected caps)`` within rounding."""
        total = float(sum(caps.values()))
        self.assertEqual(set(budgets), set(caps))
        for symbol, cap in caps.items():
            self.assertAlmostEqual(
                budgets[symbol], cash * cap / total, delta=delta,
                msg=f"{symbol} budget is not proportional to its cap",
            )

    def expected_weights(self, caps):
        total = float(sum(caps.values()))
        return {symbol: cap / total for symbol, cap in caps.items()}

    # ------------------------------------------------------- 1. cap ranking
    def test_top_n_selects_highest_caps_and_weights_by_cap(self):
        model = forced(self.q1_snapshot, top_n=2, min_order_notional=0.0)
        budgets = model.propose_buys(
            decision(
                SIGNAL_Q1, membership=self.q1_members, prices=self.prices
            )
        )
        selected = {"S5": 1000.0, "S4": 500.0}
        self.assertEqual(set(budgets), set(selected))
        self.assert_cap_proportional(budgets, selected, 1000.0)

        # The cap order, not the symbol order, decides: S0..S3 are excluded.
        self.assertEqual(sorted(budgets, reverse=True), ["S5", "S4"])
        self.assertGreater(budgets["S5"], budgets["S4"])
        self.assertLessEqual(sum(budgets.values()), 1000.0)

    def test_cap_ranking_is_independent_of_snapshot_row_order(self):
        model = forced(self.q1_snapshot, top_n=3, min_order_notional=0.0)
        context = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices
        )
        baseline = model.propose_buys(context)

        shuffled = self.q1_snapshot.sample(frac=1.0, random_state=3).reset_index(drop=True)
        other = forced(shuffled, top_n=3, min_order_notional=0.0)
        self.assertEqual(other.propose_buys(context), baseline)
        self.assertEqual(set(baseline), {"S5", "S4", "S3"})

    # --------------------------------------------- 2. volume cannot rank
    def test_stock_volume_never_changes_cap_basket_or_weights(self):
        favour_first = pd.DataFrame(0.0, index=self.dates, columns=SYMBOLS)
        favour_first["S0"] = 1e12
        favour_last = pd.DataFrame(0.0, index=self.dates, columns=SYMBOLS)
        favour_last["S3"] = 1e12
        favour_many = pd.DataFrame(1.0, index=self.dates, columns=SYMBOLS)
        favour_many["S1"] = 1e12

        model = forced(self.q1_snapshot, top_n=2, min_order_notional=0.0)
        base = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices
        )
        baseline = model.propose_buys(base)
        self.assertEqual(set(baseline), {"S5", "S4"})

        for label, panel in (
            ("first", favour_first),
            ("last", favour_last),
            ("many", favour_many),
            ("none", None),
        ):
            with self.subTest(volume=label):
                context = decision(
                    SIGNAL_Q1,
                    membership=self.q1_members,
                    prices=self.prices,
                    volume=panel,
                )
                self.assertEqual(model.propose_buys(context), baseline)

        # Column order of the volume panel is irrelevant too.
        reordered = decision(
            SIGNAL_Q1,
            membership=self.q1_members,
            prices=self.prices,
            volume=favour_first[list(reversed(SYMBOLS))],
        )
        self.assertEqual(model.propose_buys(reordered), baseline)

    # ------------------------------------------- 3. point-in-time snapshots
    def test_latest_visible_snapshot_is_used_and_future_snapshot_ignored(self):
        q2_caps = {"S0": 100.0, "S1": 400.0, "S2": 200.0,
                   "S3": 5.0, "S4": 5.0, "S5": 5.0}
        q3_caps = {symbol: 1.0 for symbol in SYMBOLS}
        q3_caps["S0"] = 10_000.0
        frame = pd.concat(
            [
                self.q1_snapshot,
                snapshot(Q2, "2015-03-31", q2_caps),
                snapshot(Q3, "2015-06-30", q3_caps),
            ],
            ignore_index=True,
        )
        model = forced(frame, top_n=2, min_order_notional=0.0)
        q2_members = members(Q2, "2015-03-31")
        q3_members = members(Q3, "2015-06-30")

        # Q1 signal: only the Q1 snapshot is known.
        q1_budgets = model.propose_buys(
            decision(SIGNAL_Q1, membership=self.q1_members, prices=self.prices)
        )
        self.assert_cap_proportional(
            q1_budgets, {"S5": 1000.0, "S4": 500.0}, 1000.0
        )

        # Q2 signal: Q2 snapshot is known, Q3 is still in the future.
        q2_budgets = model.propose_buys(
            decision(SIGNAL_Q2, membership=q2_members, prices=self.prices)
        )
        self.assertEqual(set(q2_budgets), {"S1", "S2"})
        self.assert_cap_proportional(
            q2_budgets, {"S1": 400.0, "S2": 200.0}, 1000.0
        )

        # Q3 signal: the newest quarter takes over.
        q3_budgets = model.propose_buys(
            decision(SIGNAL_Q3, membership=q3_members, prices=self.prices)
        )
        self.assertEqual(set(q3_budgets), {"S0", "S1"})
        self.assertAlmostEqual(q3_budgets["S0"], 1000.0 * 10_000.0 / 10_001.0,
                               delta=1e-3)

    def test_future_snapshot_row_cannot_change_an_earlier_basket(self):
        """A row dated after the signal must not erase the row known at it.

        ``market_cap`` is a point-in-time frame: appending a later snapshot
        must leave every earlier decision untouched (the same rule as
        ``test_future_data_does_not_change_past_state_or_proposals`` for the
        benchmark/volume series).
        """
        base_frame = self.q1_snapshot
        revision = snapshot(Q1, "2015-03-25", {"S5": 1e9})
        revised_frame = pd.concat([base_frame, revision], ignore_index=True)

        context = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices
        )
        base = forced(base_frame, top_n=2, min_order_notional=0.0).propose_buys(context)
        revised = forced(revised_frame, top_n=2, min_order_notional=0.0).propose_buys(context)

        self.assertEqual(
            revised, base,
            "a snapshot row dated after the signal changed the earlier basket",
        )
        # And the row known at the signal (S5, selection 2014-12-31) is used.
        self.assertIn("S5", revised)
        self.assert_cap_proportional(
            revised, {"S5": 1000.0, "S4": 500.0}, 1000.0
        )

    def test_latest_row_on_or_before_signal_is_used(self):
        """Between two rows of one quarter, the visible one must win."""
        base_frame = self.q1_snapshot
        late = snapshot(Q1, "2015-03-25", {"S5": 1.0, "S4": 1.0})
        frame = pd.concat([base_frame, late], ignore_index=True)

        context = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices
        )
        budgets = forced(frame, top_n=2, min_order_notional=0.0).propose_buys(context)
        self.assert_cap_proportional(
            budgets, {"S5": 1000.0, "S4": 500.0}, 1000.0
        )

    # -------------------------------------------------- 4. exclusions
    def test_names_missing_from_latest_snapshot_are_excluded(self):
        q2_caps = {"S0": 100.0, "S1": 200.0, "S2": 300.0}
        frame = pd.concat(
            [self.q1_snapshot, snapshot(Q2, "2015-03-31", q2_caps)],
            ignore_index=True,
        )
        model = forced(frame, top_n=None, min_order_notional=0.0)
        budgets = model.propose_buys(
            decision(SIGNAL_Q2, membership=members(Q2, "2015-03-31"),
                     prices=self.prices)
        )
        # S3/S4/S5 still have much larger Q1 caps; a stale quarter must not
        # resurrect them.
        self.assertEqual(set(budgets), {"S0", "S1", "S2"})
        self.assert_cap_proportional(budgets, q2_caps, 1000.0)

    def test_bad_caps_in_latest_snapshot_are_excluded(self):
        q2 = snapshot(
            Q2,
            "2015-03-31",
            {"S0": float("nan"), "S1": 0.0, "S2": -5.0,
             "S3": 100.0, "S4": float("inf"), "S5": 50.0},
        )
        frame = pd.concat([self.q1_snapshot, q2], ignore_index=True)
        model = forced(frame, top_n=None, min_order_notional=0.0)
        budgets = model.propose_buys(
            decision(SIGNAL_Q2, membership=members(Q2, "2015-03-31"),
                     prices=self.prices)
        )
        # S0/S1/S2/S4 have invalid caps in the latest snapshot and are gone,
        # even though Q1 had usable caps for them.
        self.assertEqual(set(budgets), {"S3", "S5"})
        self.assert_cap_proportional(budgets, {"S3": 100.0, "S5": 50.0}, 1000.0)

    def test_names_without_prices_or_membership_are_excluded(self):
        caps = {symbol: float(100 + index) for index, symbol in enumerate(SYMBOLS)}
        frame = snapshot(Q1, "2014-12-31", caps)
        # Only S1 and S2 are both members and priced.
        membership = members(Q1, "2014-12-31", ("S1", "S2"))
        prices = self.prices[["S1", "S2"]]
        model = forced(frame, top_n=None, min_order_notional=0.0)
        budgets = model.propose_buys(
            decision(SIGNAL_Q1, membership=membership, prices=prices)
        )
        self.assertEqual(set(budgets), {"S1", "S2"})
        self.assert_cap_proportional(
            budgets, {"S1": 101.0, "S2": 102.0}, 1000.0
        )

    # ------------------------------------------------------ 5. top_n rules
    def test_top_n_none_and_oversized_top_n_use_all_eligible_caps(self):
        all_caps = dict(zip(SYMBOLS, [10.0, 20.0, 30.0, 40.0, 500.0, 1000.0]))
        context = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices
        )
        everything = forced(
            self.q1_snapshot, top_n=None, min_order_notional=0.0
        ).propose_buys(context)
        oversized = forced(
            self.q1_snapshot, top_n=99, min_order_notional=0.0
        ).propose_buys(context)

        self.assertEqual(set(everything), set(SYMBOLS))
        self.assert_cap_proportional(everything, all_caps, 1000.0)
        self.assertEqual(oversized, everything)
        self.assertLessEqual(sum(everything.values()), 1000.0)

        only_one = forced(
            self.q1_snapshot, top_n=1, min_order_notional=0.0
        ).propose_buys(context)
        self.assertEqual(set(only_one), {"S5"})
        self.assertAlmostEqual(only_one["S5"], 1000.0, delta=1e-4)

    # ------------------------------------------------------- 6. cash/floor
    def test_budgets_respect_cash_and_min_order_notional(self):
        context = decision(
            SIGNAL_Q1,
            membership=self.q1_members,
            prices=self.prices,
            available_cash=400.0,
            cash=1000.0,
        )
        model = forced(self.q1_snapshot, top_n=None, min_order_notional=1.0)
        budgets = model.propose_buys(context)
        self.assertLessEqual(sum(budgets.values()), 400.0)
        self.assertTrue(all(value >= 1.0 for value in budgets.values()))
        self.assert_cap_proportional(
            budgets,
            dict(zip(SYMBOLS, [10.0, 20.0, 30.0, 40.0, 500.0, 1000.0])),
            400.0,
        )

        # A cap-weighted budget below the floor is dropped, never rounded up.
        tiny = {"S0": 1e9, "S1": 0.5, "S2": 0.5}
        tiny_frame = snapshot(Q1, "2014-12-31", tiny)
        small_cash = decision(
            SIGNAL_Q1,
            membership=self.q1_members,
            prices=self.prices,
            available_cash=100.0,
        )
        small = forced(tiny_frame, top_n=None, min_order_notional=1.0).propose_buys(
            small_cash
        )
        self.assertEqual(set(small), {"S0"})
        self.assertAlmostEqual(small["S0"], 100.0, delta=1e-3)

        # Too little cash for even one lot at the floor: no orders at all.
        equal_frame = snapshot(Q1, "2014-12-31",
                               {"S0": 1.0, "S1": 1.0, "S2": 1.0})
        broke = forced(equal_frame, top_n=None, min_order_notional=2.0)
        self.assertEqual(
            broke.propose_buys(
                decision(SIGNAL_Q1, membership=self.q1_members,
                         prices=self.prices, available_cash=3.0)
            ),
            {},
        )

        for cash in (0.0, -5.0, float("nan"), float("inf")):
            with self.subTest(cash=cash):
                self.assertEqual(
                    model.propose_buys(
                        decision(SIGNAL_Q1, membership=self.q1_members,
                                 prices=self.prices, available_cash=cash)
                    ),
                    {},
                )

    # ------------------------------------------------ 7. legacy fallback
    def test_legacy_liquidity_fallback_is_deterministic(self):
        favour = pd.DataFrame(0.0, index=self.dates, columns=SYMBOLS)
        favour["S0"] = 1e12
        favour["S2"] = 5e6
        budget = forced(None, top_n=2, min_order_notional=0.0)
        context = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices,
            volume=favour,
        )
        first = budget.propose_buys(context)
        self.assertEqual(budget.propose_buys(context), first)
        self.assertEqual(set(first), {"S0", "S2"})
        self.assertLessEqual(sum(first.values()), 1000.0)

        # Row/column order must not change the dollar-volume ranking.
        shuffled = context.membership.sample(frac=1.0, random_state=5)
        reordered = decision(
            SIGNAL_Q1,
            membership=shuffled,
            prices=self.prices[list(reversed(SYMBOLS))],
            volume=favour[list(reversed(SYMBOLS))],
        )
        self.assertEqual(budget.propose_buys(reordered), first)

        # Without a volume panel the fallback degrades to the first ``top_n``
        # eligible symbols in sorted order, deterministically.
        without_volume = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices
        )
        no_volume = budget.propose_buys(without_volume)
        self.assertEqual(set(no_volume), {"S0", "S1"})
        self.assertEqual(budget.propose_buys(without_volume), no_volume)

        partial = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices,
            volume=favour[["S0"]],
        )
        self.assertEqual(set(budget.propose_buys(partial)), {"S0", "S1"})

        # Large/None top_n and a fresh but equal strategy agree exactly.
        wide = forced(None, top_n=None, min_order_notional=0.0)
        self.assertEqual(
            wide.propose_buys(context),
            forced(None, top_n=None, min_order_notional=0.0).propose_buys(context),
        )
        self.assertEqual(set(wide.propose_buys(context)), set(SYMBOLS))

    def test_fallback_and_cap_path_choose_different_names(self):
        """A volume panel that favours S0 cannot drag the cap basket to S0."""
        favour = pd.DataFrame(0.0, index=self.dates, columns=SYMBOLS)
        favour["S0"] = 1e12
        context = decision(
            SIGNAL_Q1, membership=self.q1_members, prices=self.prices,
            volume=favour,
        )
        cap_basket = forced(self.q1_snapshot, top_n=2,
                            min_order_notional=0.0).propose_buys(context)
        fallback_basket = forced(None, top_n=2,
                                 min_order_notional=0.0).propose_buys(context)
        self.assertEqual(set(cap_basket), {"S5", "S4"})
        self.assertIn("S0", fallback_basket)
        self.assertNotIn("S0", cap_basket)

    def test_supplied_caps_with_no_usable_rows_do_not_rank_by_volume(self):
        """The liquidity fallback is for absent caps, not unusable caps.

        Volume is documented as bar aggregation only once ``market_cap`` is
        supplied; a volume-ranked basket here would make the basket depend on
        the stock volume panel again.
        """
        favour_first = pd.DataFrame(0.0, index=self.dates, columns=SYMBOLS)
        favour_first["S0"] = 1e12
        favour_last = pd.DataFrame(0.0, index=self.dates, columns=SYMBOLS)
        favour_last["S3"] = 1e12

        outside_universe = snapshot(Q1, "2014-12-31", {"Z": 1e9})
        future_only = snapshot(Q2, "2015-03-31",
                               {symbol: 1e9 for symbol in SYMBOLS})

        for label, frame in (("outside", outside_universe),
                             ("future", future_only)):
            for panel_label, panel in (("first", favour_first),
                                       ("last", favour_last)):
                with self.subTest(caps=label, volume=panel_label):
                    model = forced(frame, top_n=2, min_order_notional=0.0)
                    budgets = model.propose_buys(
                        decision(SIGNAL_Q1, membership=self.q1_members,
                                 prices=self.prices, volume=panel)
                    )
                    self.assertEqual(
                        budgets, {},
                        "caps were supplied but unusable; volume still ranked "
                        "the basket",
                    )

    # ------------------------------------------------- 8. refit settings
    def test_refit_settings_report_weighting_mode(self):
        capped = forced(self.q1_snapshot, top_n=3, min_order_notional=2.5)
        settings = capped.refit_settings()
        self.assertEqual(settings["weighting"], "market_cap")
        self.assertEqual(settings["market_cap_rows"], len(_market_cap_frame(self.q1_snapshot)))
        self.assertEqual(settings["top_n"], 3)
        self.assertEqual(settings["min_order_notional"], 2.5)
        # The engine deepcopies strategies; caps are read-only shared inputs.
        self.assertIs(deepcopy(capped).market_cap, capped.market_cap)

        fallback = forced(None, top_n=2)
        fallback_settings = fallback.refit_settings()
        self.assertEqual(fallback_settings["weighting"], "liquidity_fallback")
        self.assertEqual(fallback_settings["market_cap_rows"], 0)

    # ------------------------------------------------------ end to end
    def test_refit_and_propose_uses_caps_end_to_end(self):
        """No mocked model state: a real walk-forward refit, all states risk-on."""
        _, benchmark, volume, _, _ = MARKET
        options = dict(STRATEGY_OPTIONS)
        options.update(top_n=2, risk_on_states=(0, 1, 2), min_order_notional=0.0)
        model = HMMVolumeStrategy(
            benchmark, volume, market_cap=self.q1_snapshot, **options
        )
        signal = pd.Timestamp(model.bars["end"].iloc[50])
        context = decision(
            signal, membership=self.q1_members, prices=self.prices,
            volume=self.stock_volume,
        )
        model.refit(None, context)
        self.assertIn(model.state, (0, 1, 2))
        budgets = model.propose_buys(context)
        self.assertEqual(set(budgets), {"S5", "S4"})
        self.assert_cap_proportional(
            budgets, {"S5": 1000.0, "S4": 500.0}, 1000.0
        )
        self.assertEqual(model.refit_settings()["weighting"], "market_cap")


if __name__ == "__main__":
    unittest.main()
