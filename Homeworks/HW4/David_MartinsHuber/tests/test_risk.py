"""Independent expected budgets, state transitions, and engine integration."""

from copy import deepcopy
import math
from statistics import stdev
import unittest

import numpy as np
import pandas as pd

from backtesting import (Backtester, ShadowPortfolioBacktester, Strategy, RiskManager,
                         RiskPolicy, RiskContext, RiskAdjustment, Portfolio, OMS, Phase,
                         SingleNameCap, DrawdownGate, VolatilityTarget,
                         MinimumVarianceOptimizer, EqualRiskContributionOptimizer,
                         PortfolioStrategy, PortfolioOptimizer, DecisionContext,
                         EqualWeightOptimizer)
from backtesting.contracts import plain_amount
from tests.test_backtesting import inputs


def context(navs=(100.,), holdings=None, fee=0.0, available=None):
    history = pd.DataFrame({"nav": navs}, index=pd.bdate_range("2020-01-01", periods=len(navs)))
    holdings = dict(holdings or {})
    cash = navs[-1] - sum(holdings.values())
    return RiskContext(history.index[-1], history.index[-1] + pd.Timedelta(days=1),
                       Portfolio(cash, cash if available is None else available, navs[-1], holdings),
                       history, fee, "test", 2)


class DailyBudget(Strategy):
    def propose_buys(self, ctx):
        return {"A": min(100., ctx.portfolio.available_cash)}


class RelabelHoldings(Strategy):
    """Value-preserving relabel; cash + holdings still reconciles with NAV."""

    def __init__(self):
        super().__init__("relabel")

    def propose_buys(self, ctx):
        if "A" in ctx.portfolio.holdings:
            ctx.portfolio.holdings["B"] = ctx.portfolio.holdings.pop("A")
        return {"A": min(.25 * ctx.portfolio.nav, ctx.portfolio.available_cash)}


class RefitWindowTamper(Strategy):
    """Tamper in needs_refit, which is not followed by a context rebuild."""

    def __init__(self):
        super().__init__("refit_tamper")
        self.day = 0

    def needs_refit(self, ctx):
        if self.day and "A" in ctx.portfolio.holdings:
            ctx.portfolio.holdings["B"] = ctx.portfolio.holdings.pop("A")
        return False

    def propose_buys(self, ctx):
        self.day += 1
        return {"A": min(.25 * ctx.portfolio.nav, ctx.portfolio.available_cash)}


class InjectMembership(Strategy):
    def __init__(self):
        super().__init__("inject_membership")

    def propose_buys(self, ctx):
        ctx.membership.loc[len(ctx.membership)] = {
            "symbol": "Z", "quarter_start": pd.Timestamp("2020-01-01"),
            "selection_date": pd.Timestamp("2099-01-01")}
        return {"Z": 100.}


class RiskChecks(unittest.TestCase):
    def test_fee_inclusive_room_uses_all_existing_exposure(self):
        decision = RiskManager([SingleNameCap(.25)]).review(context(holdings={"A": 20}, fee=.01), {"A": 70})
        self.assertAlmostEqual(decision.approved_budgets["A"], 5.05)
        row = decision.audit.loc[decision.audit.policy.eq("SingleNameCap")].iloc[0]
        self.assertAlmostEqual(row.metrics["current_weight"], .20)
        self.assertAlmostEqual(row.metrics["projected_signal_weight"], .25)
        self.assertEqual(row.strategy_id, "test")
        self.assertEqual(row.model_version, 2)

    def test_multiple_names_are_capped_without_redistribution(self):
        decision = RiskManager([SingleNameCap(.25)]).review(context(holdings={"A": 20}), {"A": 50, "B": 50})
        self.assertEqual(decision.approved_budgets, {"A": 5., "B": 25.})

    def test_existing_breach_is_logged_even_without_a_proposal(self):
        decision = RiskManager([SingleNameCap(.25)]).review(context(holdings={"A": 40}), {})
        self.assertEqual(decision.approved_budgets, {})
        row = decision.audit.loc[decision.audit.symbol.eq("A")].iloc[0]
        self.assertEqual(row.reason, "existing_exposure_breach_no_early_sale")
        self.assertEqual(row.metrics["current_weight"], .4)

    def test_drawdown_hysteresis_and_permanent_latch(self):
        manager = RiskManager([DrawdownGate(.15, resume_drawdown=.05)])
        self.assertEqual(manager.review(context([100, 84]), {"A": 40}).approved_budgets, {})
        self.assertEqual(manager.review(context([100, 84, 92]), {"A": 40}).approved_budgets, {})
        self.assertEqual(manager.review(context([100, 84, 92, 96]), {"A": 40}).approved_budgets, {"A": 40})
        latched = RiskManager([DrawdownGate(.15)])
        latched.review(context([100, 84]), {"A": 40})
        self.assertEqual(latched.review(context([100, 84, 110]), {"A": 40}).approved_budgets, {})

    def test_realized_volatility_matches_independent_scalar_estimate(self):
        returns = [.1, -.1, .1, -.1]
        navs = [100.]
        for value in returns:
            navs.append(navs[-1] * (1 + value))
        expected_scale = .2 / (stdev(returns) * 252 ** .5)
        decision = RiskManager([VolatilityTarget(.2, lookback=4, min_observations=4)]).review(context(navs), {"A": 40})
        self.assertAlmostEqual(decision.approved_budgets["A"], 40 * expected_scale)
        row = decision.audit.loc[decision.audit.policy.eq("VolatilityTarget")].iloc[0]
        self.assertAlmostEqual(row.metrics["realized_volatility"], stdev(returns) * 252 ** .5)

    def test_volatility_warmup_and_zero_volatility_never_lever_up(self):
        for warmup, expected in [("allow", {"A": 40}), ("block", {})]:
            manager = RiskManager([VolatilityTarget(lookback=3, min_observations=3, warmup=warmup)])
            self.assertEqual(manager.review(context([100, 100]), {"A": 40}).approved_budgets, expected)
        manager = RiskManager([VolatilityTarget(lookback=3, min_observations=3)])
        self.assertEqual(manager.review(context([100] * 4), {"A": 40}).approved_budgets, {"A": 40})

    def test_cash_limit_and_large_finite_budget_scaling(self):
        result = RiskManager().review(context(available=10), {"A": 40, "B": 60})
        self.assertEqual(result.approved_budgets, {"A": 4, "B": 6})
        result = RiskManager().review(context([1e300], available=1e-300), {"A": 1e300})
        self.assertEqual(result.approved_budgets, {"A": 1e-300})

    def test_policy_order_is_fixed_and_no_policy_can_increase_a_budget(self):
        class Bad(RiskPolicy):
            priority = 40
            def apply(self, ctx, budgets):
                return RiskAdjustment({"A": budgets["A"] + 1}, "bad", {})
        manager = RiskManager([Bad(), DrawdownGate(.15)])
        before = deepcopy(manager.policies)
        with self.assertRaisesRegex(ValueError, "increase"):
            manager.review(context([100, 80]), {"A": 40})
        self.assertIsNone(manager._cutoff)
        self.assertTrue(manager.audit.empty)
        self.assertEqual([p.active for p in manager.policies if isinstance(p, DrawdownGate)],
                         [p.active for p in before if isinstance(p, DrawdownGate)])

    def test_mutating_policy_inputs_cannot_rewrite_source_history(self):
        class Mutator(RiskPolicy):
            def apply(self, ctx, budgets):
                ctx.history.iloc[0, 0] = 999
                ctx.portfolio.holdings["A"] = 999
                return RiskAdjustment(budgets, "test", {})
        ctx = context([100, 100])
        RiskManager([Mutator()]).review(ctx, {"A": 20})
        self.assertEqual(ctx.history.nav.iloc[0], 100)
        self.assertEqual(ctx.portfolio.holdings, {})

    def test_future_history_and_rewritten_history_are_rejected(self):
        manager = RiskManager()
        first = context([100])
        manager.review(first, {})
        bad = context([99, 100])
        with self.assertRaisesRegex(ValueError, "rewritten"):
            manager.review(bad, {})
        future = RiskContext(first.signal_date, first.execution_date, first.portfolio,
                             context([100, 100]).history, 0)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            RiskManager().review(future, {})

    def test_engine_drawdown_gate_blocks_next_entries_without_early_sales(self):
        dates, p, b, m = inputs(6)
        p.loc[p.date.ge(dates[2]), "adj_close"] = 1
        result = Backtester(p, b, m, DailyBudget("daily"), initial_cash=1000, fee_bps=0,
                            risk_manager=RiskManager([DrawdownGate(.05)])).run(dates[1], dates[-1])
        self.assertEqual(result.fills.date.tolist(), [dates[1], dates[2]])
        self.assertTrue(result.fills.side.eq("BUY").all())
        self.assertEqual(len(result.open_lots), 2)
        gates = result.risk.loc[result.risk.policy.eq("DrawdownGate")]
        self.assertEqual(gates.loc[gates.reason.eq("drawdown_restriction")].execution_date.min(), dates[3])
        self.assertTrue(result.history.cash.iloc[3:].eq(800).all())

    def test_portfolio_caps_merged_budgets_and_attributes_risk_approval(self):
        dates, p, b, m = inputs(6)
        source_manager = RiskManager([SingleNameCap(.05)])
        result = ShadowPortfolioBacktester(p, b, m, [DailyBudget("a"), DailyBudget("b")],
                    initial_cash=1000, shadow_initial_cash=1000, fee_bps=0,
                    risk_manager=source_manager).run(dates[1], dates[-1])
        self.assertEqual(result.portfolio.fills.notional.tolist(), [50.])
        self.assertEqual(result.contributions.iloc[:2].budget.tolist(), [50., 50.])
        self.assertEqual(result.approved_contributions.iloc[:2].approved_budget.tolist(), [25., 25.])
        self.assertTrue(source_manager.audit.empty)  # Engine owns an independent copy.
        self.assertTrue(all(shadow.risk.empty for shadow in result.shadows.values()))
        self.assertEqual(set(result.portfolio.risk.strategy_id), {"portfolio"})

    def test_future_price_changes_do_not_change_prior_risk_decisions(self):
        dates, p, b, m = inputs(10)
        def run(prices):
            return Backtester(prices, b, m, DailyBudget("daily"), initial_cash=1000,
                    risk_manager=RiskManager([DrawdownGate(.05), SingleNameCap(.25),
                        VolatilityTarget(.1, lookback=3, min_observations=2)])).run(dates[1], dates[-1])
        cutoff = dates[5]
        first = run(p)
        changed = p.copy()
        changed.loc[changed.date.gt(cutoff), "adj_close"] *= 7
        second = run(changed)
        a, z = first.risk, second.risk
        pd.testing.assert_frame_equal(a.loc[a.signal_date.le(cutoff)].reset_index(drop=True),
                                      z.loc[z.signal_date.le(cutoff)].reset_index(drop=True))
        pd.testing.assert_frame_equal(first.history.loc[:cutoff], second.history.loc[:cutoff])

    def test_invalid_configuration_and_generator_policies(self):
        for cap in [0, -1, np.nan, 1.01]:
            with self.assertRaises(ValueError):
                SingleNameCap(cap)
        with self.assertRaises(ValueError):
            DrawdownGate(.1, resume_drawdown=.2)
        with self.assertRaises(ValueError):
            VolatilityTarget(warmup="invent")
        manager = RiskManager(p for p in [SingleNameCap(.25)])
        self.assertEqual(manager.review(context(), {"A": 50}).approved_budgets, {"A": 25})

    def test_drawdown_gate_exact_threshold_activates(self):
        for threshold in (.05, .10, .15, .20, .33):
            manager = RiskManager([DrawdownGate(threshold, scale=0.)])
            nav = 100. * (1 - threshold)
            decision = manager.review(context([100., nav]), {"A": 40.})
            self.assertEqual(decision.approved_budgets, {}, f"threshold {threshold}")
            self.assertTrue(manager.policies[0].active)

    def test_drawdown_gate_exact_recovery_releases(self):
        manager = RiskManager([DrawdownGate(.10, scale=0., resume_drawdown=.05)])
        self.assertEqual(manager.review(context([100., 90.]), {"A": 40.}).approved_budgets, {})
        self.assertTrue(manager.policies[0].active)
        self.assertEqual(manager.review(context([100., 90., 95.]), {"A": 40.}).approved_budgets,
                         {"A": 40.})
        self.assertFalse(manager.policies[0].active)

    def test_engine_risk_review_ignores_relabelled_holdings(self):
        dates, p, b, m = inputs(8)
        result = Backtester(p, b, m, RelabelHoldings(), initial_cash=1000, fee_bps=0,
                            risk_manager=RiskManager([SingleNameCap(.25)])).run(dates[1], dates[-1])
        self.assertAlmostEqual((result.history.holdings_value / result.history.nav).max(), .25)
        rows = result.risk.loc[result.risk.policy.eq("SingleNameCap")
                               & result.risk.symbol.eq("A")]
        self.assertAlmostEqual(rows.metrics.iloc[-1]["current_weight"], .25)

    def test_needs_refit_mutation_cannot_hide_exposure(self):
        dates, p, b, m = inputs(8)
        result = Backtester(p, b, m, RefitWindowTamper(), initial_cash=1000, fee_bps=0,
                            risk_manager=RiskManager([SingleNameCap(.25)])).run(dates[1], dates[-1])
        self.assertAlmostEqual((result.history.holdings_value / result.history.nav).max(), .25)

    def test_engine_risk_review_ignores_inflated_snapshot(self):
        class Inflate(Strategy):
            def __init__(self):
                super().__init__("inflate")

            def propose_buys(self, ctx):
                object.__setattr__(ctx.portfolio, "nav", 1e9)
                ctx.portfolio.holdings.clear()
                return {"A": 1e9}

        dates, p, b, m = inputs(8)
        result = Backtester(p, b, m, Inflate(), initial_cash=1000, fee_bps=0,
                            risk_manager=RiskManager([SingleNameCap(.25)])).run(dates[1], dates[-1])
        self.assertAlmostEqual((result.history.holdings_value / result.history.nav).max(), .25)

    def test_membership_injection_is_rejected(self):
        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, InjectMembership(), initial_cash=1000)
        with self.assertRaisesRegex(ValueError, "universe"):
            engine.run(dates[1], dates[-1])

    def test_history_replay_allows_equivalent_dtypes(self):
        manager = RiskManager()
        manager.review(context([100, 80]), {})
        self.assertEqual(manager.review(context([100., 80., 90.]), {}).approved_budgets, {})


EPS = np.finfo(float).eps


class Sym(str):
    """Content equals a ticker, but distinct instances do not compare equal."""

    __slots__ = ()

    def __eq__(self, other):
        if other is self:
            return True
        if isinstance(other, Sym):
            return False
        return str.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return str.__hash__(self)


class TwinName(Strategy):
    """Two distinct dict keys that both look like ticker A."""

    def __init__(self):
        super().__init__("twin")
        self.keys = (Sym("A"), Sym("A"))

    def propose_buys(self, ctx):
        half = ctx.portfolio.available_cash / 2
        return {self.keys[0]: half, self.keys[1]: half}


class ToleranceStrength(unittest.TestCase):
    def test_drawdown_gate_does_not_activate_below_threshold(self):
        for delta in (1e-4, 1e-6, 1e-9, 1e-12):
            with self.subTest(delta=delta):
                threshold = 0.15
                nav = 100.0 * (1 - (threshold - delta))
                drawdown = 1 - nav / 100.0
                self.assertLess(drawdown, threshold - 8 * EPS)
                manager = RiskManager([DrawdownGate(threshold, scale=0.0)])
                decision = manager.review(context([100.0, nav]), {"A": 40.0})
                self.assertEqual(decision.approved_budgets, {"A": 40.0})
                self.assertFalse(manager.policies[0].active)

    def test_drawdown_gate_does_not_release_inside_recovery_buffer(self):
        for delta in (1e-4, 1e-6, 1e-9, 1e-12):
            with self.subTest(delta=delta):
                resume = 0.05
                nav = 100.0 * (1 - (resume + delta))
                drawdown = 1 - nav / 100.0
                self.assertGreater(drawdown, resume + 8 * EPS)
                manager = RiskManager([DrawdownGate(0.10, scale=0.0,
                                                    resume_drawdown=resume)])
                self.assertEqual(manager.review(context([100.0, 90.0]), {"A": 40.0})
                                 .approved_budgets, {})
                self.assertTrue(manager.policies[0].active)
                decision = manager.review(context([100.0, 90.0, nav]), {"A": 40.0})
                self.assertEqual(decision.approved_budgets, {})
                self.assertTrue(manager.policies[0].active)


class SnapshotReconciliation(unittest.TestCase):
    def test_risk_review_rejects_portfolio_nav_that_disagrees_with_history(self):
        history = pd.DataFrame({"nav": [100.0, 99.0]},
                               index=pd.bdate_range("2020-01-01", periods=2))
        bad = RiskContext(history.index[-1], history.index[-1] + pd.Timedelta(days=1),
                          Portfolio(100.0, 100.0, 100.0, {}), history, 0.0)
        with self.assertRaisesRegex(ValueError, "reconcile"):
            RiskManager().review(bad, {"A": 10.0})


class CashCapRounding(unittest.TestCase):
    def test_cash_cap_never_rounds_above_available_cash(self):
        decision = RiskManager().review(context(available=8.1),
                                        {s: 3.0 for s in "ABCDE"})
        self.assertLessEqual(sum(decision.approved_budgets.values()), 8.1)


class BreachAuditFidelity(unittest.TestCase):
    def test_existing_breach_reason_survives_a_new_buy(self):
        decision = RiskManager([SingleNameCap(.25)]).review(
            context(holdings={"A": 40.0}), {"A": 10.0})
        self.assertEqual(decision.approved_budgets, {})
        row = decision.audit.loc[decision.audit.policy.eq("SingleNameCap")
                                 & decision.audit.symbol.eq("A")].iloc[0]
        self.assertEqual(row.reason, "existing_exposure_breach_no_early_sale")
        self.assertEqual(row.metrics["current_weight"], 0.4)

    def test_over_cap_holding_with_buy_does_not_raise(self):
        decision = RiskManager([SingleNameCap(.25)]).review(
            context(holdings={"A": 90.0}), {"A": 5.0})
        self.assertEqual(decision.approved_budgets, {})


class SymbolIdentity(unittest.TestCase):
    def test_str_subclass_keys_are_rejected_by_the_engine(self):
        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, TwinName(), initial_cash=1000)
        with self.assertRaisesRegex(ValueError, "valid symbols"):
            engine.run(dates[1], dates[-1])

    def test_str_subclass_keys_are_rejected_by_risk_manager(self):
        with self.assertRaisesRegex(ValueError, "symbols"):
            RiskManager([SingleNameCap(.25)]).review(
                context(holdings={"A": 40.0}), {Sym("A"): 10.0})

    def test_str_subclass_keys_are_rejected_by_oms(self):
        with self.assertRaises(ValueError):
            OMS(100, 0).submit_buy("s", Sym("A"), 10, "2020-01-03", "2020-01-06")

    def test_nonmember_ticker_is_rejected(self):
        class WrongTicker(Strategy):
            def __init__(self):
                super().__init__("wrong_ticker")

            def propose_buys(self, ctx):
                return {"ZZZ": 100.0}

        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, WrongTicker(), initial_cash=1000)
        with self.assertRaisesRegex(ValueError, "universe"):
            engine.run(dates[1], dates[-1])

    def test_shadow_coordinator_marks_failed_on_symbol_tamper(self):
        dates, p, b, m = inputs(5)
        coordinator = ShadowPortfolioBacktester(
            p, b, m, [TwinName()], initial_cash=1000, shadow_initial_cash=1000)
        with self.assertRaisesRegex(ValueError, "valid symbols"):
            coordinator.run(dates[1], dates[-1])
        self.assertEqual(coordinator.phase, Phase.FAILED)

    def test_numpy_string_symbols_are_accepted(self):
        class NumpySymbol(Strategy):
            def __init__(self):
                super().__init__("numpy_symbol")

            def propose_buys(self, ctx):
                return {np.str_("A"): 100.0}

        dates, p, b, m = inputs(5)
        result = Backtester(p, b, m, NumpySymbol(), initial_cash=1000, fee_bps=0
                            ).run(dates[1], dates[-1])
        self.assertFalse(result.fills.empty)
        self.assertEqual(set(result.fills.symbol), {"A"})

    def test_membership_planning_copy_cannot_change_engine(self):
        class MutateCopy(Strategy):
            def __init__(self):
                super().__init__("mutate_copy")
                self.day = 0

            def propose_buys(self, ctx):
                self.day += 1
                if self.day == 1:
                    ctx.membership.at[ctx.membership.index[0], "symbol"] = "Z"
                    return {}
                return {"Z": 10.0}

        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, MutateCopy(), initial_cash=1000)
        with self.assertRaisesRegex(ValueError, "universe"):
            engine.run(dates[1], dates[-1])

    def test_non_string_membership_is_rejected(self):
        class StrLie:
            def __str__(self):
                return "Z"

        dates, p, b, m = inputs(5)
        bad = m.copy()
        bad["symbol"] = [StrLie(), StrLie()]
        with self.assertRaisesRegex(ValueError, "Membership symbols"):
            Backtester(p, b, bad, DailyBudget("x"))

    def test_base_exception_leaves_engine_failed(self):
        class Interrupt(Strategy):
            def __init__(self):
                super().__init__("interrupt")

            def propose_buys(self, ctx):
                raise KeyboardInterrupt

        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, Interrupt(), initial_cash=1000)
        with self.assertRaises(KeyboardInterrupt):
            engine.run(dates[1], dates[-1])
        self.assertEqual(engine.phase, Phase.FAILED)


class DivLie(float):
    def __truediv__(self, other):
        return 0.0


class MulLie(float):
    def __mul__(self, other):
        return float(self)


class LyingHoldings(dict):
    def get(self, key, default=None):
        return default


class BudgetTypeHardening(unittest.TestCase):
    def test_float_subclass_budget_cannot_bypass_single_name_cap(self):
        decision = RiskManager([SingleNameCap(.25)]).review(
            context(holdings={"A": 20.0}, fee=.01), {"A": DivLie(70.0)})
        self.assertAlmostEqual(decision.approved_budgets["A"], 5.05)

    def test_float_subclass_budget_cannot_bypass_engine_cap(self):
        class DivBudget(Strategy):
            def __init__(self):
                super().__init__("div_budget")

            def propose_buys(self, ctx):
                return {"A": DivLie(999_000.0)}

        dates, p, b, m = inputs(8)
        result = Backtester(p, b, m, DivBudget(), initial_cash=1000, fee_bps=0,
                            risk_manager=RiskManager([SingleNameCap(.25)])
                            ).run(dates[1], dates[-1])
        self.assertAlmostEqual((result.history.holdings_value / result.history.nav).max(), .25)

    def test_float_subclass_budget_cannot_bypass_drawdown_gate(self):
        manager = RiskManager([DrawdownGate(.10, scale=0.0)])
        self.assertEqual(manager.review(context([100.0, 80.0]), {"A": 40.0})
                         .approved_budgets, {})
        self.assertEqual(manager.review(context([100.0, 80.0, 80.0]), {"A": MulLie(40.0)})
                         .approved_budgets, {})

    def test_lying_holdings_mapping_is_rejected(self):
        history = pd.DataFrame({"nav": [100.0]}, index=pd.bdate_range("2020-01-01", periods=1))
        ctx = RiskContext(history.index[-1], history.index[-1] + pd.Timedelta(days=1),
                          Portfolio(80.0, 80.0, 100.0, LyingHoldings({"A": 20.0})), history, 0.0)
        with self.assertRaisesRegex(ValueError, "reconcile"):
            RiskManager([SingleNameCap(.25)]).review(ctx, {"A": 30.0})


class CoordinatorPhase(unittest.TestCase):
    def test_attribution_error_leaves_coordinator_failed(self):
        import backtesting.portfolio as bp

        original = bp.PortfolioStrategy

        class Poisoned(original):
            def propose_buys(self, ctx):
                orders = super().propose_buys(ctx)
                self.contributions.append(dict(
                    signal_date=ctx.signal_date, execution_date=["unhashable"],
                    component_id="poison", symbol="A", budget=1.0, model_version=0))
                return orders

        dates, p, b, m = inputs(6)
        bp.PortfolioStrategy = Poisoned
        try:
            coordinator = bp.ShadowPortfolioBacktester(
                p, b, m, [DailyBudget("daily")], initial_cash=1000,
                shadow_initial_cash=1000, fee_bps=0)
            with self.assertRaises(TypeError):
                coordinator.run(dates[1], dates[-1])
        finally:
            bp.PortfolioStrategy = original
        self.assertEqual(coordinator.phase, Phase.FAILED)


class CapLie(float):
    def __mul__(self, other):
        return 1e18

    __rmul__ = __mul__


class GateLie(float):
    def __sub__(self, other):
        return 1e9


class VolLie(float):
    def __truediv__(self, other):
        return 1.0


class DomainHardening(unittest.TestCase):
    def test_plain_amount_domain(self):
        for bad in (True, False, float("nan"), float("inf"), -1.0, "1", None, 10 ** 400):
            with self.assertRaises(ValueError):
                plain_amount(bad)
        for good in (0, 1, 2.5, np.float64(3.5), np.int64(4)):
            self.assertIs(type(plain_amount(good)), float)

    def test_engine_canonicalizes_budget_types(self):
        class DivBudget(Strategy):
            def __init__(self):
                super().__init__("div_budget")
                self.sent = False

            def propose_buys(self, ctx):
                if self.sent:
                    return {}
                self.sent = True
                return {"A": DivLie(999_000.0)}

        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, DivBudget(), initial_cash=1_000_000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        engine.prepare()
        self.assertIs(type(engine.raw_proposals["A"]), float)
        self.assertIs(type(engine.pending_proposals["A"]), float)
        engine.advance()
        while engine.next_session is not None:
            engine.prepare()
            engine.advance()
        engine.finish()

    def test_oms_rejects_nonfinite_budget(self):
        for bad in (float("nan"), float("inf"), -1.0, True, "1"):
            oms = OMS(100.0, 0)
            with self.assertRaises(ValueError):
                oms.submit_buy("s", "A", bad, "2020-01-03", "2020-01-06")
            self.assertEqual(oms.available_cash, 100.0)

    def test_risk_history_rejects_nonfinite_and_non_numeric(self):
        for values in ([100.0, float("inf")], [100.0, float("nan")], [100.0, "100"]):
            history = pd.DataFrame({"nav": values},
                                   index=pd.bdate_range("2020-01-01", periods=2))
            ctx = RiskContext(history.index[-1], history.index[-1] + pd.Timedelta(days=1),
                              Portfolio(100.0, 100.0, 100.0, {}), history, 0.0)
            with self.assertRaises(ValueError):
                RiskManager().review(ctx, {"A": 10.0})

    def test_many_symbols_converge_within_cash_cap(self):
        decision = RiskManager().review(context(available=7.0),
                                        {f"S{i:03d}": 1.0 for i in range(100)})
        self.assertLessEqual(sum(decision.approved_budgets.values()), 7.0)
        self.assertGreater(sum(decision.approved_budgets.values()), 0.0)

    def test_policy_parameters_are_canonicalized(self):
        cap = SingleNameCap(CapLie(0.25))
        self.assertIs(type(cap.cap), float)
        decision = RiskManager([cap]).review(context(holdings={"A": 20.0}), {"A": 100.0})
        self.assertAlmostEqual(decision.approved_budgets["A"], 5.0)

        gate = DrawdownGate(GateLie(0.15), scale=0.0)
        self.assertIs(type(gate.threshold), float)
        self.assertEqual(RiskManager([gate]).review(context([100.0, 50.0]), {"A": 40.0})
                         .approved_budgets, {})

        vol = VolatilityTarget(VolLie(0.1), lookback=3, min_observations=3)
        self.assertIs(type(vol.annual_target), float)

    def test_coordinator_base_exception_leaves_failed(self):
        class Boom(Strategy):
            def __init__(self):
                super().__init__("boom")

            def propose_buys(self, ctx):
                raise KeyboardInterrupt

        dates, p, b, m = inputs(5)
        coordinator = ShadowPortfolioBacktester(
            p, b, m, [Boom()], initial_cash=1000, shadow_initial_cash=1000)
        with self.assertRaises(KeyboardInterrupt):
            coordinator.run(dates[1], dates[-1])
        self.assertEqual(coordinator.phase, Phase.FAILED)

    def test_finish_failure_after_last_session_leaves_failed(self):
        dates, p, b, m = inputs(4)
        engine = Backtester(p, b, m, DailyBudget("daily"), initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        while engine.next_session is not None:
            engine.prepare()
            engine.advance()
        engine.oms.order_records = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            engine.finish()
        self.assertEqual(engine.phase, Phase.FAILED)

    def test_advance_is_transactional_on_partial_fill_failure(self):
        class Two(Strategy):
            def __init__(self):
                super().__init__("two")

            def propose_buys(self, ctx):
                return {"A": 100.0, "B": 100.0}

        dates, p, b, m = inputs(5)
        prices = pd.concat([p, p.assign(symbol="B")], ignore_index=True)
        membership = pd.concat([m, m.assign(symbol="B")], ignore_index=True)
        engine = Backtester(prices, b, membership, Two(), initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        engine.prepare()
        original = engine.oms.fill
        calls = {"n": 0}

        def failing_fill(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected fill failure")
            return original(*args, **kwargs)

        engine.oms.fill = failing_fill
        with self.assertRaises(RuntimeError):
            engine.advance()
        self.assertEqual(engine.phase, Phase.FAILED)
        self.assertEqual(engine.oms.cash, 1000.0)
        self.assertFalse(engine.oms.fills)
        self.assertFalse(engine.oms.lots)

    def test_plain_amount_normalises_negative_zero(self):
        self.assertEqual(math.copysign(1.0, plain_amount(-0.0)), 1.0)
        self.assertEqual(str(plain_amount(-0.0)), "0.0")

    def test_volatility_windows_are_canonical_python_ints(self):
        vol = VolatilityTarget(0.1, lookback=np.int64(63),
                               min_observations=np.int64(20),
                               sessions_per_year=np.int64(252))
        self.assertIs(type(vol.lookback), int)
        self.assertIs(type(vol.min_observations), int)
        self.assertIs(type(vol.sessions_per_year), int)

        class LieInt(int):
            def __le__(self, other):
                return True

            def __ge__(self, other):
                return True

        with self.assertRaises(ValueError):
            VolatilityTarget(0.1, lookback=LieInt(0), min_observations=LieInt(20))

    def test_drawdown_gate_scale_and_resume_are_canonicalized(self):
        gate = DrawdownGate(GateLie(0.15), scale=GateLie(0.5),
                            resume_drawdown=GateLie(0.05))
        self.assertIs(type(gate.threshold), float)
        self.assertIs(type(gate.scale), float)
        self.assertIs(type(gate.resume_drawdown), float)

    def test_risk_history_is_canonicalized_to_float(self):
        manager = RiskManager()
        manager.review(context([100, 80]), {})
        self.assertEqual(manager._history.nav.dtype, np.dtype(float))

    def test_oms_initial_config_is_canonicalized(self):
        class Negative(float):
            def __float__(self):
                return -1000.0

        with self.assertRaises(ValueError):
            OMS(initial_cash=Negative(1e9), fee_bps=5)

        oms = OMS(1000.0, fee_bps=DivLie(5.0))
        self.assertIs(type(oms.fee_rate), float)
        self.assertAlmostEqual(oms.fee_rate, 0.0005)

    def test_optimizer_settings_are_canonicalized(self):
        class Lie50(float):
            def __le__(self, other):
                return True

            def __ge__(self, other):
                return True

        with self.assertRaises(ValueError):
            MinimumVarianceOptimizer(shrinkage=Lie50(50), max_weight=0.5)
        optimizer = MinimumVarianceOptimizer(shrinkage=Lie50(0.1),
                                             max_weight=Lie50(0.5))
        self.assertIs(type(optimizer.shrinkage), float)
        self.assertIs(type(optimizer.max_weight), float)

    def test_prepare_rolls_back_orders_when_submission_fails(self):
        class Two(Strategy):
            def __init__(self):
                super().__init__("two_submit")

            def propose_buys(self, ctx):
                return {"A": 100.0, "B": 100.0}

        dates, p, b, m = inputs(5)
        prices = pd.concat([p, p.assign(symbol="B")], ignore_index=True)
        membership = pd.concat([m, m.assign(symbol="B")], ignore_index=True)
        engine = Backtester(prices, b, membership, Two(), initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        original = engine.oms.submit_buy
        calls = {"n": 0}

        def failing_submit(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected submit failure")
            return original(*args, **kwargs)

        engine.oms.submit_buy = failing_submit
        with self.assertRaises(RuntimeError):
            engine.prepare()
        self.assertEqual(engine.phase, Phase.FAILED)
        self.assertFalse(engine.oms.orders)
        self.assertEqual(engine.oms.available_cash, 1000.0)

    def test_advance_rollback_restores_marks_status_and_diagnostics(self):
        class Two(Strategy):
            def __init__(self):
                super().__init__("two_rollback")

            def propose_buys(self, ctx):
                return {"A": 100.0, "B": 100.0}

        dates, p, b, m = inputs(5)
        prices = pd.concat([p, p.assign(symbol="B")], ignore_index=True)
        membership = pd.concat([m, m.assign(symbol="B")], ignore_index=True)
        engine = Backtester(prices, b, membership, Two(), initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        engine.prepare()

        def failing_fill(*args, **kwargs):
            engine._diagnostics.append({"date": dates[1], "symbol": "X", "event": "test"})
            engine._marks["X"] = 123.0
            engine._last_seen["X"] = 99
            raise RuntimeError("injected fill failure")

        engine.oms.fill = failing_fill
        with self.assertRaises(RuntimeError):
            engine.advance()
        self.assertEqual(engine.phase, Phase.FAILED)
        self.assertEqual(engine._diagnostics, [])
        self.assertEqual(engine._marks, {})
        self.assertEqual(engine._last_seen, {})
        self.assertEqual(set(o.status for o in engine.oms.orders.values()), {"pending"})

    def test_finish_publishes_complete_only_after_result_exists(self):
        dates, p, b, m = inputs(4)
        engine = Backtester(p, b, m, DailyBudget("daily"), initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        while engine.next_session is not None:
            engine.prepare()
            engine.advance()
        seen = {}
        original = engine.oms.order_records

        def record():
            seen["phase"] = engine.phase
            return original()

        engine.oms.order_records = record
        engine.finish()
        self.assertEqual(seen["phase"], Phase.EXECUTING)
        self.assertEqual(engine.phase, Phase.COMPLETE)

    def test_result_phases_include_complete(self):
        dates, p, b, m = inputs(4)
        result = Backtester(p, b, m, DailyBudget("daily"), initial_cash=1000,
                            fee_bps=0).run(dates[1], dates[-1])
        self.assertIn("complete", result.phases.phase.tolist())
        self.assertEqual(result.phases.phase.iloc[-1], "complete")

    def test_plain_amount_wraps_arbitrary_conversion_errors(self):
        class Boom(float):
            def __float__(self):
                raise RuntimeError("boom")

        with self.assertRaises(ValueError):
            plain_amount(Boom(1.0))

    def test_erc_settings_are_canonicalized(self):
        class Lie50(float):
            def __le__(self, other):
                return True

            def __ge__(self, other):
                return True

        with self.assertRaises(ValueError):
            EqualRiskContributionOptimizer(shrinkage=Lie50(50), max_weight=0.5)
        optimizer = EqualRiskContributionOptimizer(
            shrinkage=Lie50(0.1), max_weight=Lie50(0.5), maxiter=np.int64(100))
        self.assertIs(type(optimizer.shrinkage), float)
        self.assertIs(type(optimizer.max_weight), float)
        self.assertIs(type(optimizer.maxiter), int)
        with self.assertRaises(ValueError):
            EqualRiskContributionOptimizer(rc_tolerance=1e300)
        with self.assertRaises(ValueError):
            EqualRiskContributionOptimizer(ftol=1e9)
        for bad in (2.5, 1000.0, 1_000_001, np.int64(10 ** 9)):
            with self.assertRaises(ValueError):
                EqualRiskContributionOptimizer(maxiter=bad)

    def test_cash_limit_approved_budgets_are_plain_floats(self):
        decision = RiskManager().review(context(available=7.0),
                                        {f"S{i:03d}": 1.0 for i in range(100)})
        self.assertTrue(decision.approved_budgets)
        self.assertTrue(all(type(value) is float
                            for value in decision.approved_budgets.values()))

    def test_advance_rollback_restores_rejected_order_status(self):
        class Two(Strategy):
            def __init__(self):
                super().__init__("two_rejected")

            def propose_buys(self, ctx):
                return {"A": 100.0, "B": 100.0}

        dates, p, b, m = inputs(5)
        prices = pd.concat([p, p.assign(symbol="B")], ignore_index=True)
        prices = prices.loc[~((prices.date.eq(dates[1])) & prices.symbol.eq("A"))]
        membership = pd.concat([m, m.assign(symbol="B")], ignore_index=True)
        engine = Backtester(prices, b, membership, Two(), initial_cash=1000, fee_bps=0)
        engine.begin(dates[1], dates[-1])
        engine.prepare()

        def failing_fill(*args, **kwargs):
            raise RuntimeError("injected fill failure")

        engine.oms.fill = failing_fill
        with self.assertRaises(RuntimeError):
            engine.advance()
        self.assertEqual(engine.phase, Phase.FAILED)
        self.assertEqual(set(o.status for o in engine.oms.orders.values()), {"pending"})
        self.assertEqual(engine._diagnostics, [])

    def test_oms_rejects_dust_overspend(self):
        oms = OMS(1e-8, 0)
        order = oms.submit_buy("s", "A", 1e-8, "2020-01-03", "2020-01-06")
        oms.fill(order, symbol="A", quote_date="2020-01-06", price=1.0,
                 session="2020-01-06")
        self.assertEqual(oms.cash, 0.0)
        with self.assertRaises(ValueError):
            oms.submit_buy("s", "A", 1e-8, "2020-01-07", "2020-01-08")

    def test_optimizer_weights_are_canonicalized(self):
        class Lie(float):
            def __lt__(self, other):
                return False

            def __add__(self, other):
                return 1.0

        class Hostile(PortfolioOptimizer):
            def weights(self, ctx):
                return {"a": Lie(-5.0), "b": Lie(6.0)}

        strategy = PortfolioStrategy("portfolio", EqualWeightOptimizer())
        index = pd.bdate_range("2020-01-01", periods=2)
        navs = pd.DataFrame({"a": [100.0, 101.0], "b": [100.0, 99.0]}, index=index)
        strategy.observe(index[-1], navs,
                         {"a": {"A": 10.0}, "b": {"B": 10.0}},
                         {"a": 0, "b": 0})
        strategy.optimizer = Hostile()
        ctx = DecisionContext(index[-1], index[-1] + pd.Timedelta(days=1),
                              pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                              Portfolio(1000.0, 1000.0, 1000.0, {}), 0)
        with self.assertRaises(ValueError):
            strategy.propose_buys(ctx)

    def test_equal_weight_reservations_fill_without_borrowing(self):
        oms = OMS(1.0, 0)
        budget = 1.0 / 9
        ids = [oms.submit_buy("s", f"S{i}", budget, "2020-01-02", "2020-01-03")
               for i in range(9)]
        for order_id in ids:
            order = oms.orders[order_id]
            oms.fill(order_id, symbol=order.symbol, quote_date="2020-01-03",
                     price=1.0, session="2020-01-03")
        self.assertEqual(oms.cash, 0.0)

    def test_engine_skips_dust_residue(self):
        class Dust(Strategy):
            def __init__(self):
                super().__init__("dust")

            def propose_buys(self, ctx):
                return {"A": 5e-324}

        dates, p, b, m = inputs(5)
        engine = Backtester(p, b, m, Dust(), initial_cash=1000, fee_bps=0)
        result = engine.run(dates[1], dates[-1])
        self.assertEqual(engine.phase, Phase.COMPLETE)
        self.assertTrue(result.fills.empty)

    def test_history_fees_include_delisting_exit(self):
        class BuyA(Strategy):
            def __init__(self):
                super().__init__("buy_a")
                self.sent = False

            def propose_buys(self, ctx):
                if self.sent:
                    return {}
                self.sent = True
                return {"A": 100.0}

        dates, p, b, m = inputs(5)
        prices = p.loc[~((p.date.ge(dates[2])) & p.symbol.eq("A"))]
        result = Backtester(prices, b, m, BuyA(), initial_cash=1000, fee_bps=100,
                            max_stale_sessions=0, delisting="last_price"
                            ).run(dates[1], dates[-1])
        self.assertTrue(result.diagnostics.event.eq("delisting_exit").any())
        self.assertAlmostEqual(result.history.fees.sum(), result.fills.fee.sum())

    def test_terminal_phase_uses_signal_date(self):
        dates, p, b, m = inputs(4)
        result = Backtester(p, b, m, DailyBudget("daily"), initial_cash=1000,
                            fee_bps=0).run(dates[1], dates[-1])
        self.assertEqual(result.phases.signal_date.iloc[-1], dates[-2])

    def test_policy_audit_symbols_are_validated(self):
        class BadAudit(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "bad", {}, {42: "reason"})

        with self.assertRaises(ValueError):
            RiskManager([BadAudit()]).review(context(), {"A": 1.0})

    def test_policy_symbol_metrics_must_be_mappings(self):
        class ScalarMetrics(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      symbol_metrics={"A": 0.25})

        with self.assertRaises(ValueError):
            RiskManager([ScalarMetrics()]).review(context(), {"A": 1.0})

    def test_policy_audit_mapping_subclass_is_rejected(self):
        class Sneaky(dict):
            def __iter__(self):
                return iter(["A"])

            def __len__(self):
                return 1

        class SneakyPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      symbol_reasons=Sneaky({None: "r", 7: "r"}))

        with self.assertRaises(ValueError):
            RiskManager([SneakyPolicy()]).review(context(), {"A": 1.0})

    def test_large_equal_weight_books_fill_without_abort(self):
        for capital, count in ((1e6, 300), (1e6, 500), (1e8, 11),
                               (1e9, 7), (1e10, 3)):
            with self.subTest(capital=capital, count=count):
                dates = pd.bdate_range("2020-01-02", periods=2)
                symbols = [f"S{i}" for i in range(count)]
                prices = pd.DataFrame(
                    [(day, symbol, 1.0) for day in dates for symbol in symbols],
                    columns=["date", "symbol", "adj_close"])
                benchmark = pd.DataFrame({"date": dates, "adj_close": 100.0})
                membership = pd.DataFrame({
                    "symbol": symbols,
                    "quarter_start": pd.Timestamp("2020-01-01"),
                    "selection_date": pd.Timestamp("2019-12-31")})

                class AllIn(Strategy):
                    def __init__(self):
                        super().__init__("allin_large")

                    def propose_buys(self, ctx):
                        available = ctx.portfolio.available_cash
                        return {} if available <= 0 else {
                            symbol: available / count for symbol in symbols}

                result = Backtester(prices, benchmark, membership, AllIn(),
                                    initial_cash=capital, fee_bps=5
                                    ).run(dates[1], dates[-1])
                self.assertEqual(len(result.fills), count)
                self.assertEqual(result.history.cash.iloc[-1], 0.0)

    def test_reservation_never_exceeds_available_cash(self):
        oms = OMS(1e8, 0)
        budget = 1e8 + 5e-5
        order_id = oms.submit_buy("s", "A", budget, "2020-01-02", "2020-01-03")
        self.assertLessEqual(oms.orders[order_id].budget, oms.cash)
        self.assertGreaterEqual(oms.available_cash, 0.0)

    def test_policy_audit_rejects_non_dict_mapping(self):
        from collections import UserDict

        class NonDictPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      UserDict({"A": "reason"}))

        with self.assertRaises(ValueError):
            RiskManager([NonDictPolicy()]).review(context(), {"A": 1.0})

    def test_policy_symbol_metric_values_must_be_dicts(self):
        class ListMetric(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      symbol_metrics={"A": [("k", 1)]})

        with self.assertRaises(ValueError):
            RiskManager([ListMetric()]).review(context(), {"A": 1.0})

    def test_falsy_dict_audit_mapping_is_rejected(self):
        class Falsy(dict):
            def __bool__(self):
                return False

        class FalsyPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      symbol_reasons=Falsy({"A": "reason"}))

        with self.assertRaises(ValueError):
            RiskManager([FalsyPolicy()]).review(context(), {"A": 1.0})

    def test_audit_mapping_cannot_change_between_check_and_use(self):
        class Flip(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def __bool__(self):
                self.calls += 1
                return self.calls > 1

        class FlipPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      symbol_reasons=Flip({"A": "reason"}))

        with self.assertRaises(ValueError):
            RiskManager([FlipPolicy()]).review(context(), {"A": 1.0})

    def test_policy_top_level_reason_must_be_a_string(self):
        class BadReason(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), 42, {})

        with self.assertRaises(ValueError):
            RiskManager([BadReason()]).review(context(), {"A": 1.0})

    def test_audit_copy_failure_does_not_commit_state(self):
        class Bomb:
            def __init__(self):
                self.calls = 0

            def __deepcopy__(self, memo):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("copy bomb")
                return self

        class BombPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {"x": Bomb()})

        manager = RiskManager([BombPolicy()])
        with self.assertRaises(RuntimeError):
            manager.review(context(), {"A": 1.0})
        self.assertIsNone(manager._cutoff)
        self.assertTrue(manager.audit.empty)

    def test_audit_snapshot_survives_budget_conversion(self):
        class HookFloat(float):
            def __new__(cls, value, target):
                obj = super().__new__(cls, value)
                obj.target = target
                return obj

            def __float__(self):
                self.target["A"] = 42
                return float.__float__(self)

        class HookPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                self.reasons = {"A": "clean_string"}
                return RiskAdjustment({"A": HookFloat(1.0, self.reasons)},
                                      "ok", {}, self.reasons)

        decision = RiskManager([HookPolicy()]).review(context(), {"A": 1.0})
        row = decision.audit.loc[decision.audit.policy.eq("HookPolicy")
                                 & decision.audit.symbol.eq("A")].iloc[0]
        self.assertEqual(row.reason, "clean_string")

    def test_audit_snapshot_covers_metrics_details_and_reason(self):
        injected = object()

        class HookFloat(float):
            def __new__(cls, value, policy):
                obj = super().__new__(cls, value)
                obj.policy = policy
                return obj

            def __float__(self):
                object.__setattr__(self.policy.adjustment, "reason", injected)
                self.policy.metrics["tags"].append("tampered")
                self.policy.details["A"]["tags"].append("tampered")
                self.policy.reasons["A"] = 42
                return float.__float__(self)

        class HookPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                self.metrics = {"tags": ["clean"]}
                self.details = {"A": {"tags": ["clean"]}}
                self.reasons = {"A": "clean_string"}
                self.adjustment = RiskAdjustment(
                    {"A": HookFloat(1.0, self)}, "clean_reason",
                    self.metrics, self.reasons, self.details)
                return self.adjustment

        decision = RiskManager([HookPolicy()]).review(context(), {"A": 1.0})
        row = decision.audit.loc[decision.audit.policy.eq("HookPolicy")
                                 & decision.audit.symbol.eq("A")].iloc[0]
        self.assertEqual(row.reason, "clean_string")
        self.assertEqual(row.metrics["tags"], ["clean"])

    def test_audit_snapshots_are_taken_before_any_deepcopy(self):
        class Bomb:
            def __init__(self, target):
                self.target = target

            def __deepcopy__(self, memo):
                self.target["forged"] = 1
                return self

        class BombPolicy(RiskPolicy):
            def apply(self, ctx, budgets):
                self.metrics = {"clean": 1}
                return RiskAdjustment(
                    dict(budgets), "ok", self.metrics, {"A": "reason"},
                    {"A": {"bomb": Bomb(self.metrics)}})

        decision = RiskManager([BombPolicy()]).review(context(), {"A": 1.0})
        row = decision.audit.loc[decision.audit.policy.eq("BombPolicy")
                                 & decision.audit.symbol.eq("A")].iloc[0]
        self.assertNotIn("forged", row.metrics)
        self.assertEqual(row.metrics["clean"], 1)

    def test_tolerance_scales_with_opening_capital_not_remaining_cash(self):
        oms = OMS(1e8, 0)
        first = oms.submit_buy("s", "A", 1e8 - 1e-4, "2020-01-02", "2020-01-03")
        oms.fill(first, symbol="A", quote_date="2020-01-03", price=1.0,
                 session="2020-01-03")
        remaining = oms.available_cash
        second = oms.submit_buy("s", "B", remaining + 5e-5, "2020-01-06",
                                "2020-01-07")
        self.assertLessEqual(oms.orders[second].budget, oms.cash)

    def test_reservation_rounding_never_reports_negative_available_cash(self):
        cash, first = 939149162778.5106, 358007640672.50507
        oms = OMS(cash, 0)
        oms.submit_buy("s", "A", first, "2020-01-02", "2020-01-03")
        oms.submit_buy("s", "B", oms.available_cash, "2020-01-02", "2020-01-03")
        self.assertGreaterEqual(oms.available_cash, 0.0)

    def test_per_symbol_reason_value_must_be_a_string(self):
        class BadReasonValue(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {}, {"A": 42})

        with self.assertRaises(ValueError):
            RiskManager([BadReasonValue()]).review(context(), {"A": 1.0})

    def test_top_level_metrics_must_be_a_mapping(self):
        class BadMetrics(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", [("k", 1)])

        with self.assertRaises(ValueError):
            RiskManager([BadMetrics()]).review(context(), {"A": 1.0})

    def test_details_only_symbol_still_gets_an_audit_row(self):
        class DetailsOnly(RiskPolicy):
            def apply(self, ctx, budgets):
                return RiskAdjustment(dict(budgets), "ok", {},
                                      symbol_metrics={"Z": {"note": 1}})

        decision = RiskManager([DetailsOnly()]).review(context(), {"A": 1.0})
        self.assertIn("Z", set(decision.audit.symbol))
