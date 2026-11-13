"""FF5 regression, baseline, tilt, self-fitting, and delisting tests."""

import unittest

import numpy as np
import pandas as pd

from backtesting import (Backtester, DecisionContext, FamaFrench5, FactorResult,
                         HeldPriceGapError, MarketData, OMS, Phase, Portfolio, Strategy)
from strategies.factor import CapWeightedMarketStrategy, FactorTiltStrategy
from tests.test_backtesting import BuyOnce, inputs


def context(symbols, signal="2020-01-31", execution="2020-02-03",
            nav=100.0, cash=None, holdings=None):
    dates = pd.bdate_range("2019-12-02", "2020-01-31")
    prices = pd.DataFrame(1.0, index=dates, columns=list(symbols))
    membership = pd.DataFrame({"symbol": list(symbols)})
    cash = nav if cash is None else cash
    return DecisionContext(
        pd.Timestamp(signal), pd.Timestamp(execution), prices, membership,
        pd.DataFrame(), Portfolio(cash, cash, nav, dict(holdings or {})), 0)


def factor_result(characteristics, factor_returns=None):
    return FactorResult(
        factor_returns if factor_returns is not None else pd.DataFrame(),
        characteristics, pd.DataFrame(), pd.DataFrame())


def characteristics(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "ME", "eligible"])


class SelfFittingStrategy(Strategy):
    self_fitting = True

    def __init__(self):
        super().__init__("self_fitting")
        self.fits = 0
        self.received_none = None

    def needs_refit(self, context):
        return self.fits == 0

    def refit(self, data, context):
        self.received_none = data is None
        self.fits += 1

    def propose_buys(self, context):
        return {}


class StrategyChecks(unittest.TestCase):
    def test_cap_weighted_market_uses_market_cap_weights(self):
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), "A", 1.0, True),
            (pd.Timestamp("2020-01-31"), "B", 2.0, True),
            (pd.Timestamp("2020-01-31"), "C", 7.0, True),
        ])
        strategy = CapWeightedMarketStrategy(factor_result(chars))
        budgets = strategy.propose_buys(context("ABC"))
        self.assertAlmostEqual(budgets["A"], 10.0, delta=1e-5)
        self.assertAlmostEqual(budgets["B"], 20.0, delta=1e-5)
        self.assertAlmostEqual(budgets["C"], 70.0, delta=1e-5)

    def test_tilt_excludes_bottom_quintile_and_reallocates_to_top(self):
        names = list("ABCDE")
        market_caps = dict(zip(names, [1.0, 2.0, 3.0, 4.0, 10.0]))
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), name, market_caps[name], True)
            for name in names])
        strategy = FactorTiltStrategy(
            factor_result(chars), ramp_months=0, top_q=0.2, bottom_q=0.2)
        strategy.scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3,
                                     "D": 0.4, "E": 0.5})
        budgets = strategy.propose_buys(context(names))
        self.assertNotIn("A", budgets)  # bottom quintile is never purchased
        # Cap weights: A .05, B .10, C .15, D .20, E .50. A's .05 moves to E.
        self.assertAlmostEqual(budgets["B"], 10.0, delta=1e-5)
        self.assertAlmostEqual(budgets["C"], 15.0, delta=1e-5)
        self.assertAlmostEqual(budgets["D"], 20.0, delta=1e-5)
        self.assertAlmostEqual(budgets["E"], 55.0, delta=1e-5)
        self.assertAlmostEqual(sum(budgets.values()), 100.0, delta=1e-5)

    def test_cap_weighted_market_rejects_invalid_min_order_notional(self):
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), "A", 1.0, True)])
        for value in (float("nan"), float("inf"), -5.0, "x", True):
            with self.assertRaisesRegex(ValueError, "min_order_notional"):
                CapWeightedMarketStrategy(
                    factor_result(chars), min_order_notional=value)

    def test_cap_weighted_market_fills_only_positive_gaps(self):
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), "A", 1.0, True),
            (pd.Timestamp("2020-01-31"), "B", 2.0, True),
            (pd.Timestamp("2020-01-31"), "C", 7.0, True),
        ])
        strategy = CapWeightedMarketStrategy(factor_result(chars))
        budgets = strategy.propose_buys(
            context("ABC", cash=10.0, holdings={"A": 50.0}))
        self.assertNotIn("A", budgets)
        self.assertAlmostEqual(sum(budgets.values()), 10.0, delta=1e-5)

    def test_mid_month_uses_the_last_fit_cross_section(self):
        old = pd.Timestamp("2020-01-31")
        mid = pd.Timestamp("2020-02-14")
        rows = [(old, name, 1.0, True) for name in "ABCDE"]
        rows += [(mid, name, 1.0, True) for name in "FGHIJ"]
        strategy = FactorTiltStrategy(
            factor_result(characteristics(rows)), ramp_months=0)
        strategy._fit_date = old
        strategy.scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3,
                                     "D": 0.4, "E": 0.5})
        budgets = strategy.propose_buys(context(
            list("ABCDEFGHIJ"), signal="2020-02-14", execution="2020-02-18",
            cash=50.0))
        self.assertTrue(budgets)
        self.assertTrue(set(budgets).issubset(set("BCDE")))
        self.assertNotIn("A", budgets)

    def test_tiny_residual_cash_does_not_create_a_dribble_order(self):
        names = list("ABCDE")
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), name, 1.0, True) for name in names])
        strategy = FactorTiltStrategy(factor_result(chars), ramp_months=0)
        strategy.scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3,
                                     "D": 0.4, "E": 0.5})
        self.assertEqual(strategy.propose_buys(context(names, cash=0.5)), {})

    def test_ramp_invests_one_third_of_nav_in_market(self):
        chars = characteristics([
            (pd.Timestamp(date), symbol, me, True)
            for date in ("2020-01-31", "2020-02-14", "2020-02-28")
            for symbol, me in (("A", 1.0), ("B", 2.0), ("C", 7.0))
        ])
        strategy = FactorTiltStrategy(factor_result(chars), ramp_months=3)
        first = strategy.propose_buys(context("ABC"))
        self.assertAlmostEqual(sum(first.values()), 100.0 / 3.0, delta=1e-5)
        self.assertEqual(strategy._ramp_remaining, 2)

        mid_month = strategy.propose_buys(context(
            "ABC", signal="2020-02-14", execution="2020-02-18"))
        self.assertEqual(mid_month, {})
        self.assertEqual(strategy._ramp_remaining, 2)

        second = strategy.propose_buys(context(
            "ABC", signal="2020-02-28", execution="2020-03-02"))
        self.assertAlmostEqual(sum(second.values()), 100.0 / 3.0, delta=1e-5)
        self.assertEqual(strategy._ramp_remaining, 1)

    def test_post_ramp_reinvests_matured_cash_mid_month(self):
        names = list("ABCDE")
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), name, 1.0, True) for name in names])
        strategy = FactorTiltStrategy(factor_result(chars), ramp_months=0)
        strategy._fit_date = pd.Timestamp("2020-01-31")
        strategy.scores = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3,
                                     "D": 0.4, "E": 0.5})
        budgets = strategy.propose_buys(context(
            names, signal="2020-02-14", execution="2020-02-18", cash=50.0))
        self.assertAlmostEqual(sum(budgets.values()), 50.0, delta=1e-5)
        self.assertNotIn("A", budgets)

    def test_regression_produces_scores_from_matured_targets(self):
        rng = np.random.default_rng(7)
        dates = pd.date_range("2017-01-31", periods=36, freq="ME")
        factors = pd.DataFrame(
            rng.normal(0.01, 0.04, (len(dates), 5)), index=dates,
            columns=["MKT", "SMB", "HML", "RMW", "CMA"])
        beta = np.array([[1.0, 0.5, 0.2, 0.1, 0.0],
                         [0.8, -0.2, 0.4, 0.0, 0.3]])
        prices = pd.DataFrame(
            100.0 * (1 + factors.to_numpy() @ beta.T + rng.normal(0, 0.01, (len(dates), 2))).cumprod(axis=0),
            index=dates, columns=["A", "B"])
        chars = characteristics([
            (dates[-1], "A", 1.0, True), (dates[-1], "B", 1.0, True)])
        strategy = FactorTiltStrategy(
            factor_result(chars, factors), window=60, min_obs=6, ramp_months=0)
        ctx = DecisionContext(
            dates[-1], dates[-1] + pd.Timedelta(days=1), prices,
            pd.DataFrame({"symbol": ["A", "B"]}), pd.DataFrame(),
            Portfolio(100.0, 100.0, 100.0, {}), 0)
        strategy.refit(None, ctx)
        self.assertEqual(set(strategy.scores.index), {"A", "B"})
        self.assertTrue((strategy._observations >= 6).all())
        self.assertEqual(int(strategy._observations.max()), 31)
        self.assertEqual(strategy._fit_date, dates[-1])
        # A zeroed or constant score would collapse to one value.
        self.assertEqual(len(set(np.round(strategy.scores.to_numpy(), 12))), 2)

    def test_refit_rejects_missing_or_misaligned_factor_panel(self):
        names = ["A", "B"]
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), name, 1.0, True) for name in names])
        dates = pd.date_range("2019-01-31", periods=12, freq="ME")
        prices = pd.DataFrame(100.0, index=dates, columns=names)
        ctx = DecisionContext(
            dates[-1], dates[-1] + pd.Timedelta(days=1), prices,
            pd.DataFrame({"symbol": names}), pd.DataFrame(),
            Portfolio(100.0, 100.0, 100.0, {}), 0)
        empty = FactorTiltStrategy(factor_result(chars), window=6, min_obs=6)
        with self.assertRaisesRegex(ValueError, "factor_returns"):
            empty.refit(None, ctx)

        shifted = pd.DataFrame(
            0.0, index=dates + pd.Timedelta(days=1),
            columns=["MKT", "SMB", "HML", "RMW", "CMA"])
        misaligned = FactorTiltStrategy(
            factor_result(chars, shifted), window=6, min_obs=6)
        with self.assertRaisesRegex(ValueError, "cover the signal calendar"):
            misaligned.refit(None, ctx)

    def test_min_order_notional_must_be_numeric(self):
        names = ["A", "B"]
        chars = characteristics([
            (pd.Timestamp("2020-01-31"), name, 1.0, True) for name in names])
        with self.assertRaisesRegex(ValueError, "min_order_notional"):
            FactorTiltStrategy(factor_result(chars), min_order_notional="x")

    def test_expected_factor_return_ignores_unmatured_rows(self):
        rng = np.random.default_rng(11)
        dates = pd.date_range("2017-01-31", periods=36, freq="ME")
        factors = pd.DataFrame(
            rng.normal(0.01, 0.04, (len(dates), 5)), index=dates,
            columns=["MKT", "SMB", "HML", "RMW", "CMA"])
        changed = factors.copy()
        changed.loc[dates[-1], :] += 100.0
        prices = pd.DataFrame(
            100.0 * (1 + factors.to_numpy() @ np.eye(5, 2)).cumprod(axis=0),
            index=dates, columns=["A", "B"])
        chars = characteristics([
            (dates[-1], "A", 1.0, True), (dates[-1], "B", 1.0, True)])
        ctx = DecisionContext(
            dates[-1], dates[-1] + pd.Timedelta(days=1), prices,
            pd.DataFrame({"symbol": ["A", "B"]}), pd.DataFrame(),
            Portfolio(100.0, 100.0, 100.0, {}), 0)
        first = FactorTiltStrategy(
            factor_result(chars, factors), window=60, min_obs=6, ramp_months=0)
        second = FactorTiltStrategy(
            factor_result(chars, changed), window=60, min_obs=6, ramp_months=0)
        first.refit(None, ctx)
        second.refit(None, ctx)
        pd.testing.assert_series_equal(first.scores, second.scores)

    def test_regression_score_sign_tracks_alpha(self):
        rng = np.random.default_rng(19)
        dates = pd.date_range("2017-01-31", periods=36, freq="ME")
        factors = pd.DataFrame(
            rng.normal(0.0, 0.02, (len(dates), 5)), index=dates,
            columns=["MKT", "SMB", "HML", "RMW", "CMA"])
        prices = pd.DataFrame({
            "A": 100.0 * (1.02 ** np.arange(len(dates))),
            "B": 100.0 * (0.98 ** np.arange(len(dates))),
        }, index=dates)
        chars = characteristics([
            (dates[-1], "A", 1.0, True), (dates[-1], "B", 1.0, True)])
        strategy = FactorTiltStrategy(
            factor_result(chars, factors), window=60, min_obs=6, ramp_months=0)
        ctx = DecisionContext(
            dates[-1], dates[-1] + pd.Timedelta(days=1), prices,
            pd.DataFrame({"symbol": ["A", "B"]}), pd.DataFrame(),
            Portfolio(100.0, 100.0, 100.0, {}), 0)
        strategy.refit(None, ctx)
        self.assertGreater(strategy.scores["A"], 0.0)
        self.assertLess(strategy.scores["B"], 0.0)

    def test_self_fitting_refit_runs_without_training_panel(self):
        dates, prices, benchmark, membership = inputs(8)
        strategy = SelfFittingStrategy()
        engine = Backtester(prices, benchmark, membership, strategy)
        result = engine.run(dates[1], dates[-1])
        self.assertEqual(len(result.refits), 1)
        self.assertEqual(result.refits.iloc[0]["status"], "success")
        self.assertEqual(result.refits.iloc[0]["rows"], 0)
        self.assertTrue(engine.strategy.received_none)
        self.assertEqual(engine.strategy.fits, 1)
        self.assertEqual(result.refits.model_version.tolist(), [1])
        self.assertEqual(engine.model_version, 1)

    def test_self_fitting_ignores_an_unused_training_panel(self):
        dates, prices, benchmark, membership = inputs(8)
        panel = pd.DataFrame({
            "date": [dates[2]], "symbol": ["A"],
            "available_date": [dates[2]], "label_end": [dates[5]],
            "target": [0.1], "feature": [1.0],
        })
        engine = Backtester(prices, benchmark, membership, SelfFittingStrategy(),
                            training_panel=panel, feature_columns=["feature"])
        result = engine.run(dates[1], dates[-1])
        self.assertEqual(result.refits.iloc[0]["status"], "success")
        self.assertEqual(result.refits.iloc[0]["rows"], 0)
        self.assertTrue(engine.strategy.received_none)

    def test_delisting_last_price_configuration_is_rejected(self):
        dates, prices, benchmark, membership = inputs(10)
        with self.assertRaisesRegex(ValueError, "stale mark"):
            Backtester(prices, benchmark, membership, BuyOnce(), delisting="last_price")

    def test_factor_strategy_runs_through_the_engine(self):
        from tests.test_famafrench import synthetic
        data = MarketData.from_frames(*synthetic())
        base = FamaFrench5(lag_months=6, weighting="equal").build(
            data, start="2017-01-01", end="2021-12-31")
        # The nine-name synthetic universe is too small to populate every 2x3
        # cell, so replace only the factor-return series with a non-degenerate
        # synthetic one; the eligible cross-sections come from the real builder.
        rng = np.random.default_rng(3)
        synthetic_returns = pd.DataFrame(
            rng.normal(0.005, 0.02, (len(base.factor_returns), 5)),
            index=base.factor_returns.index, columns=base.factor_returns.columns)
        factors = FactorResult(
            synthetic_returns, base.characteristics, base.portfolios, base.diagnostics)
        strategy = FactorTiltStrategy(
            factors, window=24, min_obs=12, ramp_months=2)
        engine = Backtester(market_data=data, strategy=strategy,
                            initial_cash=1000.0, fee_bps=0)
        result = engine.run("2020-01-02", "2020-12-31")
        self.assertGreaterEqual(len(result.refits), 10)
        self.assertTrue(result.refits.status.eq("success").all())
        self.assertGreater(len(result.fills), 0)
        self.assertGreater(len(engine.strategy.scores), 0)
        self.assertTrue((result.history.cash >= 0).all())

    def test_missing_quotes_stop_without_unobserved_or_early_sale(self):
        dates, prices, benchmark, membership = inputs(10)
        quotes = prices.loc[prices.date.le(dates[2])]
        engine = Backtester(quotes, benchmark, membership, BuyOnce(),
                            initial_cash=100, max_stale_sessions=1)
        with self.assertRaises(HeldPriceGapError):
            engine.run(dates[1], dates[-1])
        self.assertTrue(all(fill["side"] == "BUY" for fill in engine.oms.fills))
        self.assertEqual(len(engine.oms.lots), 1)
        self.assertEqual(engine.phase, Phase.FAILED)

    def test_forced_sale_validates_lot_and_date(self):
        oms = OMS(100.0, 0)
        with self.assertRaises(ValueError):
            oms.forced_sale(999, date="2020-01-03", price=10.0)
        order = oms.submit_buy("s", "A", 100.0, "2020-01-02", "2020-01-03")
        oms.fill(order, symbol="A", quote_date="2020-01-03",
                 price=10.0, session="2020-01-03")
        with self.assertRaises(ValueError):
            oms.forced_sale(order, date="2020-01-02", price=10.0)

    def test_refit_schema_is_stable_without_a_refit(self):
        dates, prices, benchmark, membership = inputs(6)

        class NoFit(Strategy):
            def __init__(self):
                super().__init__("no_fit")

            def propose_buys(self, context):
                return {}

        result = Backtester(prices, benchmark, membership, NoFit()).run(
            dates[1], dates[-1])
        self.assertEqual(list(result.refits.columns), [
            "signal_date", "model_version", "rows", "first_date", "last_date",
            "latest_label_end", "latest_available", "settings", "status"])
        self.assertTrue(result.refits.empty)

    def test_buy_fill_rejects_a_real_cash_shortfall(self):
        oms = OMS(100.0, 0)
        order = oms.submit_buy("s", "A", 100.0, "2020-01-02", "2020-01-03")
        oms.cash = 60.0
        with self.assertRaisesRegex(ValueError, "Insufficient cash"):
            oms.fill(order, symbol="A", quote_date="2020-01-03",
                     price=10.0, session="2020-01-03")

    def test_delisting_error_remains_the_default(self):
        dates, prices, benchmark, membership = inputs(10)
        quotes = prices.loc[prices.date.le(dates[2])]
        engine = Backtester(quotes, benchmark, membership, BuyOnce(),
                            initial_cash=100, max_stale_sessions=1)
        with self.assertRaises(HeldPriceGapError):
            engine.run(dates[1], dates[-1])


if __name__ == "__main__":
    unittest.main()
