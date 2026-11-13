"""Pairs_Tilt: common volume clock, pair research, and sleeve execution."""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtesting import DecisionContext, FactorResult, MarketData, Portfolio
from research.pairs import (PairsConfig, PairsResearch, PairsResult,
                            _cluster_cross_section, _kalman_hedge_ratio)
from strategies.pairs import PairsTiltStrategy, _scale_to_cash


def frame(symbol):
    return dict(
        symbol=symbol, fiscal_date=pd.Timestamp("2017-12-31"), period="quarter",
        filing_date=pd.Timestamp("2018-02-15"), revenue=100.0, net_income=10.0,
        operating_income=20.0, total_assets=500.0, total_debt=50.0,
        cash_and_equivalents=20.0, free_cash_flow=5.0, operating_cash_flow=15.0,
        total_equity=100.0, interest_expense=2.0, shares_outstanding=100.0,
        reported_currency="USD")


def synthetic():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2018-01-01", "2020-12-31")
    symbols = [f"S{i}" for i in range(12)]
    logs = {}
    for index, symbol in enumerate(symbols):
        if index % 2 == 0:
            trend = np.cumsum(rng.normal(0, 0.01, len(dates)))
        logs[symbol] = trend + rng.normal(0, 0.02, len(dates))
    prices = pd.DataFrame({s: np.exp(logs[s]) * 20.0 for s in symbols}, index=dates)
    long_prices = (prices.stack().rename("adj_close").reset_index()
                   .rename(columns={"level_0": "date", "level_1": "symbol"}))
    long_prices["close"] = long_prices["adj_close"]
    long_prices["volume"] = 1000
    benchmark = pd.DataFrame({"date": dates, "adj_close": 100.0})
    universe = pd.DataFrame({"symbol": symbols, "sector": ["Technology"] * len(symbols)})
    membership = pd.DataFrame([
        dict(symbol=symbol, quarter_start=q.to_period("Q").start_time,
             quarter_end=q, selection_date=q.to_period("Q").start_time - pd.Timedelta(days=1),
             universe_rank=1)
        for q in pd.date_range("2018-03-31", "2020-12-31", freq="QE")
        for symbol in symbols])
    fundamentals = pd.DataFrame([frame(symbol) for symbol in symbols])
    data = MarketData.from_frames(long_prices, fundamentals, universe,
                                  benchmark, membership)
    months = pd.date_range("2018-01-31", "2020-12-31", freq="ME")
    characteristics = pd.DataFrame([
        dict(date=date, symbol=symbol, ME=float(rng.uniform(1e8, 1e9)),
             BM=float(rng.uniform(0.1, 1.0)), OP=float(rng.uniform(0.01, 0.2)),
             INV=float(rng.uniform(-0.1, 0.2)), eligible=True,
             sector="Technology")
        for date in months for symbol in symbols])
    factors = FactorResult(pd.DataFrame(), characteristics, pd.DataFrame(), pd.DataFrame())
    return data, factors


def empty_clock():
    return None


def context(symbols, signal="2020-01-31", execution="2020-02-03",
            nav=90.0, cash=None, holdings=None):
    dates = pd.bdate_range("2019-12-02", "2020-02-28")
    prices = pd.DataFrame(1.0, index=dates, columns=list(symbols))
    membership = pd.DataFrame({"symbol": list(symbols)})
    cash = nav if cash is None else cash
    return DecisionContext(
        pd.Timestamp(signal), pd.Timestamp(execution), prices, membership,
        pd.DataFrame(), Portfolio(cash, cash, nav, dict(holdings or {})), 0)


def signals_frame(rows):
    columns = ["date", "pair_id", "leg_a", "leg_b", "long_symbol", "cluster",
               "beta", "alpha", "zscore", "adf_p", "adf_p_adj", "half_life",
               "correlation"]
    return pd.DataFrame(rows, columns=columns)


class PairsResearchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.factors = synthetic()
        cls.config = PairsConfig(
            min_clusters=2, max_clusters=4, min_cluster_size=2,
            candidate_pairs_per_cluster=20, pair_lookback_bars=30,
            min_history_bars=25, min_half_life=0.1, max_half_life=100.0,
            entry_z=0.0, cluster_refresh_bars=10)
        cls.result = PairsResearch(cls.config).build(
            cls.data, cls.factors, start="2018-01-01", end="2020-12-31")

    def test_default_config_matches_documented_gates(self):
        config = PairsConfig()
        self.assertEqual((config.entry_z, config.min_half_life,
                          config.max_half_life, config.adf_pvalue,
                          config.sector_dummies, config.max_stale_bars),
                         (2.0, 2.0, 26.0, 0.05, False, 0))

    def test_entry_z_gate_is_binding(self):
        strict = replace(self.config, entry_z=0.5)
        result = PairsResearch(strict).build(
            self.data, self.factors, start="2018-01-01", end="2020-12-31")
        self.assertGreater(len(result.signals), 0)
        self.assertGreaterEqual(result.signals.zscore.abs().min(), 0.5)

    def test_common_clock_and_eligible_pairs(self):
        bars = self.result.bar_clock.bars
        self.assertGreater(len(bars), 20)
        # Constant volume and a 10x target imply roughly ten-session bars.
        self.assertTrue(bars["days"].between(5, 20).all())
        self.assertGreater(len(self.result.signals), 0)
        self.assertTrue(self.result.signals.long_symbol.isin(
            set(self.result.signals.leg_a) | set(self.result.signals.leg_b)).all())
        self.assertTrue(self.result.signals.date.le(pd.Timestamp("2020-12-31")).all())

    def test_rebalance_cadence_uses_a_subset_of_bar_dates(self):
        legacy = PairsResearch(replace(self.config, rebalance_days=None)).build(
            self.data, self.factors, start="2018-01-01", end="2020-12-31")
        self.assertLessEqual(len(self.result.coverage),
                             len(legacy.coverage))
        self.assertTrue(set(self.result.coverage.date).issubset(
            set(legacy.coverage.date)))
        # Explicit signal dates are honored and mapped causally.
        explicit_dates = list(legacy.coverage.date[::10])
        explicit = PairsResearch(self.config).build(
            self.data, self.factors, start="2018-01-01", end="2020-12-31",
            signal_dates=explicit_dates)
        self.assertEqual(list(explicit.coverage.date), explicit_dates)

    def test_future_characteristics_do_not_change_prior_output(self):
        cutoff = pd.Timestamp("2020-06-30")
        chars = self.factors.characteristics.copy()
        mask = chars.date.gt(cutoff)
        chars.loc[mask, "ME"] *= 100.0
        chars.loc[mask, "eligible"] = False
        changed = FactorResult(pd.DataFrame(), chars, pd.DataFrame(), pd.DataFrame())
        second = PairsResearch(self.config).build(
            self.data, changed, start="2018-01-01", end="2020-12-31")
        before = self.result.signals.loc[self.result.signals.date.le(cutoff)].reset_index(drop=True)
        after = second.signals.loc[second.signals.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)
        before_clusters = self.result.clusters.loc[self.result.clusters.date.le(cutoff)].reset_index(drop=True)
        after_clusters = second.clusters.loc[second.clusters.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before_clusters, after_clusters)

    def test_future_volume_does_not_change_prior_pair_signals(self):
        cutoff = pd.Timestamp("2020-06-30")
        prices, fundamentals, universe, benchmark, membership = (
            self.data.prices, self.data.fundamentals, self.data.universe,
            self.data.benchmark, self.data.membership)
        changed = prices.copy()
        mask = changed.date.gt(cutoff)
        changed.loc[mask, "volume"] *= 50.0
        data = MarketData.from_frames(changed, fundamentals, universe,
                                      benchmark, membership)
        second = PairsResearch(self.config).build(
            data, self.factors, start="2018-01-01", end="2020-12-31")
        before = self.result.signals.loc[self.result.signals.date.le(cutoff)].reset_index(drop=True)
        after = second.signals.loc[second.signals.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_future_prices_do_not_change_prior_pair_signals(self):
        cutoff = pd.Timestamp("2020-06-30")
        prices, fundamentals, universe, benchmark, membership = (
            self.data.prices, self.data.fundamentals, self.data.universe,
            self.data.benchmark, self.data.membership)
        changed = prices.copy()
        mask = changed.date.gt(cutoff)
        changed.loc[mask, ["close", "adj_close"]] *= 7.0
        data = MarketData.from_frames(changed, fundamentals, universe,
                                      benchmark, membership)
        second = PairsResearch(self.config).build(
            data, self.factors, start="2018-01-01", end="2020-12-31")
        before = self.result.signals.loc[self.result.signals.date.le(cutoff)].reset_index(drop=True)
        after = second.signals.loc[second.signals.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)


class PairsStatisticsChecks(unittest.TestCase):
    def test_kalman_recovers_a_known_hedge_ratio(self):
        rng = np.random.default_rng(21)
        x = np.linspace(2.0, 4.0, 80)
        y = 1.5 + 2.0 * x + rng.normal(0, 0.01, len(x))
        alpha, beta, zscore = _kalman_hedge_ratio(y, x, PairsConfig())
        self.assertAlmostEqual(alpha, 1.5, delta=0.15)
        self.assertAlmostEqual(beta, 2.0, delta=0.15)
        self.assertTrue(np.isfinite(zscore))

    def test_cluster_two_groups_at_small_cross_section_size(self):
        rng = np.random.default_rng(8)
        rows = []
        for index in range(20):
            group = 0.2 if index < 10 else 0.8
            rows.append(dict(symbol=f"S{index}", ME=1e8 + index * 1e6,
                             BM=group + rng.normal(0, 0.01),
                             OP=group / 2 + rng.normal(0, 0.01),
                             INV=group / 4 + rng.normal(0, 0.01),
                             sector="Technology", eligible=True))
        cross = pd.DataFrame(rows).set_index("symbol")
        labels = _cluster_cross_section(cross, PairsConfig(
            min_clusters=5, max_clusters=8, min_cluster_size=3))
        self.assertEqual(labels.nunique(), 2)

    def test_kalman_z_is_unit_scale(self):
        rng = np.random.default_rng(12)
        values = []
        for _ in range(150):
            x = rng.uniform(2.0, 4.0, 104)
            y = 1.5 + 2.0 * x + rng.normal(0, 0.02, 104)
            values.append(_kalman_hedge_ratio(y, x, PairsConfig())[2])
        self.assertAlmostEqual(float(np.mean(values)), 0.0, delta=0.25)
        self.assertGreater(float(np.std(values)), 0.6)
        self.assertLess(float(np.std(values)), 1.5)

    def test_cluster_does_not_collapse_on_a_single_outlier(self):
        rng = np.random.default_rng(3)
        rows = []
        for index in range(24):
            group = 0.2 if index < 12 else 0.8
            rows.append(dict(symbol=f"S{index}", ME=1e8 + index * 1e6,
                             BM=group + rng.normal(0, 0.01),
                             OP=group / 2 + rng.normal(0, 0.01),
                             INV=group / 4 + rng.normal(0, 0.01),
                             sector="Technology", eligible=True))
        rows.append(dict(symbol="OUT", ME=1e15, BM=1e6, OP=1e6, INV=1e6,
                         sector="Technology", eligible=True))
        cross = pd.DataFrame(rows).set_index("symbol")
        labels = _cluster_cross_section(cross, PairsConfig(
            min_clusters=2, max_clusters=6, min_cluster_size=3))
        self.assertGreater(labels.nunique(), 1)
        self.assertGreaterEqual(labels.value_counts().min(), 3)

class PairsStrategyChecks(unittest.TestCase):
    def setUp(self):
        names = list("ABCDEFG")
        self.chars = pd.DataFrame([
            dict(date=pd.Timestamp("2020-01-31"), symbol=name,
                 ME=float(index + 1), eligible=True, sector="Technology",
                 BM=0.5, OP=0.1, INV=0.0)
            for index, name in enumerate(names)])
        self.factors = FactorResult(pd.DataFrame(), self.chars,
                                    pd.DataFrame(), pd.DataFrame())
        date = pd.Timestamp("2020-01-31")
        self.signals = signals_frame([
            (date, "A|B", "A", "B", "A", 0, 1.0, 0.0, -2.5, 0.01, 0.01, 5.0, 0.8),
            (date, "A|C", "A", "C", "C", 0, 1.0, 0.0, -2.0, 0.02, 0.02, 6.0, 0.7),
            (date, "D|E", "D", "E", "E", 1, 1.0, 0.0, 2.4, 0.03, 0.03, 7.0, 0.6),
            (date, "F|G", "F", "G", "F", 2, 1.0, 0.0, 2.2, 0.04, 0.04, 8.0, 0.5),
        ])
        self.pairs = PairsResult(self.signals, pd.DataFrame(), pd.DataFrame(),
                                 pd.DataFrame(), empty_clock())

    def test_one_pair_per_stock_and_equal_pair_capital(self):
        strategy = PairsTiltStrategy(self.pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=1)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=90.0))
        # A is used by A|B, so A|C is skipped. D|E and F|G fill the three slots.
        self.assertIn("A", budgets)
        self.assertNotIn("C", budgets)
        self.assertIn("E", budgets)
        self.assertIn("F", budgets)
        self.assertAlmostEqual(budgets["A"], 30.0, delta=1e-5)
        self.assertAlmostEqual(budgets["E"], 30.0, delta=1e-5)
        self.assertAlmostEqual(budgets["F"], 30.0, delta=1e-5)

    def test_unused_pair_capital_falls_back_to_market(self):
        self.pairs = PairsResult(self.signals.iloc[:1], pd.DataFrame(),
                                 pd.DataFrame(), pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(self.pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=1)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=90.0))
        self.assertAlmostEqual(budgets["A"], 30.0, delta=1e-5)
        self.assertAlmostEqual(sum(budgets.values()), 90.0, delta=1e-5)
        self.assertGreater(len(budgets), 2)  # fallback names are present

    def test_ramp_invests_one_ninth_and_waits_ten_days(self):
        strategy = PairsTiltStrategy(self.pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=9, min_order_notional=0.0)
        first = strategy.propose_buys(context(list("ABCDEFG"), cash=90.0))
        self.assertAlmostEqual(sum(first.values()), 10.0, delta=1e-5)
        self.assertEqual(strategy._ramp_remaining, 8)
        skipped = strategy.propose_buys(context(
            list("ABCDEFG"), signal="2020-02-05", execution="2020-02-06",
            cash=80.0))
        self.assertEqual(skipped, {})
        second = strategy.propose_buys(context(
            list("ABCDEFG"), signal="2020-02-10", execution="2020-02-11",
            cash=80.0))
        self.assertAlmostEqual(sum(second.values()), 10.0, delta=1e-5)
        self.assertEqual(strategy._ramp_remaining, 7)

    def test_no_pairs_holds_the_market(self):
        self.pairs = PairsResult(pd.DataFrame(columns=self.signals.columns),
                                 pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                 empty_clock())
        strategy = PairsTiltStrategy(self.pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=1)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=90.0))
        self.assertAlmostEqual(sum(budgets.values()), 90.0, delta=1e-5)
        self.assertGreater(len(budgets), 3)


    def test_ramp_is_not_consumed_without_a_cross_section(self):
        names = list("ABCDEFG")
        late = pd.DataFrame([
            dict(date=pd.Timestamp("2020-03-31"), symbol=name,
                 ME=float(index + 1), eligible=True, sector="Technology",
                 BM=0.5, OP=0.1, INV=0.0)
            for index, name in enumerate(names)])
        factors = FactorResult(pd.DataFrame(), late, pd.DataFrame(), pd.DataFrame())
        strategy = PairsTiltStrategy(self.pairs, factors, ramp_sleeves=9,
                                     min_order_notional=0.0)
        first = strategy.propose_buys(context(names, signal="2020-01-31",
                                              execution="2020-02-03", cash=90.0))
        self.assertEqual(first, {})
        self.assertEqual(strategy._ramp_remaining, 9)
        second = strategy.propose_buys(context(names, signal="2020-03-31",
                                               execution="2020-04-01", cash=90.0))
        self.assertAlmostEqual(sum(second.values()), 10.0, delta=1e-5)
        self.assertEqual(strategy._ramp_remaining, 8)

    def test_post_ramp_invests_at_most_one_sleeve(self):
        strategy = PairsTiltStrategy(self.pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=9, min_order_notional=0.0)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=900.0,
                                                  nav=900.0))
        self.assertGreaterEqual(sum(budgets.values()), 100.0 - 1e-5)
        self.assertLessEqual(sum(budgets.values()), 100.0 + 1e-5)

    def test_pair_requires_both_legs_to_be_quotable(self):
        names = list("ABCDEFG")
        chars = pd.DataFrame([
            dict(date=pd.Timestamp("2020-01-31"), symbol=name, ME=1.0,
                 eligible=True, sector="Technology", BM=0.5, OP=0.1, INV=0.0)
            for name in names])
        factors = FactorResult(pd.DataFrame(), chars, pd.DataFrame(), pd.DataFrame())
        date = pd.Timestamp("2020-01-31")
        signals = signals_frame([
            (date, "A|B", "A", "B", "A", 0, 1.0, 0.0, -2.5, .01, .01, 5.0, .8),
            (date, "C|D", "C", "D", "C", 1, 1.0, 0.0, -2.5, .01, .01, 5.0, .8),
        ])
        pairs = PairsResult(signals, pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(pairs, factors, max_pairs=1,
                                     ramp_sleeves=1, min_order_notional=0.0)
        strategy._ramp_remaining = 0
        ctx = context(names, cash=90.0)
        membership = ctx.membership.loc[ctx.membership.symbol.ne("B")]
        ctx = DecisionContext(ctx.signal_date, ctx.execution_date, ctx.prices,
                              membership, pd.DataFrame(), ctx.portfolio, 0)
        budgets = strategy.propose_buys(ctx)
        self.assertIn("C", budgets)
        self.assertNotIn("A", budgets)

    def test_stale_signal_falls_back_to_market(self):
        old = self.signals.copy()
        old["date"] = pd.Timestamp("2019-06-28")
        pairs = PairsResult(old, pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=1, min_order_notional=0.0,
                                     max_signal_age_days=45)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=90.0))
        self.assertAlmostEqual(sum(budgets.values()), 90.0, delta=1e-5)
        self.assertGreater(len(budgets), 3)

    def test_identical_leg_pair_is_skipped(self):
        date = pd.Timestamp("2020-01-31")
        rows = signals_frame([
            (date, "A|A", "A", "A", "A", 0, 1.0, 0.0, -2.5, .01, .01, 5.0, .8),
        ])
        names = list("ABCDEFG")
        chars = pd.DataFrame([
            dict(date=date, symbol=name, ME=float(index + 1), eligible=True,
                 sector="Technology", BM=0.5, OP=0.1, INV=0.0)
            for index, name in enumerate(names)])
        factors = FactorResult(pd.DataFrame(), chars, pd.DataFrame(), pd.DataFrame())
        pairs = PairsResult(rows, pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(pairs, factors, ramp_sleeves=1,
                                     min_order_notional=0.0)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(names, cash=90.0))
        self.assertAlmostEqual(sum(budgets.values()), 90.0, delta=1e-5)
        self.assertGreater(len(budgets), 3)

    def test_tiny_sleeve_still_deploys_fallback(self):
        names = list("ABCDEFG")
        date = pd.Timestamp("2020-01-31")
        signals = signals_frame([
            (date, "A|B", "A", "B", "A", 0, 1.0, 0.0, -2.5, .01, .01, 5.0, .8),
        ])
        chars = pd.DataFrame([
            dict(date=date, symbol=name, ME=float(index + 1), eligible=True,
                 sector="Technology", BM=0.5, OP=0.1, INV=0.0)
            for index, name in enumerate(names)])
        factors = FactorResult(pd.DataFrame(), chars, pd.DataFrame(), pd.DataFrame())
        pairs = PairsResult(signals, pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(pairs, factors, max_pairs=3,
                                     ramp_sleeves=1, min_order_notional=1.0)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(names, cash=3.0, nav=3.0))
        self.assertGreater(sum(budgets.values()), 0.0)

    def test_delisted_name_is_not_used_by_the_fallback(self):
        ctx = context(list("ABCDEFG"), cash=90.0)
        prices = ctx.prices.copy()
        prices.iloc[-1, prices.columns.get_loc("A")] = np.nan
        ctx = DecisionContext(ctx.signal_date, ctx.execution_date, prices,
                              ctx.membership, pd.DataFrame(), ctx.portfolio, 0)
        strategy = PairsTiltStrategy(self.pairs, self.factors, max_pairs=3,
                                     ramp_sleeves=1, min_order_notional=0.0)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(ctx)
        self.assertNotIn("A", budgets)

    def test_max_pairs_cap_with_four_disjoint_pairs(self):
        date = pd.Timestamp("2020-01-31")
        rows = []
        for index, pair in enumerate(("A|B", "C|D", "E|F", "G|H")):
            first, second = pair.split("|")
            rows.append((date, pair, first, second, first, index, 1.0, 0.0,
                         -2.5, 0.01, 0.01, 5.0, 0.8))
        names = list("ABCDEFGH")
        chars = pd.DataFrame([
            dict(date=date, symbol=name, ME=1.0, eligible=True,
                 sector="Technology", BM=0.5, OP=0.1, INV=0.0)
            for name in names])
        factors = FactorResult(pd.DataFrame(), chars, pd.DataFrame(), pd.DataFrame())
        pairs = PairsResult(signals_frame(rows), pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(pairs, factors, max_pairs=3,
                                     ramp_sleeves=1, min_order_notional=0.0)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(names, cash=90.0))
        self.assertEqual(len([name for name in budgets if name in "ACEG"]), 3)

    def test_future_signal_falls_back_to_market(self):
        future = self.signals.copy()
        future["date"] = pd.Timestamp("2021-01-31")
        pairs = PairsResult(future, pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), empty_clock())
        strategy = PairsTiltStrategy(pairs, self.factors, ramp_sleeves=1,
                                     min_order_notional=0.0)
        strategy._ramp_remaining = 0
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=90.0))
        self.assertAlmostEqual(sum(budgets.values()), 90.0, delta=1e-5)
        self.assertGreater(len(budgets), 3)

    def test_exact_floor_is_not_stranded(self):
        self.assertEqual(_scale_to_cash({"A": 1000.0}, 1000.0, 1000.0),
                         {"A": 1000.0})

    def test_small_account_ramp_still_progresses(self):
        strategy = PairsTiltStrategy(self.pairs, self.factors, ramp_sleeves=9)
        budgets = strategy.propose_buys(context(list("ABCDEFG"), cash=3.0, nav=3.0))
        self.assertGreater(sum(budgets.values()), 0.0)
        self.assertEqual(strategy._ramp_remaining, 8)

if __name__ == "__main__":
    unittest.main()
