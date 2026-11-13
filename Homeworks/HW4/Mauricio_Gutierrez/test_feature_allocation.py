"""Synthetic checks for the A/B/C feature and allocation experiment."""
import unittest
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from feature_engineering import (rolling_beta, market_relative_features, cross_sectional_ranks,
                                 add_benchmark_targets, load_development_inputs)
from strategy_backtest import (BacktestConfig, PortfolioSnapshot, InverseVolatilityStrategy,
                               inverse_volatility_weights, Backtester, reconcile_account,
                               closed_trade_errors, portfolio_chart_data)


class MarketFeatureTests(unittest.TestCase):
    def test_beta_two_and_pairwise_missing_observations(self):
        index = pd.bdate_range("2010-01-01", periods=400)
        market = pd.Series(np.sin(np.arange(400)) * 0.01, index=index)
        stock = 2 * market
        stock.iloc[150:160] = np.nan
        beta = rolling_beta(stock, market)
        self.assertTrue(beta.iloc[:125].isna().all())
        np.testing.assert_allclose(beta.dropna(), 2.0, atol=1e-12)
        self.assertTrue(rolling_beta(stock, market * 0).isna().all())

    def test_beta_and_other_market_features_do_not_read_future_prices(self):
        dates = pd.bdate_range("2010-01-01", periods=400)
        market_returns = np.sin(np.arange(400)) * 0.01
        market = pd.DataFrame({"date": dates, "adj_close": 100 * np.cumprod(1 + market_returns)})
        prices = pd.DataFrame({"symbol": "A", "date": dates, "adj_close": 100 * np.cumprod(1 + 2 * market_returns)})
        original = market_relative_features(prices, dates, market)
        changed_market, changed_prices = market.copy(), prices.copy()
        changed_market.loc[301:, "adj_close"] *= 4
        changed_prices.loc[301:, "adj_close"] *= 8
        changed = market_relative_features(changed_prices, dates, changed_market)
        pd.testing.assert_frame_equal(original.iloc[:301], changed.iloc[:301])
        prefix = market_relative_features(prices.iloc[:301], dates[:301], market.iloc[:301])
        pd.testing.assert_frame_equal(original.iloc[:301], prefix)

    def test_ranks_use_same_date_eligible_stocks_and_preserve_raw_volatility(self):
        panel = pd.DataFrame({"date": pd.to_datetime(["2013-01-02"] * 4 + ["2013-01-03"]),
                              "eligible": [True, True, True, False, True],
                              "momentum_63": [1.0, 2, 2, 99, -100],
                              "relative_momentum_63": [0.5, 1.5, 1.5, 98.5, -200],
                              "volatility_63": [0.1, 0.2, 0.3, 0.4, 0.5]})
        ranked = cross_sectional_ranks(panel, ["momentum_63", "relative_momentum_63", "volatility_63"])
        np.testing.assert_allclose(ranked.momentum_63_rank.iloc[:3], [1 / 3, 2.5 / 3, 2.5 / 3])
        pd.testing.assert_series_equal(ranked.momentum_63_rank, ranked.relative_momentum_63_rank, check_names=False)
        self.assertTrue(pd.isna(ranked.momentum_63_rank.iloc[3]))
        pd.testing.assert_series_equal(ranked.volatility_63, panel.volatility_63)

    def test_excess_target_uses_exact_entry_and_exit(self):
        benchmark = pd.DataFrame({"date": pd.to_datetime(["2013-01-02", "2013-01-03", "2013-04-03"]),
                                  "adj_close": [50.0, 100, 110]})
        panel = pd.DataFrame({"date": pd.to_datetime(["2013-01-02"] * 2),
                              "entry_date": pd.to_datetime(["2013-01-03"] * 2),
                              "label_end": pd.to_datetime(["2013-04-03", None]), "target": [0.2, np.nan]})
        result = add_benchmark_targets(panel, benchmark)
        self.assertAlmostEqual(result.excess_target.iloc[0], 0.1)
        self.assertTrue(pd.isna(result.excess_target.iloc[1]))

    def test_loader_excludes_post_cutoff_observations(self):
        scratch = (Path(__file__).resolve().parent / "outputs/test_scratch").resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            assert Path(temporary).resolve().is_relative_to(scratch)
            path = Path(temporary) / "data.duckdb"
            with duckdb.connect(str(path)) as con:
                con.execute("CREATE TABLE prices AS SELECT 'A' symbol, DATE '2015-12-31' date, 100.0 adj_close UNION ALL SELECT 'A', DATE '2016-01-04', -999.0")
                con.execute("CREATE TABLE calendar AS SELECT date FROM prices")
                con.execute("CREATE TABLE benchmark AS SELECT '^SP500TR' symbol, date, adj_close FROM prices")
                con.execute("CREATE TABLE universe AS SELECT 'A' symbol, 'Tech' sector")
                con.execute("CREATE TABLE universe_membership AS SELECT 'A' symbol, DATE '2015-10-01' quarter_start, DATE '2015-12-31' quarter_end, DATE '2015-09-30' selection_date, 1 universe_rank UNION ALL SELECT 'A', DATE '2016-01-01', DATE '2016-03-31', DATE '2015-12-31', 1")
            prices, calendar, membership, _, benchmark, audit = load_development_inputs(path)
            self.assertEqual(len(prices), 1)
            self.assertEqual(len(calendar), 1)
            self.assertEqual(len(membership), 1)
            self.assertEqual(audit.price_rows_read, 1)
            self.assertEqual(benchmark.date.max(), pd.Timestamp("2015-12-31"))


class InverseAllocationTests(unittest.TestCase):
    def test_weights_sum_to_one_and_lower_volatility_gets_more_cash(self):
        weights = inverse_volatility_weights(pd.Series({"A": 0.1, "B": 0.2, "C": 0.4}))
        self.assertAlmostEqual(weights.sum(), 1.0)
        np.testing.assert_allclose(weights, [4 / 7, 2 / 7, 1 / 7])
        with self.assertRaises(ValueError):
            inverse_volatility_weights(pd.Series({"A": 0.0}))

    def test_sector_replacement_recomputes_ten_weights_with_open_exposure(self):
        symbols = [f"S{i:02d}" for i in range(12)]
        scores = pd.DataFrame({"symbol": symbols, "prediction": np.arange(12, 0, -1), "volatility_63": 0.2})
        sectors = {s: "Tech" if i < 8 else f"Other{i}" for i, s in enumerate(symbols)}
        sectors["OLD"] = "Tech"
        cfg = BacktestConfig(top_n=10)
        snapshot = PortfolioSnapshot(pd.Timestamp("2013-01-02"), 750_000, pd.Series({"OLD": 250_000.0}))
        plan = InverseVolatilityStrategy(cfg).generate_orders(snapshot, scores, sectors)
        self.assertEqual(len(plan.symbols), 10)
        self.assertNotIn("S07", plan.symbols)
        self.assertIn("S10", plan.symbols)
        self.assertAlmostEqual(plan.planned_cash, snapshot.nav / 3)
        expected = inverse_volatility_weights(scores.set_index("symbol").loc[plan.symbols, "volatility_63"])
        np.testing.assert_allclose(pd.Series(plan.cash_by_symbol) / plan.budget, expected)
        after_nav = snapshot.nav - plan.budget * cfg.fee_rate / (1 + cfg.fee_rate)
        tech = 250_000 + sum(cash / (1 + cfg.fee_rate) for s, cash in plan.cash_by_symbol.items() if sectors[s] == "Tech")
        self.assertLessEqual(tech / after_nav, cfg.sector_limit)
        self.assertTrue((pd.Series(plan.cash_by_symbol) / (1 + cfg.fee_rate) / after_nav <= cfg.stock_limit).all())

    def test_stock_cap_missing_volatility_and_no_partial_batch(self):
        scores = pd.DataFrame({"symbol": list("ABCDE"), "prediction": [5, 4, 3, 2, 1],
                               "volatility_63": [0.00001, np.nan, 0.2, 0.3, 0.4]})
        cfg = BacktestConfig(top_n=3)
        snapshot = PortfolioSnapshot(pd.Timestamp("2013-01-02"), 1_000_000, pd.Series(dtype=float))
        sectors = {s: s for s in scores.symbol}
        plan = InverseVolatilityStrategy(cfg).generate_orders(snapshot, scores, sectors)
        self.assertEqual(plan.symbols, list("CDE"))
        self.assertEqual({r["reason"] for r in plan.rejections}, {"invalid_volatility", "stock_limit"})
        short = InverseVolatilityStrategy(cfg).generate_orders(snapshot, scores.iloc[:4], sectors)
        self.assertEqual(short.symbols, [])
        self.assertEqual(short.planned_cash, 0)

    def test_weighted_execution_cash_maturities_and_closed_trade_units(self):
        calendar = pd.bdate_range("2013-01-01", "2013-06-28")
        prices = pd.MultiIndex.from_product([calendar, list("ABCDE")], names=["date", "symbol"]).to_frame(index=False)
        prices["adj_close"] = 100.0
        forecasts = prices[["date", "symbol"]].copy()
        forecasts["prediction"] = forecasts.symbol.map(dict(zip("ABCDE", [5, 4, 3, 2, 1])))
        forecasts["volatility_63"] = forecasts.symbol.map(dict(zip("ABCDE", [0.1, 0.2, 0.3, 0.4, 0.5])))
        membership = pd.DataFrame([{"symbol": s, "quarter_start": pd.Timestamp(q), "selection_date": pd.Timestamp(q) - pd.Timedelta(days=1)}
                                   for q in ["2013-01-01", "2013-04-01"] for s in "ABCDE"])
        cfg = BacktestConfig(top_n=3)
        engine = Backtester(cfg, InverseVolatilityStrategy(cfg))
        result = engine.run(prices, calendar, forecasts, {s: s for s in "ABCDE"}, membership, calendar[0], calendar[-1])
        reconcile_account(result, calendar, cfg)
        first = result.orders.loc[result.orders.signal_date.eq(calendar[0])]
        self.assertGreater(first.loc[first.symbol.eq("A"), "cash_budget"].iloc[0], first.loc[first.symbol.eq("B"), "cash_budget"].iloc[0])
        changed = forecasts.copy()
        changed.loc[changed.date.gt("2013-03-01"), "volatility_63"] *= 4
        later = engine.run(prices, calendar, changed, {s: s for s in "ABCDE"}, membership, calendar[0], calendar[-1])
        pd.testing.assert_frame_equal(result.daily.loc[result.daily.date.le("2013-03-01")], later.daily.loc[later.daily.date.le("2013-03-01")])
        benchmark = pd.DataFrame({"date": calendar, "adj_close": 100 * 1.001 ** np.arange(len(calendar))})
        errors = closed_trade_errors(result, forecasts, forecasts, benchmark, excess_target=True)
        self.assertEqual(len(errors), result.trades.side.eq("SELL").sum())
        np.testing.assert_allclose(errors.realized_return, -errors.index_return)
        self.assertFalse(errors.lot_id.duplicated().any())

    def test_monthly_volatility_is_annualized_within_month(self):
        dates = pd.to_datetime(["2013-01-02", "2013-01-03", "2013-01-04", "2013-02-01", "2013-02-04"])
        curves = pd.DataFrame({"A": [100, 110, 99, 99, 99]}, index=dates)
        charts = portfolio_chart_data(curves)
        self.assertAlmostEqual(charts["monthly_volatility"].iloc[0, 0], np.sqrt(0.02) * np.sqrt(252))
        self.assertEqual(charts["monthly_volatility"].iloc[1, 0], 0)
        self.assertEqual(charts["monthly_nav"].iloc[0, 0], 99)


if __name__ == "__main__":
    unittest.main()
