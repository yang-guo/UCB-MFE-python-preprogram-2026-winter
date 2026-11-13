"""Small accounting and timing cases. No database or notebook is required.

Run from the repository root:
python -m unittest discover -s Homeworks/HW4/Mauricio_Gutierrez -p 'test_*.py' -v
Synthetic business-day calendars are test fixtures, not exchange calendars.
"""
import unittest
import io

import numpy as np
import pandas as pd

from strategy_backtest import (
    BacktestConfig, Backtester, MissingHeldPriceError, MonthlyRidgeStrategy,
    PortfolioSnapshot, compare_performance,
)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2010-01-01", "2010-07-30")
        self.symbols = list("ABCDEFG")
        self.prices = pd.MultiIndex.from_product(
            [self.calendar, self.symbols], names=["date", "symbol"]
        ).to_frame(index=False).assign(adj_close=100.0)
        self.forecasts = self.prices[["date", "symbol"]].copy()
        self.forecasts["prediction"] = self.forecasts.symbol.map(dict(zip(self.symbols, range(7, 0, -1))))
        self.sectors = {s: f"sector_{s}" for s in self.symbols}
        self.membership = pd.DataFrame([
            {"symbol": s, "quarter_start": pd.Timestamp(q), "selection_date": pd.Timestamp(q) - pd.Timedelta(days=1)}
            for q in ["2010-01-01", "2010-04-01", "2010-07-01"] for s in self.symbols
        ])
        self.cfg = BacktestConfig()

    def run_case(self, prices=None, forecasts=None, end="2010-07-30", membership=None):
        return Backtester(self.cfg).run(
            self.prices if prices is None else prices, self.calendar,
            self.forecasts if forecasts is None else forecasts, self.sectors,
            self.membership if membership is None else membership, "2010-01-01", end,
        )

    def test_first_purchase_amount_fee_and_next_session(self):
        result = self.run_case(end="2010-01-08")
        buys = result.trades
        self.assertEqual(len(buys), 5)
        self.assertTrue(buys.date.eq(pd.Timestamp("2010-01-04")).all())
        self.assertTrue(buys.signal_date.eq(pd.Timestamp("2010-01-01")).all())
        np.testing.assert_allclose(buys.cash_cost, 1_000_000 / 15)
        np.testing.assert_allclose(buys.notional, (1_000_000 / 15) / 1.001)
        self.assertAlmostEqual(result.daily.iloc[-1].nav, 1_000_000 - buys.fee.sum())
        self.assertAlmostEqual(result.daily.iloc[-1].cash, 1_000_000 * 2 / 3)

    def test_cash_ledger_and_marked_open_lots_reconcile(self):
        result = self.run_case()
        flows = result.trades.groupby("date").cash_change.sum().reindex(result.daily.date, fill_value=0)
        np.testing.assert_allclose(result.daily.cash, self.cfg.initial_cash + flows.cumsum(), atol=1e-7)
        np.testing.assert_allclose(result.daily.nav, self.cfg.initial_cash - result.daily.fees.cumsum(), atol=1e-7)
        self.assertGreaterEqual(result.daily.cash.min(), -1e-7)
        self.assertAlmostEqual(result.open_lots.market_value.sum(), result.daily.iloc[-1].holdings_value)
        self.assertTrue((result.open_lots.due_date > result.daily.iloc[-1].date).all())

    def test_each_lot_has_its_own_calendar_day_exit(self):
        result = self.run_case()
        buys = result.trades.loc[result.trades.side.eq("BUY")]
        sells = result.trades.loc[result.trades.side.eq("SELL")]
        self.assertTrue(buys.symbol.duplicated().any())
        self.assertFalse(sells.lot_id.duplicated().any())
        for row in sells.itertuples():
            due = row.entry_date + pd.Timedelta(days=90)
            expected_exit = self.calendar[self.calendar.searchsorted(due)]
            self.assertEqual(row.date, expected_exit)
        first_exit = sells.loc[sells.lot_id.eq(1), "date"].iloc[0]
        self.assertEqual(first_exit, pd.Timestamp("2010-04-05"))  # Sunday maturity.

    def test_sector_replacement_and_all_existing_exposure(self):
        scores = self.forecasts.loc[self.forecasts.date.eq(self.calendar[0])]
        sectors = {**self.sectors, "OLD1": "tech", "OLD2": "tech", "A": "tech", "B": "tech"}
        snapshot = PortfolioSnapshot(self.calendar[0], 550_000, pd.Series({"OLD1": 225_000.0, "OLD2": 225_000.0}))
        plan = MonthlyRidgeStrategy(self.cfg).generate_orders(snapshot, scores, sectors)
        self.assertEqual(plan.symbols, list("CDEFG"))
        self.assertEqual([r["reason"] for r in plan.rejections], ["sector_limit", "sector_limit"])
        self.assertAlmostEqual(plan.budget, 1_000_000 / 3)
        # This aggregate can include several purchases of A.
        snapshot = PortfolioSnapshot(self.calendar[0], 710_000, pd.Series({"A": 290_000.0}))
        plan = MonthlyRidgeStrategy(self.cfg).generate_orders(snapshot, scores, self.sectors)
        self.assertEqual(plan.symbols, list("BCDEF"))
        self.assertEqual(plan.rejections[0]["reason"], "stock_limit")

    def test_cash_cap_unknown_sector_short_list_and_negative_scores(self):
        strategy = MonthlyRidgeStrategy(self.cfg)
        snapshot = PortfolioSnapshot(self.calendar[0], 100_000, pd.Series({s: 180_000.0 for s in list("CDEFG")}))
        scores = self.forecasts.loc[self.forecasts.date.eq(self.calendar[0])].copy()
        scores["prediction"] -= 10  # All forecasts negative; no positive-return filter.
        plan = strategy.generate_orders(snapshot, scores, self.sectors)
        self.assertEqual(len(plan.symbols), 5)
        self.assertEqual(plan.budget, 100_000)
        missing_sector = {**self.sectors, "A": None}
        plan = strategy.generate_orders(snapshot, scores, missing_sector)
        self.assertNotIn("A", plan.symbols)
        plan = strategy.generate_orders(snapshot, scores.head(4), self.sectors)
        self.assertEqual(plan.symbols, [])
        self.assertEqual(plan.reason, "fewer_than_required_stocks")

    def test_future_data_cannot_change_earlier_decisions(self):
        before = self.run_case()
        changed_prices = self.prices.copy()
        changed_prices.loc[changed_prices.date.gt("2010-05-03"), "adj_close"] *= 1.5
        changed_forecasts = self.forecasts.copy()
        changed_forecasts.loc[changed_forecasts.date.gt("2010-05-03"), "prediction"] *= -1
        changed_forecasts["target"] = 1e9
        changed_forecasts["exit_price"] = 0
        after = self.run_case(changed_prices, changed_forecasts)
        pd.testing.assert_frame_equal(before.daily.loc[before.daily.date.le("2010-05-03")],
                                      after.daily.loc[after.daily.date.le("2010-05-03")])
        pd.testing.assert_frame_equal(before.decisions.loc[before.decisions.signal_date.le("2010-05-03")],
                                      after.decisions.loc[after.decisions.signal_date.le("2010-05-03")])
        prefix = self.run_case(end="2010-05-03")
        pd.testing.assert_frame_equal(before.daily.loc[before.daily.date.le("2010-05-03")], prefix.daily)

    def test_missing_entry_cancels_without_future_replacement(self):
        prices = self.prices.loc[~(self.prices.date.eq("2010-01-04") & self.prices.symbol.eq("A"))]
        result = self.run_case(prices=prices, end="2010-01-08")
        self.assertEqual(result.decisions.iloc[0].selected_count, 5)
        self.assertEqual(len(result.trades), 4)
        self.assertEqual(result.orders.loc[result.orders.symbol.eq("A"), "status"].iloc[0], "cancelled_missing_entry")
        self.assertNotIn("F", result.trades.symbol.tolist())

    def test_missing_held_or_exit_quote_stops_the_run(self):
        for date in ["2010-01-05", "2010-04-05"]:
            prices = self.prices.loc[~(self.prices.date.eq(date) & self.prices.symbol.eq("A"))]
            with self.assertRaises(MissingHeldPriceError):
                self.run_case(prices=prices)

    def test_membership_is_dated_and_only_first_month_session_is_used(self):
        membership = self.membership.copy()
        membership.loc[membership.symbol.eq("A"), "selection_date"] = pd.Timestamp("2011-01-01")
        result = self.run_case(membership=membership, end="2010-01-08")
        self.assertNotIn("A", result.trades.symbol.tolist())
        forecasts = self.forecasts.loc[self.forecasts.date.ne(self.calendar[0])]
        result = self.run_case(forecasts=forecasts, end="2010-01-08")
        self.assertTrue(result.trades.empty)
        self.assertEqual(len(result.decisions), 1)
        for name in ["trades", "orders", "rejections", "open_lots"]:
            table = getattr(result, name)
            saved = pd.read_csv(io.StringIO(table.to_csv(index=False)))
            self.assertTrue(saved.empty)
            self.assertEqual(saved.columns.tolist(), table.columns.tolist())

    def test_price_drift_is_reported_without_early_sale(self):
        prices = self.prices.copy()
        prices.loc[prices.date.ge("2010-01-05") & prices.symbol.eq("A"), "adj_close"] = 1_000.0
        result = self.run_case(prices=prices, end="2010-03-31")
        self.assertTrue(result.daily.exposure_breach.any())
        self.assertFalse(result.trades.side.eq("SELL").any())
        self.assertIn("existing_exposure_limit", result.decisions.reason.tolist())

    def test_benchmark_dates_metrics_and_missing_quote(self):
        daily = pd.DataFrame({"date": pd.to_datetime(["2010-01-01", "2010-01-04", "2010-01-05"]),
                              "nav": [1_000_000.0, 1_100_000.0, 990_000.0]})
        benchmark = daily[["date"]].assign(adj_close=[100.0, 110.0, 99.0])
        curves, metrics = compare_performance(daily, benchmark, self.cfg)
        np.testing.assert_allclose(curves.Strategy, curves["S&P 500 TR"])
        self.assertAlmostEqual(metrics.loc["Strategy", "max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics.loc["Strategy", "total_return"], -0.01)
        self.assertAlmostEqual(metrics.loc["Strategy", "CAGR"], 0.99 ** (365.25 / 4) - 1)
        with self.assertRaises(ValueError):
            compare_performance(daily, benchmark.iloc[:2], self.cfg)


class NoQuoteTests(unittest.TestCase):
    def case(self, missing=(), end=None, recovery=1.0, holding_days=4):
        from strategy_backtest import reconcile_account
        calendar = pd.bdate_range('2013-01-02', periods=25)
        prices = pd.DataFrame({'date':calendar,'symbol':'A','adj_close':100.0 + np.arange(25)})
        prices = prices.loc[~prices.date.isin(missing)]
        forecasts = pd.DataFrame({'date':[calendar[0]],'symbol':['A'],'prediction':[0.1]})
        membership = pd.DataFrame({'symbol':['A'],'quarter_start':[pd.Timestamp('2013-01-01')],
                                   'selection_date':[pd.Timestamp('2012-12-31')]})
        cfg = BacktestConfig(top_n=1, holding_days=holding_days, stock_limit=1, sector_limit=1)
        result = Backtester(cfg, allow_stale_quotes=True, no_quote_recovery=recovery).run(
            prices, calendar, forecasts, {'A':'sector'}, membership, calendar[0], end or calendar[-1])
        reconcile_account(result, calendar, cfg)
        return result, calendar, cfg

    def test_stale_mark_uses_last_observation_and_flags_session(self):
        result, _, _ = self.case([pd.Timestamp('2013-01-04')], end='2013-01-04')
        flag = result.valuation_flags.iloc[0]
        self.assertEqual(flag.mark_price, 101)
        self.assertEqual(flag.price_date, pd.Timestamp('2013-01-03'))
        self.assertAlmostEqual(result.daily.iloc[-1].holdings_value, flag.market_value)
        self.assertEqual(result.open_lots.iloc[0].mark_price, 101)

    def test_exit_uses_first_quote_after_maturity(self):
        result, _, _ = self.case(pd.bdate_range('2013-01-07','2013-01-09'))
        sell = result.trades.loc[result.trades.side.eq('SELL')].iloc[0]
        self.assertEqual(sell.date, pd.Timestamp('2013-01-10'))
        self.assertFalse(sell.no_quote_exit)

    def test_ten_sessions_timeout_uses_last_close(self):
        result, cal, _ = self.case(pd.bdate_range('2013-01-07','2013-02-05'))
        sell = result.trades.loc[result.trades.side.eq('SELL')].iloc[0]
        deadline = cal[cal.searchsorted(pd.Timestamp('2013-01-07'), side='right')+9]
        self.assertEqual(sell.date, deadline)
        self.assertTrue(sell.no_quote_exit)
        self.assertEqual(sell.price, 102)

    def test_quote_on_deadline_has_priority(self):
        result, _, _ = self.case(pd.bdate_range('2013-01-07','2013-01-18'))
        sell = result.trades.loc[result.trades.side.eq('SELL')].iloc[0]
        self.assertEqual(sell.date, pd.Timestamp('2013-01-21'))
        self.assertFalse(sell.no_quote_exit)

    def test_missing_entry_never_opens_a_lot(self):
        result, _, _ = self.case([pd.Timestamp('2013-01-03')])
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.orders.iloc[0].status, 'cancelled_missing_entry')

    def test_cutoff_does_not_force_early_timeout(self):
        missing = pd.bdate_range('2013-01-07','2013-02-05')
        result, _, _ = self.case(missing, end='2013-01-11')
        self.assertEqual(len(result.open_lots), 1)
        self.assertFalse(result.trades.side.eq('SELL').any())
        longer, _, _ = self.case(missing)
        pd.testing.assert_frame_equal(result.daily, longer.daily.loc[longer.daily.date.le('2013-01-11')].reset_index(drop=True))

    def test_weekend_maturity_counts_ten_later_sessions(self):
        result, _, _ = self.case(pd.bdate_range('2013-01-07','2013-02-05'), holding_days=3)
        sell = result.trades.loc[result.trades.side.eq('SELL')].iloc[0]
        self.assertEqual(sell.date, pd.Timestamp('2013-01-18'))

    def test_zero_recovery_cash_reconciles(self):
        missing = pd.bdate_range('2013-01-07','2013-02-05')
        normal, _, _ = self.case(missing)
        stress, _, _ = self.case(missing, recovery=0)
        sell = stress.trades.loc[stress.trades.side.eq('SELL')].iloc[0]
        self.assertEqual(sell.notional, 0)
        self.assertEqual(sell.fee, 0)
        proceeds = normal.trades.loc[normal.trades.side.eq('SELL'), 'cash_change'].sum()
        self.assertAlmostEqual(normal.daily.iloc[-1].nav - stress.daily.iloc[-1].nav, proceeds)


class DevelopmentCacheTests(unittest.TestCase):
    def test_grouped_cache_matches_original_price_financial_and_rank_functions(self):
        import tempfile
        from pathlib import Path
        import duckdb
        from feature_engineering import (build_full_development_cache, load_development_inputs,
            price_features, apply_membership, make_labels, build_financial_panel)
        scratch = Path(__file__).parent / 'outputs/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            source, cache = Path(directory)/'source.duckdb', Path(directory)/'cache.duckdb'
            dates = pd.bdate_range('2008-01-01','2011-12-30')
            prices = pd.concat([pd.DataFrame({'date':dates,'symbol':s,
                'adj_close':100 + np.arange(len(dates))*.02 + np.sin(np.arange(len(dates))*.13+i)}) for i,s in enumerate('ABCDEF')],ignore_index=True)
            membership = pd.DataFrame([{'symbol':s,'quarter_start':q.start_time,'quarter_end':q.end_time.normalize(),
                'selection_date':q.start_time-pd.Timedelta(days=1),'universe_rank':i+1}
                for q in pd.period_range('2008Q1','2011Q4',freq='Q') for i,s in enumerate('ABCDEF')])
            universe = pd.DataFrame({'symbol':list('ABCDEF'),'sector':['large']*5+['small']})
            fundamentals = pd.DataFrame([{'symbol':s,'fiscal_date':pd.Timestamp('2009-12-31'),
                'filing_date':pd.Timestamp('2010-02-01'),'available_date':pd.Timestamp('2010-02-02'),
                'period':'quarter','total_assets':100.,'revenue':20.,'net_income':float(i+1),
                'operating_income':float(i+2),'total_debt':float(i+10),'cash_and_equivalents':float(i+3),
                'free_cash_flow':2.,'operating_cash_flow':3.} for i,s in enumerate('ABCDEF')])
            tables = {'prices':prices,'calendar':pd.DataFrame({'date':dates}),'universe':universe,
                'universe_membership':membership,'fundamentals':fundamentals,
                'benchmark':pd.DataFrame({'date':dates,'symbol':'^SP500TR','adj_close':100.})}
            with duckdb.connect(str(source)) as con:
                for name, frame in tables.items():
                    con.register('frame',frame)
                    con.execute(f'CREATE TABLE {name} AS SELECT * FROM frame')
            build_full_development_cache(source,cache,lookback_start='2008-01-01',end='2011-12-30',group_size=2)
            prices, calendar, members, _, _, _ = load_development_inputs(source,end='2011-12-30')
            expected = make_labels(apply_membership(price_features(prices,calendar),members,calendar),prices,calendar)
            expected = expected.loc[expected.eligible & expected.date.ge('2010-01-01')].reset_index(drop=True)
            financial = build_financial_panel(expected,source,end='2011-12-30')
            with duckdb.connect(str(cache),read_only=True) as con:
                for name, reference in [('A',expected),('D',financial)]:
                    actual = con.execute(f'SELECT * FROM {name} ORDER BY date,symbol').fetchdf()
                    pd.testing.assert_frame_equal(actual,reference[actual.columns].reset_index(drop=True),check_dtype=False)


class TargetAndScreenTests(unittest.TestCase):
    def test_target_clip_uses_only_completed_training_rows(self):
        from feature_engineering import annual_split, clip_training_target
        dates = pd.bdate_range('2010-01-01',periods=100)
        train = pd.DataFrame({'date':dates,'symbol':'A','eligible':True,
                              'target':np.arange(100,dtype=float),'label_end':dates+pd.Timedelta(days=90)})
        later = pd.DataFrame({'date':pd.to_datetime(['2012-12-20','2013-01-02','2014-01-02']),
            'symbol':'A','eligible':True,'target':[1e8,1e9,1e10],
            'label_end':pd.to_datetime(['2013-03-20','2013-04-02','2014-04-02'])})
        panel = pd.concat([train,later],ignore_index=True)
        original = panel.copy(deep=True)
        fit, _ = annual_split(panel,2013)
        clipped, audit = clip_training_target(fit.target)
        self.assertAlmostEqual(audit['target_clip_lower'],.99)
        self.assertAlmostEqual(audit['target_clip_upper'],98.01)
        self.assertEqual(audit['target_rows_clipped'],2)
        self.assertEqual(clipped.iloc[0],.99)
        pd.testing.assert_frame_equal(panel,original)
        panel.loc[panel.index>=100,'target']=-1e15
        changed, _ = annual_split(panel,2013)
        _, changed_audit = clip_training_target(changed.target)
        self.assertEqual(audit,changed_audit)

    def test_volatility_screen_uses_only_each_signal_date(self):
        from feature_engineering import signal_volatility_screen
        frame = pd.DataFrame({'date':pd.to_datetime(['2013-01-02']*5+['2013-01-03']*5),
            'symbol':list('ABCDE')*2,'volatility_63':[1.,2.,3.,4.,5.,10.,20.,30.,40.,50.],
            'target':np.arange(10,dtype=float)})
        screened = signal_volatility_screen(frame)
        self.assertAlmostEqual(screened.volatility_threshold.iloc[0],4.2)
        self.assertEqual(screened.passes_volatility_screen.iloc[:5].tolist(),[True]*4+[False])
        frame.loc[frame.date.eq('2013-01-03'),'volatility_63']=1e9
        frame['target']=-1e20
        changed = signal_volatility_screen(frame)
        pd.testing.assert_series_equal(screened.volatility_threshold.iloc[:5],changed.volatility_threshold.iloc[:5])
        pd.testing.assert_series_equal(screened.passes_volatility_screen.iloc[:5],changed.passes_volatility_screen.iloc[:5])
        same_day = signal_volatility_screen(frame.iloc[:5])
        pd.testing.assert_series_equal(screened.volatility_threshold.iloc[:5],same_day.volatility_threshold)

    def test_forecast_deciles_keep_raw_outcomes_and_rank_before_scoring(self):
        from feature_engineering import forecast_decile_report
        frame = pd.DataFrame({'date':pd.to_datetime(['2013-01-02']*20),
            'symbol':[f'S{i:02}' for i in range(20)],'prediction':np.arange(20),
            'target':np.arange(20,dtype=float),'volatility_63':.2})
        frame.loc[19,'target']=1000.0
        table,spread=forecast_decile_report(frame)
        self.assertEqual(table.loc[table.decile.eq(10),'mean_realized_return'].iloc[0],509.0)
        self.assertEqual(spread.top_minus_bottom.iloc[0],508.5)
        frame.loc[0,'target']=np.nan
        changed,_=forecast_decile_report(frame)
        self.assertEqual(changed.loc[changed.decile.eq(10),'mean_realized_return'].iloc[0],509.0)


if __name__ == "__main__":
    unittest.main()
