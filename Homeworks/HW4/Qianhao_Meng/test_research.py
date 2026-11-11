"""Timing tests for the data and model steps that feed the backtester."""

import unittest
import warnings

import numpy as np
import pandas as pd

import data
import model


NO_EVENTS = pd.DataFrame({"symbol": [], "date": pd.to_datetime([]), "type": [], "factor": []})


def event_table(*rows):
    events = pd.DataFrame(rows, columns=["symbol", "date", "type", "factor"])
    return events.assign(date=pd.to_datetime(events.date))


def price_table(**closes):
    dates = pd.bdate_range("2024-01-01", periods=6)
    frames = [pd.DataFrame({"symbol": symbol, "date": dates, "close": values, "adj_close": values, "volume": 1000})
              for symbol, values in closes.items()]
    return pd.concat(frames, ignore_index=True).astype({"close": float, "adj_close": float}), dates


def synthetic_panel(dates, symbols=40, seed=1):
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame([(date, f"S{i:02d}") for date in dates for i in range(symbols)], columns=["date", "symbol"])
    panel["entry_date"] = panel.date + pd.Timedelta(days=3)
    panel["label_end"] = panel.entry_date + pd.Timedelta(days=90)
    for name in model.COMPONENTS:
        panel[name] = rng.normal(size=len(panel))
    panel["target"] = 0.1 * panel.momentum_252_21 + rng.normal(0, 0.05, len(panel))
    return panel


class SignalTimingTests(unittest.TestCase):
    def test_monthly_signal_is_the_close_before_a_new_month(self):
        sessions = pd.to_datetime(["2024-05-30", "2024-05-31", "2024-06-03", "2024-06-04"])
        self.assertEqual(data.monthly_signals(sessions).tolist(), [pd.Timestamp("2024-05-31")])

    def test_quarantine_drops_quotes_from_the_first_break_only(self):
        prices, dates = price_table(A=[10, 11, 50, 12, 12, 13], B=[10, 20, 39, 20, 6, 7])
        kept, breaks = data.quarantine_breaks(prices, NO_EVENTS)
        # A jumps 11 to 50, more than four times. B halves and doubles but never breaks the limit.
        self.assertEqual(kept.loc[kept.symbol.eq("A"), "adj_close"].tolist(), [10, 11])
        self.assertEqual(len(kept.loc[kept.symbol.eq("B")]), 6)
        expected = {"symbol": "A", "break_date": dates[2], "vendor_last_date": dates[5], "vendor_last_close": 13.0}
        self.assertEqual(breaks.to_dict("records"), [expected])
        # Later prices cannot change which earlier quotes are kept.
        prefix, _ = data.quarantine_breaks(prices.loc[prices.date.le(dates[1])], NO_EVENTS)
        pd.testing.assert_frame_equal(prefix, kept.loc[kept.date.le(dates[1])])

    def test_reviewed_events_change_how_a_jump_is_treated(self):
        prices, dates = price_table(A=[100, 110, 22, 23, 24, 25])
        # Confirmed as a real collapse: every quote is kept, so the loss reaches features, targets, and NAV.
        kept, breaks = data.quarantine_breaks(prices, event_table(("A", dates[2], "real", np.nan)))
        self.assertEqual(len(kept), 6)
        self.assertTrue(breaks.empty)
        # Reviewed but unresolved: the row documents the finding and the stock stays quarantined.
        _, breaks = data.quarantine_breaks(prices, event_table(("A", dates[2], "unresolved", np.nan)))
        self.assertEqual(breaks.break_date.tolist(), [dates[2]])
        # A documented 5-for-1 split rescales the earlier prices, which leaves an ordinary 0% session.
        fixed = data.apply_splits(prices, event_table(("A", dates[2], "split", 0.2)))
        self.assertEqual(fixed.adj_close.tolist(), [20, 22, 22, 23, 24, 25])
        self.assertEqual(fixed.volume.tolist(), [5000, 5000, 1000, 1000, 1000, 1000])
        self.assertTrue(data.quarantine_breaks(fixed, NO_EVENTS)[1].empty)
        # A factor that does not explain the vendor's jump is rejected.
        with self.assertRaises(ValueError):
            data.apply_splits(prices, event_table(("A", dates[2], "split", 0.5)))

    def test_ranks_use_only_the_same_date_and_point_the_rewarded_way(self):
        dates = pd.to_datetime(["2020-01-31", "2020-02-28"])
        panel = synthetic_panel(dates, symbols=5)
        panel.loc[0, "operating_roa"] = np.nan
        ranked = model.add_ranks(panel)
        first = ranked.loc[ranked.date.eq(dates[0])]
        # Higher momentum ranks higher, and lower volatility ranks higher.
        self.assertEqual(first.rank_momentum_252_21.idxmax(), first.momentum_252_21.idxmax())
        self.assertEqual(first.rank_volatility_63.idxmax(), first.volatility_63.idxmin())
        # A missing component is neutral, and the composite is the plain average of the four ranks.
        row = ranked.loc[0]
        self.assertEqual(row.rank_operating_roa, 0.5)
        expected = np.mean([row.rank_momentum_252_21, row.rank_drawdown_252, row.rank_volatility_63, 0.5])
        self.assertAlmostEqual(model.composite_scores(ranked).score.iloc[0], expected)
        # Changing the later date leaves the earlier date's ranks untouched.
        changed = panel.copy()
        changed.loc[changed.date.eq(dates[1]), list(model.COMPONENTS)] *= -3
        pd.testing.assert_frame_equal(model.add_ranks(changed).loc[lambda x: x.date.eq(dates[0]), model.RANKS],
                                      first[model.RANKS])

    def test_fits_ignore_outcomes_that_end_on_or_after_the_fit_date(self):
        dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29"])
        ranked = model.add_ranks(synthetic_panel(dates))
        fit_date = dates[-1]
        scores, audit = model.walk_forward_scores(ranked, 100.0, [fit_date])
        changed = ranked.copy()
        changed.loc[changed.label_end.ge(fit_date), "target"] = 99.0
        rescored, _ = model.walk_forward_scores(changed, 100.0, [fit_date])
        # Only January's outcomes ended before the May fit, so the other months cannot matter.
        self.assertEqual(audit.training_rows.iloc[0], 40)
        self.assertLess(audit.last_label_end.iloc[0], fit_date)
        np.testing.assert_allclose(scores.score, rescored.score)
        self.assertEqual(scores.date.unique().tolist(), [fit_date])


@unittest.skipUnless(data.DATABASE.is_file(), "needs the bCourses database")
class DatabaseTimingTests(unittest.TestCase):
    SYMBOLS = ["AAPL", "MSFT", "LCI", "AAM"]

    def load(self, end):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return data.load_research_data(data.DATABASE, end, "2013-01-01", self.SYMBOLS)

    def test_later_data_cannot_change_earlier_signals(self):
        early_panel = self.load("2013-12-31")[0]
        late_panel = self.load("2015-12-31")[0]
        late_prefix = late_panel.loc[late_panel.date.le(early_panel.date.max())].reset_index(drop=True)
        keys = ["date", "symbol", *data.FEATURES]
        pd.testing.assert_frame_equal(early_panel[keys], late_prefix[keys])
        # A target already known at the first cutoff is the same after two more years of data.
        known = early_panel.target.notna()
        self.assertTrue(early_panel.label_end.dropna().le("2013-12-31").all())
        np.testing.assert_allclose(early_panel.target[known], late_prefix.target[known])

    def test_lecture_worked_examples_are_reproduced(self):
        aapl = self.load("2015-12-31")[0].loc[lambda x: x.symbol.eq("AAPL")].set_index("date")
        # Lecture 7.2 section 4: the 2012-08-31 signal enters on September 4 and exits on December 3.
        self.assertEqual(aapl.loc["2012-08-31", "entry_date"], pd.Timestamp("2012-09-04"))
        self.assertEqual(aapl.loc["2012-08-31", "label_end"], pd.Timestamp("2012-12-03"))
        # Lecture 7.2 section 2: the December quarter, filed January 28, replaces 20.05% with 22.70%.
        self.assertAlmostEqual(aapl.loc["2013-12-31", "net_margin"], 0.2005, places=4)
        self.assertAlmostEqual(aapl.loc["2014-01-31", "net_margin"], 0.2270, places=4)

    def test_broken_price_series_stop_at_the_break(self):
        panel, _, quotes, breaks = self.load("2015-12-31")
        # LCI's adjusted close jumps from 28.09 to 474.08 on 2013-12-13, listed as unresolved in price_events.csv.
        self.assertEqual(quotes.LCI.last_valid_index(), pd.Timestamp("2013-12-12"))
        self.assertEqual(breaks.set_index("symbol").break_date["LCI"], pd.Timestamp("2013-12-13"))
        self.assertLess(panel.loc[panel.symbol.eq("LCI"), "date"].max(), pd.Timestamp("2013-12-13"))
        self.assertTrue(panel.loc[panel.symbol.eq("LCI") & panel.entry_date.gt("2013-09-14"), "target"].isna().all())
        # AAM first prints 0.33 beside 34,300 in April 2010, before the backtest window.
        self.assertNotIn("AAM", quotes.columns[quotes.notna().any()])


if __name__ == "__main__":
    unittest.main()
