"""Checks that future outcomes and historical eligibility cannot leak into fits."""

import unittest
from pathlib import Path
import tempfile

import duckdb
import pandas as pd

from corporate_actions import load_actions
from research import SCREEN, fit_and_score, screen_rows


class TimingTests(unittest.TestCase):
    def panel(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2010-01-01", "2010-02-01", "2012-11-01", "2012-12-31", "2017-01-01"]),
            "label_end": pd.to_datetime(["2010-04-05", "2010-05-05", "2013-02-01", "2013-04-05", "2017-04-05"]),
            "target": [0.05, 0.1, 0.8, -0.3, 1000.0], "symbol": "A",
            "eligible": True, "universe_rank": 20, "close": 50.0,
            "volatility_63": 0.2, "max_abs_daily_63": 0.03,
            "complete_returns_252": True, "momentum_21": [0.01, 0.02, 0.03, 0.04, 100.0],
        })

    def test_incomplete_and_future_targets_cannot_change_fit_or_predictions(self):
        spec = {"screen": SCREEN, "features": ["momentum_21"], "alpha": 100}
        signals = pd.DatetimeIndex(["2012-12-31"])
        panel = self.panel()
        expected, audit = fit_and_score(panel, signals, spec)
        panel.loc[2:, "target"] = [-1e9, 1e9, 1e12]
        actual, other_audit = fit_and_score(panel, signals, spec)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(audit, other_audit)
        self.assertEqual(audit["training_rows"], 2)
        self.assertEqual(audit["latest_training_outcome"], "2010-05-05")

    def test_dated_membership_and_observed_history_are_required(self):
        panel = self.panel()
        panel.loc[0, "eligible"] = False  # Not a dated member at this entry.
        panel.loc[1, "complete_returns_252"] = False
        panel.loc[2, "max_abs_daily_63"] = 0.50
        panel.loc[3, "universe_rank"] = 201
        self.assertEqual(screen_rows(panel, SCREEN).tolist(), [False, False, False, False, True])

    def test_merger_exchange_converts_adjusted_units_through_actual_shares(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "fixture.duckdb"
            with duckdb.connect(str(database)) as con:
                con.execute("CREATE TABLE prices(symbol VARCHAR, date DATE, close DOUBLE, adj_close DOUBLE)")
                con.execute("INSERT INTO prices VALUES ('XLNX','2022-02-11',200,100), ('AMD','2022-02-14',50,40), ('VMW','2023-11-21',100,80)")
            action = load_actions(database, [{"kind": "stock", "old_symbol": "XLNX",
                "new_symbol": "AMD", "effective_date": "2022-02-14", "share_exchange_ratio": 1.7234}])[0]
        old_units = 10
        successor_value = old_units * action["adjusted_unit_ratio"] * 40
        actual_share_value = old_units * (100 / 200) * 1.7234 * 50
        self.assertAlmostEqual(successor_value, actual_share_value)

    def test_cash_merger_amount_uses_old_adjustment_factor(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "fixture.duckdb"
            with duckdb.connect(str(database)) as con:
                con.execute("CREATE TABLE prices(symbol VARCHAR, date DATE, close DOUBLE, adj_close DOUBLE)")
                con.execute("INSERT INTO prices VALUES ('XLNX','2022-02-11',200,100), ('AMD','2022-02-14',50,40), ('VMW','2023-11-21',100,80)")
            action = load_actions(database, [{"kind": "cash", "old_symbol": "VMW",
                "new_symbol": "CASH:VMW", "effective_date": "2023-11-22", "cash_per_share": 142.5}])[0]
        self.assertAlmostEqual(action["cash_per_adjusted_unit"], 142.5 * 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
