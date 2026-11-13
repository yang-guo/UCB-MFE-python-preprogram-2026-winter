"""FF5 construction tests: hand-calculated sorts, timing, exclusions."""

import unittest

import numpy as np
import pandas as pd

from backtesting import FamaFrench5, MarketData


def fundamental(symbol, fiscal, filed, *, equity, assets, operating, interest,
                shares=100.0):
    return dict(
        symbol=symbol, fiscal_date=pd.Timestamp(fiscal), period="quarter",
        filing_date=pd.Timestamp(filed), revenue=100.0, net_income=10.0,
        operating_income=operating, total_assets=assets, total_debt=50.0,
        cash_and_equivalents=20.0, free_cash_flow=5.0, operating_cash_flow=15.0,
        total_equity=equity, interest_expense=interest,
        shares_outstanding=shares, reported_currency="USD",
    )


def synthetic():
    """Nine names, five years of monthly prices and quarterly filings."""
    calendar = pd.bdate_range("2017-01-03", "2021-12-31")
    bases = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0,
             "E": 50.0, "F": 60.0, "FS": 70.0, "NEG": 80.0, "OUT": 90.0}
    prices = pd.DataFrame([
        dict(date=date, symbol=symbol,
             close=base * (1.0 + 0.0002 * i), adj_close=base * (1.0 + 0.0002 * i),
             volume=1000)
        for symbol, base in bases.items()
        for i, date in enumerate(calendar)
    ])
    benchmark = pd.DataFrame({
        "date": calendar,
        "adj_close": 100.0 * (1.0 + 0.0002 * np.arange(len(calendar))),
    })
    sectors = {"A": "Technology", "B": "Technology", "C": "Technology",
               "D": "Industrials", "E": "Industrials", "F": "Industrials",
               "FS": "Financial Services", "NEG": "Industrials", "OUT": "Industrials"}
    universe = pd.DataFrame({"symbol": list(sectors), "sector": list(sectors.values())})

    rows = []
    quarters = pd.date_range("2016-03-31", "2021-12-31", freq="QE")
    for symbol, base in bases.items():
        for i, fiscal in enumerate(quarters):
            year = fiscal.year
            equity = -10.0 if symbol == "NEG" else 100.0 + 10.0 * (i % 4)
            assets = 500.0 * (1.10 ** (year - 2016)) + 10.0 * (i % 4)
            rows.append(fundamental(
                symbol, fiscal, fiscal + pd.Timedelta(days=30),
                equity=equity, assets=assets, operating=20.0 + (i % 4),
                interest=2.0))
    fundamentals = pd.DataFrame(rows)

    membership_rows = []
    for quarter in pd.date_range("2017-03-31", "2021-12-31", freq="QE"):
        start = quarter.to_period("Q").start_time
        for symbol in bases:
            # OUT is a member only in 2020-Q1; everything else is always a member.
            if symbol == "OUT" and start != pd.Timestamp("2020-01-01"):
                continue
            membership_rows.append(dict(
                symbol=symbol, quarter_start=start,
                quarter_end=quarter, selection_date=start - pd.Timedelta(days=1),
                universe_rank=1,
            ))
    membership = pd.DataFrame(membership_rows)
    return prices, fundamentals, universe, benchmark, membership


def monthly_frames():
    """Four month-ends with a known adjusted-close path."""
    calendar = pd.bdate_range("2020-01-01", "2020-04-30")
    month_ends = pd.DatetimeIndex(
        pd.Series(calendar).groupby(calendar.to_period("M")).max().to_numpy())
    symbols = list("ABCDEF")
    values = [100.0, 110.0, 121.0, 133.1]
    prices = pd.DataFrame([
        dict(date=date, symbol=symbol, close=value, adj_close=value, volume=1000)
        for symbol in symbols
        for date, value in zip(month_ends, values)
    ])
    benchmark = pd.DataFrame({"date": calendar, "adj_close": 100.0})
    universe = pd.DataFrame({"symbol": symbols, "sector": ["Technology"] * 6})
    membership = pd.DataFrame([
        dict(symbol=symbol, quarter_start=pd.Timestamp("2020-01-01"),
             quarter_end=pd.Timestamp("2020-03-31"),
             selection_date=pd.Timestamp("2019-12-31"), universe_rank=1)
        for symbol in symbols
    ] + [
        dict(symbol=symbol, quarter_start=pd.Timestamp("2020-04-01"),
             quarter_end=pd.Timestamp("2020-06-30"),
             selection_date=pd.Timestamp("2020-03-31"), universe_rank=1)
        for symbol in symbols
    ])
    fundamentals = pd.DataFrame([
        fundamental(symbol, "2019-12-31", "2020-01-15", equity=100.0,
                    assets=500.0, operating=20.0, interest=2.0)
        for symbol in symbols
    ])
    return prices, fundamentals, universe, benchmark, membership


def hand_cross_section():
    """Six names with one name in every size x characteristic cell."""
    return pd.DataFrame({
        "date": pd.Timestamp("2020-01-31"),
        "symbol": list("ABCDEF"),
        "ME": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "BE": [100.0] * 6,
        "BM": [0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
        "OP": [0.01, 0.03, 0.05, 0.02, 0.04, 0.06],
        "INV": [-0.03, 0.0, 0.03, -0.02, 0.01, 0.04],
        "FWD_RET": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "is_member": [True] * 6,
        "sector": ["Technology"] * 6,
    })


class FamaFrenchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = synthetic()
        cls.data = MarketData.from_frames(*cls.frames)

    def test_hand_calculated_two_by_three_factors(self):
        model = FamaFrench5(lag_months=0, weighting="equal", min_names=6)
        factors, portfolios, diagnostics, cross = model._factor_row(hand_cross_section())
        self.assertAlmostEqual(factors["MKT"], 0.035)
        self.assertAlmostEqual(factors["SMB"], -0.03)
        self.assertAlmostEqual(factors["HML"], 0.02)
        self.assertAlmostEqual(factors["RMW"], 0.02)
        self.assertAlmostEqual(factors["CMA"], -0.02)
        self.assertEqual(diagnostics["eligible"], 6)
        self.assertEqual(diagnostics["market_n"], 6)

        indexed = cross.set_index("symbol")
        self.assertEqual(indexed.loc["A", "size_group"], "S")
        self.assertEqual(indexed.loc["A", "bm_group"], "L")
        self.assertEqual(indexed.loc["A", "op_group"], "W")
        self.assertEqual(indexed.loc["A", "inv_group"], "C")
        self.assertEqual(indexed.loc["F", "size_group"], "B")
        self.assertEqual(indexed.loc["F", "bm_group"], "H")
        self.assertEqual(indexed.loc["F", "op_group"], "R")
        self.assertEqual(indexed.loc["F", "inv_group"], "A")

        lookup = {(row["characteristic"], row["portfolio"]): row["return"]
                  for row in portfolios}
        self.assertAlmostEqual(lookup[("BM", "S/L")], 0.01)
        self.assertAlmostEqual(lookup[("BM", "S/N")], 0.02)
        self.assertAlmostEqual(lookup[("BM", "S/H")], 0.03)
        self.assertAlmostEqual(lookup[("BM", "B/L")], 0.04)
        self.assertAlmostEqual(lookup[("BM", "B/N")], 0.05)
        self.assertAlmostEqual(lookup[("BM", "B/H")], 0.06)

    def test_build_returns_all_five_factors(self):
        model = FamaFrench5(lag_months=6, weighting="equal")
        result = model.build(self.data, start="2019-01-01", end="2021-12-31")
        self.assertEqual(list(result.factor_returns.columns),
                         ["MKT", "SMB", "HML", "RMW", "CMA"])
        self.assertGreater(len(result.factor_returns), 20)
        self.assertTrue(result.factor_returns.loc[:"2021-06-30"].notna().any().any())
        for column in ("date", "symbol", "ME", "BE", "BM", "OP", "INV",
                       "FWD_RET", "eligible", "size_group"):
            self.assertIn(column, result.characteristics.columns)
        self.assertEqual(set(result.portfolios.characteristic), {"BM", "OP", "INV"})

    def test_exclusions_remove_financials_and_negative_book_equity(self):
        model = FamaFrench5(lag_months=6, weighting="equal")
        result = model.build(self.data, start="2020-01-01", end="2020-12-31")
        date = pd.Timestamp("2020-06-30")
        cross = result.characteristics.loc[result.characteristics.date.eq(date)].set_index("symbol")
        self.assertFalse(bool(cross.loc["FS", "eligible"]))
        self.assertFalse(bool(cross.loc["NEG", "eligible"]))
        self.assertFalse(bool(cross.loc["OUT", "eligible"]))  # not a 2020-Q3 member
        self.assertTrue(bool(cross.loc["A", "eligible"]))

    def test_investment_uses_year_over_year_assets(self):
        model = FamaFrench5(lag_months=6, weighting="equal")
        cross = model.exposures(self.data, "2020-06-30").set_index("symbol")
        self.assertAlmostEqual(cross.loc["A", "INV"], 0.10, delta=0.005)

    def test_future_filing_does_not_change_prior_characteristics(self):
        model = FamaFrench5(lag_months=6, weighting="equal")
        first = model.build(self.data, start="2019-01-01", end="2020-12-31")
        cutoff = pd.Timestamp("2020-06-30")
        before = first.characteristics.loc[first.characteristics.date.le(cutoff)].reset_index(drop=True)

        prices, fundamentals, universe, benchmark, membership = self.frames
        future = fundamental(
            "A", "2020-06-30", "2020-12-31", equity=99_999.0, assets=99_999.0,
            operating=9_999.0, interest=0.0)
        changed = MarketData.from_frames(
            prices, pd.concat([fundamentals, pd.DataFrame([future])], ignore_index=True),
            universe, benchmark, membership)
        second = model.build(changed, start="2019-01-01", end="2020-12-31")
        after = second.characteristics.loc[second.characteristics.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_future_prices_do_not_change_prior_factor_returns(self):
        model = FamaFrench5(lag_months=6, weighting="equal")
        first = model.build(self.data, start="2019-01-01", end="2021-12-31")
        cutoff = pd.Timestamp("2020-06-30")

        prices, fundamentals, universe, benchmark, membership = self.frames
        changed_prices = prices.copy()
        mask = changed_prices.date.gt(cutoff)
        changed_prices.loc[mask, ["close", "adj_close"]] *= 7.0
        changed = MarketData.from_frames(
            changed_prices, fundamentals, universe, benchmark, membership)
        second = model.build(changed, start="2019-01-01", end="2021-12-31")

        before = first.factor_returns.loc[first.factor_returns.index < cutoff]
        after = second.factor_returns.loc[second.factor_returns.index < cutoff]
        pd.testing.assert_frame_equal(before, after)

    def test_forward_return_uses_next_month_close(self):
        data = MarketData.from_frames(*monthly_frames())
        calendar = pd.DatetimeIndex(data.calendar)
        month_ends = pd.DatetimeIndex(
            pd.Series(calendar).groupby(calendar.to_period("M")).max().to_numpy())
        chars = FamaFrench5()._characteristics(data, month_ends).set_index(
            ["date", "symbol"])
        self.assertAlmostEqual(chars.loc[(month_ends[0], "A"), "FWD_RET"], 0.10)
        self.assertAlmostEqual(chars.loc[(month_ends[1], "A"), "FWD_RET"], 0.10)
        self.assertAlmostEqual(chars.loc[(month_ends[2], "A"), "FWD_RET"], 0.10)

    def test_six_month_lag_uses_the_prior_vintage(self):
        prices, fundamentals, universe, benchmark, membership = synthetic()
        old = fundamental("A", "2017-12-31", "2018-02-15", equity=100.0,
                          assets=1000.0, operating=10.0, interest=1.0)
        new = fundamental("A", "2018-12-31", "2019-03-15", equity=200.0,
                          assets=2000.0, operating=20.0, interest=2.0)
        changed = fundamentals.loc[fundamentals.symbol.ne("A")]
        changed = pd.concat([changed, pd.DataFrame([old, new])], ignore_index=True)
        data = MarketData.from_frames(
            prices, changed, universe, benchmark, membership)
        model = FamaFrench5(lag_months=6)
        old_cross = model.exposures(data, "2019-06-28").set_index("symbol")
        new_cross = model.exposures(data, "2020-01-31").set_index("symbol")
        self.assertEqual(old_cross.loc["A", "BE"], 100.0)
        self.assertEqual(new_cross.loc["A", "BE"], 200.0)

    def test_missing_cell_makes_the_leg_nan_not_reweighted(self):
        cross = hand_cross_section()
        cross["BM"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        model = FamaFrench5(lag_months=0, weighting="equal", min_names=6)
        factors, _, _, grouped = model._factor_row(cross)
        self.assertTrue(grouped.loc[grouped.symbol.isin(["A", "B", "C"]),
                                    "bm_group"].eq("H").sum() == 0)
        self.assertTrue(np.isnan(factors["HML"]))
        self.assertTrue(np.isnan(factors["SMB"]))

    def test_incomplete_filing_does_not_shadow_prior_complete_values(self):
        prices, fundamentals, universe, benchmark, membership = synthetic()
        old = fundamental("A", "2017-12-31", "2018-02-15", equity=100.0,
                          assets=1000.0, operating=10.0, interest=1.0)
        stub = fundamental("A", "2018-12-31", "2019-03-15", equity=200.0,
                           assets=2000.0, operating=20.0, interest=2.0)
        for column in ("total_equity", "total_assets", "operating_income"):
            stub[column] = None
        changed = fundamentals.loc[fundamentals.symbol.ne("A")]
        changed = pd.concat([changed, pd.DataFrame([old, stub])], ignore_index=True)
        data = MarketData.from_frames(
            prices, changed, universe, benchmark, membership)
        cross = FamaFrench5(lag_months=6).exposures(
            data, "2020-01-31").set_index("symbol")
        self.assertEqual(cross.loc["A", "BE"], 100.0)

    def test_tied_breakpoints_do_not_swallow_the_cross_section(self):
        cross = hand_cross_section()
        cross["INV"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        model = FamaFrench5(lag_months=0, weighting="equal", min_names=6)
        factors, _, diagnostics, grouped = model._factor_row(cross)
        self.assertEqual(diagnostics["eligible"], 6)
        self.assertTrue(grouped.inv_group.eq("N").all())
        self.assertTrue(np.isnan(factors["CMA"]))

    def test_min_names_and_min_me_are_reported(self):
        cross = hand_cross_section().iloc[:5].copy()
        model = FamaFrench5(lag_months=0, weighting="equal", min_names=6)
        factors, portfolios, diagnostics, _ = model._factor_row(cross)
        self.assertTrue(all(np.isnan(value) for value in factors.values()))
        self.assertEqual(portfolios, [])
        self.assertEqual(diagnostics["eligible"], 5)
        self.assertEqual(diagnostics["market_n"], 0)

        screened = FamaFrench5(lag_months=0, weighting="equal", min_me=10.0)
        factors, _, diagnostics, _ = screened._factor_row(hand_cross_section())
        self.assertTrue(all(np.isnan(value) for value in factors.values()))
        self.assertEqual(diagnostics["excluded_me"], 6)

    def test_final_factor_row_is_dropped(self):
        model = FamaFrench5(lag_months=6, weighting="equal")
        result = model.build(self.data)
        self.assertFalse(result.factor_returns.iloc[-1].isna().all())

    def test_missing_required_financial_column_is_rejected(self):
        prices, fundamentals, universe, benchmark, membership = self.frames
        changed = MarketData.from_frames(
            prices, fundamentals.drop(columns=["total_equity"]),
            universe, benchmark, membership)
        model = FamaFrench5()
        with self.assertRaisesRegex(ValueError, "total_equity"):
            model.build(changed, start="2020-01-01", end="2020-06-30")


if __name__ == "__main__":
    unittest.main()
