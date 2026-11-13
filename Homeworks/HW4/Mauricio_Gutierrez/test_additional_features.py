"""Known-answer and timing checks for monthly volatility, RSI, and P/S."""
import unittest

import numpy as np
import pandas as pd

from feature_engineering import (
    ADDITIONAL_PRICE_FEATURES, additional_price_features, monthly_volatility,
    price_to_sales, relative_strength_index,
)


class AdditionalFeatureTests(unittest.TestCase):
    def test_monthly_volatility_known_returns_and_gap(self):
        # Returns +10%, -10%, +10% have sample variance 0.04/3.
        prices = pd.Series([100.0, 110.0, 99.0, 108.9])
        result = monthly_volatility(prices, window=3)
        self.assertTrue(result.iloc[:3].isna().all())
        self.assertAlmostEqual(result.iloc[3], np.sqrt(0.28))
        prices.iloc[1] = np.nan
        self.assertTrue(monthly_volatility(prices, window=3).isna().all())

    def test_wilder_seed_and_recursive_update(self):
        prices = pd.Series([100.0, 102.0, 101.0, 104.0, 102.0])
        rsi = relative_strength_index(prices, window=3)
        self.assertTrue(rsi.iloc[:3].isna().all())
        self.assertAlmostEqual(rsi.iloc[3], 100 * 5 / 6)
        self.assertAlmostEqual(rsi.iloc[4], 100 * 10 / 18)

    def test_rsi_boundaries_flat_prices_and_gap_reset(self):
        for prices, expected in [([1, 2, 3, 4], 100.0), ([4, 3, 2, 1], 0.0), ([2, 2, 2, 2], 50.0)]:
            self.assertEqual(relative_strength_index(pd.Series(prices), window=3).iloc[-1], expected)
        prices = pd.Series([1, 2, 3, 4, np.nan, 5, 6, 7, 8])
        rsi = relative_strength_index(prices, window=3)
        self.assertTrue(rsi.iloc[4:8].isna().all())
        self.assertEqual(rsi.iloc[8], 100.0)

    def test_future_mutation_and_history_truncation(self):
        calendar = pd.bdate_range("2010-01-01", periods=120)
        prices = pd.DataFrame({"symbol": "A", "date": calendar,
                               "adj_close": 100 + np.arange(120) / 10 + np.sin(np.arange(120))})
        original = additional_price_features(prices, calendar)
        changed = prices.copy()
        changed.loc[changed.index > 80, "adj_close"] *= 10
        future_changed = additional_price_features(changed, calendar)
        truncated = additional_price_features(prices.iloc[:81], calendar[:81])
        pd.testing.assert_frame_equal(original.iloc[:81], future_changed.iloc[:81])
        pd.testing.assert_frame_equal(original.iloc[:81], truncated)
        self.assertEqual(original.columns[-2:].tolist(), ADDITIONAL_PRICE_FEATURES)

    def test_price_to_sales_requires_valid_values_and_currencies(self):
        cap = pd.Series([200.0, 200, 200, 200, 200, np.inf, -1])
        sales = pd.Series([100.0, 0, 100, 100, 100, 100, 100])
        cap_currency = pd.Series(["USD", "USD", None, "USD", "", "USD", "USD"])
        sales_currency = pd.Series(["usd", "USD", "USD", "EUR", "", "USD", "USD"])
        result = price_to_sales(cap, sales, cap_currency, sales_currency)
        self.assertEqual(result.iloc[0], 2.0)
        self.assertTrue(result.iloc[1:].isna().all())
        with self.assertRaises(ValueError):
            price_to_sales(cap, sales.iloc[:2], cap_currency, sales_currency)


if __name__ == "__main__":
    unittest.main()
