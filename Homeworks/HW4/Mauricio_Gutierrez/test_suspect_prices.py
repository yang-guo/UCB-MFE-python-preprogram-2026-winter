"""Two small past-only jump-rule tests. No backtest or database is used."""
import unittest
import numpy as np
import pandas as pd
from feature_engineering import suspect_price_flags


class SuspectPriceTests(unittest.TestCase):
    def fixture(self, multiple):
        calendar = pd.bdate_range('2010-01-01',periods=330)
        prices = pd.DataFrame({'date':calendar,'symbol':'X','adj_close':100.0})
        prices.loc[20:,'adj_close'] *= multiple
        rows = prices[['date','symbol']].copy()
        rows['label_end'] = rows.date + pd.Timedelta(days=90)
        rows.loc[rows.label_end.gt(calendar[-1]),'label_end'] = pd.NaT
        return prices,calendar,rows

    def test_normal_twenty_percent_move_is_kept(self):
        prices,calendar,rows = self.fixture(1.2)
        flags = suspect_price_flags(prices,calendar,rows)
        self.assertFalse(flags[['suspect_lookback','suspect_label']].any().any())

    def test_jump_blocks_next_252_sessions_and_contaminated_training_rows(self):
        prices,calendar,rows = self.fixture(6.0)
        flags = suspect_price_flags(prices,calendar,rows)
        self.assertFalse(flags.loc[:19,'suspect_lookback'].any())
        self.assertTrue(flags.loc[20:272,'suspect_lookback'].all())
        self.assertFalse(flags.loc[273:,'suspect_lookback'].any())
        # A pre-jump signal remains a candidate but its future label is unusable.
        self.assertTrue(flags.loc[0,'suspect_label'])
        dropped = flags.suspect_lookback | flags.suspect_label
        self.assertTrue(dropped.loc[:272].all())
        self.assertFalse(dropped.loc[273:].any())
        prefix = suspect_price_flags(prices.iloc[:20],calendar[:20],rows.iloc[:20].assign(label_end=pd.NaT))
        np.testing.assert_array_equal(prefix.suspect_lookback,flags.loc[:19,'suspect_lookback'])


if __name__ == '__main__':
    unittest.main()
