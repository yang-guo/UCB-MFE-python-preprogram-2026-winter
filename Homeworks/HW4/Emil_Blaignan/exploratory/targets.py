"""Train-only target handling (HW4 pre-processing, step 2).

Genuine extreme 90-day returns (penny-stock jumps, warrants) cannot be told
apart from data errors by a revert test, and deleting them would be look-ahead.
Instead, limit their influence at fit time:

    w = TargetWinsorizer(0.001, 0.999).fit(train.target)   # train labels whose
    y = w.transform(train.target)                          # holding periods ended

Quantile bounds come from the training window only. Evaluation and the
backtester always use the realized, un-clipped returns. Alternatively train on
`target_rank` (cross-sectional percentile, built in preprocess.py).
"""
import numpy as np
import pandas as pd


class TargetWinsorizer:
    def __init__(self, lower=0.001, upper=0.999):
        self.lower, self.upper = lower, upper

    def fit(self, y):
        y = pd.Series(y).dropna()
        self.lo_, self.hi_ = y.quantile([self.lower, self.upper])
        return self

    def transform(self, y):
        return np.clip(y, self.lo_, self.hi_)
