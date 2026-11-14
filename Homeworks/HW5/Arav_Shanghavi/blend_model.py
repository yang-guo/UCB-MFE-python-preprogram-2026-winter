"""The two sklearn pieces behind my HW4 blend score.

HW4 turned six factors into percentile ranks among the stocks tradable on a
signal date, flipped volatility so calm stocks rank high, set a missing value
to 0.5, and averaged the six numbers. Nothing was regressed, so the only part
that has to be learned is the cross-section the percentiles are measured
against. That is what `CrossSectionPercentile.fit` stores.

The service and the export script both import this module by the plain name
`blend_model`, so the saved pipeline unpickles the same way in either place.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class CrossSectionPercentile(TransformerMixin, BaseEstimator):
    """Percentile of each value against a frozen set of reference stocks.

    Matches pandas rank(pct=True) with average ties: a value with `below`
    reference values under it and `same` values equal to it gets
    (below + (same + 1) / 2) / n, where n counts the non-missing reference
    values of that column only. For a stock that is itself in the reference
    this reproduces HW4 exactly. A brand-new value lands between its
    neighbours, and anything past the reference maximum is capped at 1.
    """

    def __init__(self, directions=None):
        self.directions = directions

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        signs = self.directions if self.directions is not None else (1,) * frame.shape[1]
        if len(signs) != frame.shape[1]:
            raise ValueError("need one direction per column")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.signs_ = np.asarray(signs, dtype=float)
        # one sorted array per column, missing values dropped column by column
        self.reference_ = [np.sort(col[~np.isnan(col)]) for col in frame.to_numpy(float).T]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("columns differ from the ones seen in fit")
            X = X.to_numpy(float)
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns")

        pct = np.empty_like(values)
        for j, ref in enumerate(self.reference_):
            col = values[:, j]
            below = np.searchsorted(ref, col, side="left")
            same = np.searchsorted(ref, col, side="right") - below
            p = np.minimum((below + (same + 1) / 2) / len(ref), 1.0)
            if self.signs_[j] < 0:
                p = 1 - p
            # a blank factor sits in the middle, as it did in HW4
            pct[:, j] = np.where(np.isnan(col), 0.5, p)
        return pct


class EqualWeightScore(RegressorMixin, BaseEstimator):
    """Plain average of the percentile columns. Fitting only records the width."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float).mean(axis=1)
