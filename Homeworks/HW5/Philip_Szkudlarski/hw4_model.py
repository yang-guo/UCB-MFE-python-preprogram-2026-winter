"""Reusable HW4 scoring model, kept separate from the HTTP layer.

My HW4 final model ("pairs+fcf+revenue+momentum [trailing 126d]") ranks stocks
with a closed-form composite score. For each of the four buy signals ``s`` on
decision day ``t``:

    score_i(t) = sum_s w_s(t) * p_is(t)  +  #{s : stock i flagged by s and w_s(t) > 0}

* ``p_is`` is stock i's percentile among the stocks that signal ``s`` flagged on
  day t (pandas ``rank(pct=True)``, average ties).
* ``w_s`` are the trailing-126-day "strategy momentum" weights: the positive
  part of each signal's mean finished 90-day excess return, normalized to sum to 1.

Both pieces are *learned* from HW4 data and frozen in the exported artifact:

1. ``FlaggedPercentileTransformer`` (learned preprocessing) stores, per signal,
   the sorted cross-section of flagged signal strengths on the snapshot date and
   maps a raw strength to the same percentile HW4 computed.
2. ``SignalScoreRegressor`` (fitted model) stores the trailing weights ``w_s``.

They are chained in a scikit-learn ``Pipeline`` and saved with joblib. This
module must be importable when the artifact is loaded (joblib stores the class
path ``hw4_model.<Class>``), so it is copied into the Docker image with app.py.

Unflagged signals are represented as NaN. A stock that no positive-weight
signal flags is not a buy candidate in HW4 (score NaN there); the serving model
returns 0.0 for it, which is below every candidate's score (candidates score > 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Training feature order (must never change for a given artifact).
FEATURES = [
    "pairs_divergence_z",
    "fcf_quality_score",
    "revenue_growth_vs_industry",
    "momentum_percentile",
]


class FlaggedPercentileTransformer(TransformerMixin, BaseEstimator):
    """Map each raw signal strength to its percentile among the flagged names
    seen at fit time; NaN (signal not flagged) stays NaN.

    For a value present in the reference cross-section the result equals
    ``pandas.Series.rank(pct=True)`` exactly (average ties), which is what HW4
    used. A new value is ranked as if it joined the cross-section:
    ``(n_below + 1) / (n + 1)``. HW4 strengths are float32, so inputs are cast
    to float32 before comparison to reproduce HW4 ties exactly.
    """

    def fit(self, X, y=None):
        X = self._as_frame(X, fitting=True)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.reference_ = {
            c: np.sort(X[c].dropna().to_numpy(dtype=np.float32)) for c in X.columns
        }
        empty = [c for c, ref in self.reference_.items() if ref.size == 0]
        if empty:
            raise ValueError(f"No flagged reference values for: {empty}")
        return self

    def transform(self, X):
        check_is_fitted(self, "reference_")
        X = self._as_frame(X)
        out = np.full(X.shape, np.nan, dtype=np.float64)
        for j, col in enumerate(self.feature_names_in_):
            ref = self.reference_[col]
            n = ref.size
            v = X[col].to_numpy(dtype=np.float32)
            have = ~np.isnan(v)
            below = np.searchsorted(ref, v[have], side="left")
            at_or_below = np.searchsorted(ref, v[have], side="right")
            equal = at_or_below - below
            pct = np.where(
                equal > 0,
                (below + (equal + 1) / 2) / n,  # pandas average rank / n
                (below + 1) / (n + 1),  # unseen value joins the cross-section
            )
            out[have, j] = pct
        return pd.DataFrame(out, columns=list(self.feature_names_in_), index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)

    def _as_frame(self, X, fitting=False):
        """Require a DataFrame with the training columns in training order."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with named signal columns")
        if not fitting and list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"Expected columns {list(self.feature_names_in_)}, got {list(X.columns)}"
            )
        return X


class SignalScoreRegressor(RegressorMixin, BaseEstimator):
    """HW4 composite score from per-signal percentiles and frozen weights.

    ``weights`` maps signal name -> trailing-126-day weight on the snapshot
    date. ``expected_returns`` maps signal name -> HW4's mu-hat on that date:
    the mean 90-day excess return (vs the same-day universe average, as a
    fraction) of the stocks the signal flagged over the last 126 signal days
    whose outcomes had finished. HW4's weights are w_s = max(mu_s, 0) / sum.
    Both were estimated from finished HW4 outcomes (see export_hw4_model.py),
    so ``fit`` only validates and stores them.
    """

    def __init__(self, weights=None, expected_returns=None):
        self.weights = weights
        self.expected_returns = expected_returns

    def fit(self, X, y=None):
        cols = list(X.columns)
        for name, given in (
            ("weights", self.weights),
            ("expected_returns", self.expected_returns),
        ):
            if set(given) != set(cols):
                raise ValueError(f"{name} {sorted(given)} do not match columns {cols}")
        w = np.array([self.weights[c] for c in cols], dtype=np.float64)
        mu = np.array([self.expected_returns[c] for c in cols], dtype=np.float64)
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        if not np.all(np.isfinite(mu[w > 0])):
            raise ValueError("every positive-weight signal needs a finite expected return")
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.n_features_in_ = len(cols)
        self.coef_ = w
        self.expected_returns_ = mu
        return self

    def _live(self, X):
        check_is_fitted(self, "coef_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Percentile columns are not in training order")
        P = X.to_numpy(dtype=np.float64)
        return P, ~np.isnan(P) & (self.coef_ > 0)  # flagged by a positive-weight signal

    def predict(self, X):
        P, live = self._live(X)
        score = (np.where(live, P, 0.0) * self.coef_).sum(axis=1) + live.sum(axis=1)
        return np.where(live.any(axis=1), score, 0.0)  # 0.0 = not a buy candidate

    def expected_excess_return(self, X):
        """Weight-averaged mu-hat of the positive-weight signals flagging each row.

        A historical signal-level average (what stocks flagged by these signals
        earned over 90 days, in excess of the universe, recently), not a
        stock-specific forecast. NaN for non-candidates.
        """
        _, live = self._live(X)
        w = np.where(live, self.coef_, 0.0)
        total = w.sum(axis=1)
        num = (w * np.nan_to_num(self.expected_returns_)).sum(axis=1)
        return np.where(total > 0, num / np.where(total > 0, total, 1.0), np.nan)


def expected_excess_return(pipeline, X):
    """Run the fitted preprocessing, then the scorer's expected-return method."""
    return pipeline[-1].expected_excess_return(pipeline[:-1].transform(X))
