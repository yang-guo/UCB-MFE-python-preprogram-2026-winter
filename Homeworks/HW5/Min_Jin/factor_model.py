"""
The HW4 model that this service serves: the fixed-weight factor score.

FactorScore and FACTOR_SIGNS are copied with the same behavior from
Homeworks/HW4/Min_Jin/model.py

The model has no learned preprocessing as its inputs are already same-date
percentile ranks computed in HW4 features.add_peer_features.
"""

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# Five signals
# +1: higher is better.
# -1: lower is better.
FACTOR_SIGNS = {
    # profitable firms (Novy-Marx 2013; Fama-French RMW)
    "rank_operating_roa": +1,
    # firms that generate cash
    "rank_fcf_to_assets": +1,
    # close to the 1-year high (George & Hwang 2004)
    "rank_drawdown_252": +1,
    # 12-month momentum, skipping the last month (Jegadeesh & Titman 1993)
    "rank_momentum_252_21": +1,
    # low volatility (Ang et al. 2006; Frazzini & Pedersen 2014)
    "rank_volatility_63": -1,
}
FACTOR_FEATURES = list(FACTOR_SIGNS)


class FactorScore(RegressorMixin, BaseEstimator):
    """
    Score each stock by averaging its 5 factor ranks.

    How the score works:
    - Each rank is between 0 (worst that month) and 1 (best that month).
    - For volatility, lower is better, so its rank is flipped to 1 - rank.
    - The score is the average of the 5 ranks. A missing rank is left out of
      the average. A higher score means a more attractive stock.
    """

    def fit(self, X, y=None):
        # nothing to learn but kept so the code can call .fit().
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # keep each rank as is or flip it to 1 - rank
        signed = pd.DataFrame({c: X[c] if s > 0 else 1 - X[c] for c, s in FACTOR_SIGNS.items()})
        # average across the 5 columns, one score per stock
        # If all 5 are missing, use 0.5 (a middle score), so the stock is never a top pick by accident
        return signed.mean(axis=1).fillna(0.5).to_numpy()


def to_frame(rows, features=FACTOR_FEATURES):
    """
    Heler function to build the model input exactly as in HW4.
    return a DataFrame with the training column names/order.
    """
    return pd.DataFrame(rows, columns=features)
