"""Additional causal market-state features beyond the core lecture examples."""

import numpy as np


def add_regime_features(panel, quotes, benchmark):
    p = panel.copy()
    market_return = benchmark.pct_change(fill_method=None)
    p["market_momentum_63"] = p.date.map(benchmark / benchmark.shift(63) - 1)
    p["market_volatility_63"] = p.date.map(
        market_return.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252)
    )
    p["market_trend_200"] = p.date.map(
        benchmark / benchmark.rolling(200, min_periods=200).mean() - 1
    )
    returns = quotes.pct_change(fill_method=None)
    avg = returns.rolling(126, min_periods=126).mean()
    mavg = market_return.rolling(126, min_periods=126).mean()
    covariance = returns.mul(market_return, axis=0).rolling(
        126, min_periods=126
    ).mean() - avg.mul(mavg, axis=0)
    beta = covariance.div(
        market_return.rolling(126, min_periods=126).var(ddof=0), axis=0
    )
    rows = quotes.index.get_indexer(p.date)
    cols = quotes.columns.get_indexer(p.symbol)
    p["market_beta_126"] = beta.to_numpy()[rows, cols]
    market_momentum = p.date.map(benchmark.shift(21) / benchmark.shift(126) - 1)
    p["residual_momentum_126_21"] = np.log1p(
        p.momentum_126_21
    ) - p.market_beta_126 * np.log1p(market_momentum)
    new = [
        "market_momentum_63",
        "market_volatility_63",
        "market_trend_200",
        "market_beta_126",
        "residual_momentum_126_21",
    ]
    assert not np.isinf(p[new].to_numpy()).any()
    return p
