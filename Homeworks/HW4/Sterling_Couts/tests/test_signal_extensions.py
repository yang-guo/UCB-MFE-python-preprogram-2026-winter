"""Independent rolling-beta arithmetic and future-price isolation."""

import numpy as np
import pandas as pd

from signal_extensions import add_regime_features


def test_regime_features_match_manual_beta_and_ignore_future_prices():
    rng = np.random.default_rng(51)
    dates = pd.bdate_range("2010-01-01", periods=500)
    mr = rng.normal(0.0002, 0.01, 500)
    sr = 1.3 * mr + rng.normal(0, 0.005, 500)
    benchmark = pd.Series(100 * np.cumprod(1 + mr), index=dates)
    quotes = pd.DataFrame({"T": 100 * np.cumprod(1 + sr)}, index=dates)
    selected = [dates[300], dates[400]]
    panel = pd.DataFrame({"date": selected, "symbol": "T"})
    panel["momentum_126_21"] = [
        quotes["T"].iloc[279] / quotes["T"].iloc[174] - 1,
        quotes["T"].iloc[379] / quotes["T"].iloc[274] - 1,
    ]
    observed = add_regime_features(panel, quotes, benchmark)
    r = quotes["T"].pct_change(fill_method=None).iloc[175:301]
    m = benchmark.pct_change(fill_method=None).iloc[175:301]
    expected = np.cov(r, m, ddof=0)[0, 1] / np.var(m, ddof=0)
    np.testing.assert_allclose(observed.market_beta_126.iloc[0], expected, atol=1e-12)
    future = quotes.copy()
    future.loc[future.index > dates[300], "T"] *= 10
    changed_benchmark = benchmark.copy()
    changed_benchmark.loc[changed_benchmark.index > dates[300]] *= 5
    changed = add_regime_features(panel, future, changed_benchmark)
    pd.testing.assert_series_equal(observed.iloc[0], changed.iloc[0])
