"""Small independent arithmetic cases, calendar gaps and future perturbations."""

import numpy as np
import pandas as pd
import pytest

from price_path_signals import (
    relative_long_volatility,
    relative_volume,
    sample_and_rank,
    trend_consistency,
)


def prices(values):
    return pd.DataFrame(
        {"A": values}, index=pd.bdate_range("2009-01-01", periods=len(values))
    )


def test_consistency_distinguishes_equal_cumulative_returns():
    # Both paths gain 8%; one has three gains, the other two losses and a jump.
    q = prices([100.0, 102.0, 105.0, 108.0, 109.0])
    q["B"] = [100.0, 99.0, 98.0, 108.0, 109.0]
    result = trend_consistency(q, window=3, skip=1)
    np.testing.assert_allclose(result.iloc[-1], [1.0, -1 / 3])
    assert q.A.iloc[3] / q.A.iloc[0] == q.B.iloc[3] / q.B.iloc[0]


def test_flat_days_are_zero_but_missing_days_are_not():
    q = prices([100.0, 100.0, 101.0, 101.0])
    assert trend_consistency(q, window=3, skip=0).iloc[-1, 0] == pytest.approx(1 / 3)
    q.iloc[1, 0] = np.nan
    assert pd.isna(trend_consistency(q, window=3, skip=0).iloc[-1, 0])


def test_volume_windows_are_nonoverlapping_and_do_not_compare_company_size():
    v = prices([10.0, 10.0, 10.0, 20.0, 20.0])
    q = prices([100.0] * 5)
    result, audit = relative_volume(v, q, q, recent=2, prior=3)
    assert result.iloc[-1, 0] == pytest.approx(2.0)
    scaled, _ = relative_volume(v * 100, q, q, recent=2, prior=3)
    pd.testing.assert_frame_equal(result, scaled)
    assert not audit["adjustment_factor_seam"].any().any()


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan])
def test_invalid_volume_stays_missing(invalid):
    v = prices([10.0, 10.0, invalid, 20.0, 20.0])
    q = prices([100.0] * 5)
    result, _ = relative_volume(v, q, q, recent=2, prior=3)
    assert pd.isna(result.iloc[-1, 0])


def test_possible_split_seam_suppresses_feature_without_repairing_data():
    v = prices([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    raw = prices([100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    adj = prices([50.0] * 9)
    copy = v.copy()
    result, audit = relative_volume(v, raw, adj, recent=2, prior=3)
    assert audit["adjustment_factor_seam"].iloc[3, 0]
    assert pd.isna(result.iloc[7, 0])
    assert result.iloc[8, 0] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(v, copy)


def test_volatility_ratio_matches_hand_calculation():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.04, -0.03])
    q = prices(np.r_[100.0, 100 * np.cumprod(1 + returns)])
    result, _ = relative_long_volatility(q, short=3, long=6)
    assert result.iloc[-1, 0] == pytest.approx(
        np.std(returns[-3:], ddof=1) / np.std(returns, ddof=1)
    )
    constant, _ = relative_long_volatility(prices([100.0] * 7), short=3, long=6)
    assert pd.isna(constant.iloc[-1, 0])


def test_long_window_requires_history_and_flags_extreme_moves():
    q = prices([100.0, 101.0, 102.0, 300.0, 301.0, 302.0, 303.0])
    result, audit = relative_long_volatility(q, short=3, long=6)
    assert result.iloc[:-1, 0].isna().all()
    assert audit["large_move_in_long_window"].iloc[-1, 0]
    assert pd.isna(result.iloc[-1, 0])


def test_all_features_ignore_future_observations():
    q = prices(100 * np.exp(np.cumsum(np.sin(np.arange(60)) * 0.01)))
    v = prices(1000 + np.arange(60) * 10.0)
    q2, v2 = q.copy(), v.copy()
    q2.iloc[40:] *= 3
    v2.iloc[40:] *= 9
    pairs = [
        (trend_consistency(q, 10, 2), trend_consistency(q2, 10, 2)),
        (relative_volume(v, q, q, 3, 10)[0], relative_volume(v2, q2, q2, 3, 10)[0]),
        (relative_long_volatility(q, 5, 20)[0], relative_long_volatility(q2, 5, 20)[0]),
    ]
    for before, after in pairs:
        pd.testing.assert_frame_equal(before.iloc[:40], after.iloc[:40])


def test_peer_ranks_are_date_specific_and_keep_missing_candidates():
    matrix = prices([1.0, 10.0])
    matrix["B"], matrix["C"] = [2.0, np.nan], [0.0, 100.0]
    panel = pd.DataFrame(
        {
            "date": [matrix.index[0]] * 3 + [matrix.index[1]] * 2,
            "symbol": ["A", "B", "C", "A", "B"],
            "eligible": [True, True, False, True, True],
            "risk_screen_pass": True,
        },
        index=[6, 4, 9, 1, 2],
    )
    result = sample_and_rank(panel, {"test": matrix})
    assert result.index.equals(panel.index) and len(result) == len(panel)
    np.testing.assert_allclose(result.rank_test.iloc[[0, 1, 3]], [0.5, 1.0, 1.0])
    assert result.rank_test.iloc[[2, 4]].isna().all()
