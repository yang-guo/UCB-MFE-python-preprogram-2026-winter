"""Three causal price/volume hypotheses, separate from the preserved pipeline.

Trend consistency is a simple signed-day proxy inspired by Da, Gurun and
Warachka (2014), not a replication of their strategy. Relative volume is
motivated by Lee and Swaminathan (2000), not an assumption that more volume
is bullish. All features use past/current observations and next-session entry.
AI-assisted implementation; hand examples test timing and economic arithmetic.
"""

import numpy as np
import pandas as pd


FEATURES = ("trend_consistency", "relative_volume", "relative_long_volatility")


def positive_finite(frame):
    """Keep the observed calendar; invalid cells become missing, never filled."""
    return frame.where(np.isfinite(frame) & frame.gt(0))


def trend_consistency(quotes, window=105, skip=21):
    """(Positive days - negative days) / valid days over the 126-to-21 interval.

    Require every return in the window. A flat observed day contributes zero;
    a missing day does not. Skipping the latest 21 sessions matches the existing
    six-to-one-month momentum feature. This is a signed path descriptor: steady
    rises are positive, steady declines negative. The fitted model learns its
    incremental effect conditional on the existing cumulative-return inputs.
    """
    if window < 1 or skip < 0:
        raise ValueError("Invalid trend window")
    returns = positive_finite(quotes).pct_change(fill_method=None)
    return np.sign(returns).rolling(window, min_periods=window).mean().shift(skip)


def relative_volume(volume, raw_close, adjusted, recent=21, prior=126):
    """Recent mean volume / mean of the preceding, nonoverlapping volume window.

    Use positive finite recorded volumes only. A >1.5x or <1/1.5x change in the
    adjustment factor adj_close/close flags a possible unit/corporate-action
    seam. Suppress this NEW feature while such a seam lies in its lookback;
    do not fabricate a split correction, remove a candidate, or alter prices.
    This conservative rule cannot certify all volume histories. Exceptional
    volume alone is audited, not deleted: it is the hypothesis being tested.
    """
    if recent < 1 or prior < 1:
        raise ValueError("Invalid volume window")
    for other in [raw_close, adjusted]:
        if not volume.index.equals(other.index) or not volume.columns.equals(
            other.columns
        ):
            raise ValueError("Volume and price axes differ")
    v = positive_finite(volume)
    factor = positive_finite(adjusted) / positive_finite(raw_close)
    factor = factor.where(np.isfinite(factor) & factor.gt(0))
    factor_move = factor / factor.shift(1)
    seam = factor_move.gt(1.5) | factor_move.lt(1 / 1.5)
    width = recent + prior
    seam_window = seam.rolling(width, min_periods=width).max().eq(1)
    known_units = factor.notna().rolling(width, min_periods=width).sum().eq(width)
    denominator = v.rolling(prior, min_periods=prior).mean().shift(recent)
    ratio = v.rolling(recent, min_periods=recent).mean() / denominator
    usable = known_units & ~seam_window & np.isfinite(ratio)
    return ratio.where(usable), {
        "adjustment_factor_seam": seam,
        "seam_in_volume_window": seam_window,
        "known_factor_window": known_units,
    }


def relative_long_volatility(quotes, short=63, long=756):
    """Recent return std / three-year std; identical annualization cancels.

    Require complete unfilled windows and a nonzero long-run denominator.
    A long window containing a >2x/<0.5x daily price ratio leaves this feature
    missing, reusing the existing risk-screen magnitude as a conservative
    observability rule. It does not label the move erroneous or clip a return.
    Candidate eligibility and all other model inputs remain unchanged.
    """
    if not 2 <= short <= long:
        raise ValueError("Invalid volatility windows")
    q = positive_finite(quotes)
    ret = q.pct_change(fill_method=None)
    recent = ret.rolling(short, min_periods=short).std(ddof=1)
    historical = ret.rolling(long, min_periods=long).std(ddof=1)
    largest_move = np.log(q / q.shift(1)).abs().rolling(long, min_periods=long).max()
    usable = historical.gt(0) & largest_move.le(np.log(2))
    ratio = (recent / historical).where(usable)
    return ratio.where(np.isfinite(ratio)), {
        "complete_long_window": historical.notna(),
        "large_move_in_long_window": largest_move.gt(np.log(2)),
    }


def sample_and_rank(panel, matrices):
    """Sample signal-close values and rank only that date's eligible peers."""
    result = panel.copy()
    eligible = result.eligible & result.risk_screen_pass
    for name, matrix in matrices.items():
        days = matrix.index.get_indexer(pd.DatetimeIndex(result.date))
        symbols = matrix.columns.get_indexer(result.symbol)
        if (days < 0).any() or (symbols < 0).any():
            raise ValueError("Feature matrix is missing a candidate key")
        result[name] = matrix.to_numpy()[days, symbols]
        assert not np.isinf(result[name]).any()
        result["rank_" + name] = (
            result[name].where(eligible).groupby(result.date).rank(pct=True)
        )
    return result
