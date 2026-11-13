"""Volume bars for a 90-day holding horizon.

The rule requested for the research is:

1. look back over a trailing 90-session window,
2. take the median daily volume in that window,
3. fill a bucket until it reaches ``5 * median`` (a weekly proxy),
4. keep the bucket's start/end dates and aggregate the price return.

The bucket target is fixed at the bar's start using the trailing median
through the **prior** session, so the target does not move while the bar is
being filled and does not use the start session's own volume.  Every bar
return uses the close of the day before the bar starts, so the construction
is causal.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeBars:
    """Completed bars, the rolling median, and the aligned inputs."""

    bars: pd.DataFrame
    median_volume: pd.Series
    price: pd.Series
    volume: pd.Series

    @property
    def count(self) -> int:
        return len(self.bars)


def volume_bars(
    price: pd.Series,
    volume: pd.Series,
    *,
    median_window: int = 90,
    bucket_multiple: float = 5.0,
    target_prior_session: bool = False,
) -> VolumeBars:
    """Aggregate daily observations into volume buckets.

    Parameters
    ----------
    price, volume
        Daily series indexed by session date.  They are inner-joined and
        sorted before aggregation.
    median_window
        Trailing sessions used for the median daily volume.
    bucket_multiple
        Bucket target as a multiple of the trailing median.  ``5`` gives a
        roughly weekly bar at daily sampling.
    target_prior_session
        When true, fix the target from the trailing median through the prior
        session, so the bar-start session's own volume is not used.  The
        legacy default preserves the original lecture-research bar construction.
    """
    if (isinstance(median_window, bool)
            or not isinstance(median_window, numbers.Integral)
            or median_window < 2):
        raise ValueError("median_window must be an integer of at least 2")
    if (isinstance(bucket_multiple, bool)
            or not isinstance(bucket_multiple, numbers.Real)
            or not np.isfinite(bucket_multiple)
            or bucket_multiple <= 0):
        raise ValueError("bucket_multiple must be finite and positive")

    frame = pd.concat(
        [price.rename("price"), volume.rename("volume")], axis=1
    ).dropna()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise ValueError("No overlapping price and volume observations")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Price and volume must be finite")
    if (frame["price"] <= 0).any() or (frame["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")

    p = frame["price"].to_numpy(dtype=float)
    v = frame["volume"].to_numpy(dtype=float)
    median = (
        frame["volume"]
        .rolling(median_window, min_periods=median_window)
        .median()
        .to_numpy(dtype=float)
    )

    rows: list[dict] = []
    i = median_window if target_prior_session else median_window - 1
    while i < len(frame):
        target_index = i - 1 if target_prior_session else i
        target = bucket_multiple * median[target_index]
        if not np.isfinite(target) or target <= 0:
            i += 1
            continue

        start = i
        cumulative = 0.0
        while i < len(frame) and cumulative < target:
            cumulative += v[i]
            i += 1
        if cumulative < target:
            break  # Drop an incomplete final bucket.

        end = i - 1
        base = start - 1  # The prior session closes the previous bar.
        rows.append(
            {
                "bar_id": len(rows),
                "start": frame.index[start],
                "end": frame.index[end],
                "start_pos": start,
                "end_pos": end,
                "base_pos": base,
                "days": end - start + 1,
                "base_price": p[base],
                "close_price": p[end],
                "log_return": float(np.log(p[end] / p[base])),
                "bar_volume": cumulative,
                "target": target,
                "median_volume": median[target_index],
                "bucket_multiple": bucket_multiple,
            }
        )

    if not rows:
        raise ValueError("The volume rule produced no completed bars")
    bars = pd.DataFrame(rows)
    if not np.isfinite(bars["log_return"]).all():
        raise ValueError("Bar log returns must be finite")
    return VolumeBars(
        bars=bars,
        median_volume=pd.Series(
            median, index=frame.index, name="median_volume"
        ),
        price=frame["price"],
        volume=frame["volume"],
    )
