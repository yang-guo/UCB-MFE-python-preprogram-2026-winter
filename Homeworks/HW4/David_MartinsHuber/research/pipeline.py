"""End-to-end research pipeline with two independent samples.

The volume-bar study and the FFD study are deliberately separate:

* ``volume`` uses the volume-bar sample and bar log returns.
* ``ffd`` uses the native daily log-price sample and daily FFD values.

They share the underlying market history and the forward-return evaluation
only.  Neither sample is filtered or redefined by the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bars import VolumeBars, volume_bars
from .hmm import (
    WalkForwardResult,
    walk_forward_ffd_states,
    walk_forward_states,
)
from .market import MarketSeries


@dataclass(frozen=True)
class ResearchConfig:
    """Settings for the two independent research pipelines."""

    # Volume-bar pipeline.
    median_window: int = 90
    bucket_multiple: float = 5.0
    volume_min_train: int = 250          # bars
    volume_refit_every: int = 13         # bars, quarterly
    # Daily FFD pipeline.
    ffd_min_train: int = 504             # daily observations, ~2 years
    ffd_refit_every: int = 63            # daily observations, quarterly
    ffd_d_refit_every: int = 252         # daily observations, annually
    # Shared HMM settings.
    n_states: int = 3
    n_init: int = 3
    seed: int = 7
    n_iter: int = 200
    ffd_threshold: float = 1e-5
    ffd_pvalue: float = 0.05
    ffd_d_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            float(value) for value in np.round(np.arange(0.0, 1.0001, 0.05), 2)
        )
    )


@dataclass(frozen=True)
class VolumeResearch:
    market: str
    bars: VolumeBars
    states: WalkForwardResult


@dataclass(frozen=True)
class FFDResearch:
    market: str
    daily_log_price: pd.Series
    states: WalkForwardResult


@dataclass(frozen=True)
class MarketResearch:
    market: str
    volume: VolumeResearch
    ffd: FFDResearch


def run_market(series: MarketSeries, config: ResearchConfig) -> MarketResearch:
    """Run the volume-bar and daily-FFD studies independently."""
    price, volume = series.aligned()

    bars = volume_bars(
        price,
        volume,
        median_window=config.median_window,
        bucket_multiple=config.bucket_multiple,
    )
    volume_states = walk_forward_states(
        bars.bars["log_return"].to_numpy(dtype=float),
        bars.bars["end"],
        min_train=config.volume_min_train,
        refit_every=config.volume_refit_every,
        n_states=config.n_states,
        n_init=config.n_init,
        seed=config.seed,
        n_iter=config.n_iter,
    )

    daily_log_price = pd.Series(
        np.log(price.to_numpy(dtype=float)), index=price.index, name="log_price"
    )
    ffd_states = walk_forward_ffd_states(
        daily_log_price.to_numpy(dtype=float),
        daily_log_price.index,
        min_train=config.ffd_min_train,
        refit_every=config.ffd_refit_every,
        d_refit_every=config.ffd_d_refit_every,
        n_states=config.n_states,
        n_init=config.n_init,
        seed=config.seed,
        n_iter=config.n_iter,
        threshold=config.ffd_threshold,
        d_grid=config.ffd_d_grid,
        pvalue=config.ffd_pvalue,
    )

    return MarketResearch(
        series.name,
        VolumeResearch(series.name, bars, volume_states),
        FFDResearch(series.name, daily_log_price, ffd_states),
    )


def state_summary(
    result: WalkForwardResult,
    period_returns: pd.Series,
    n_states: int = 3,
) -> pd.DataFrame:
    """Describe each out-of-sample state using its period return.

    ``period_returns`` is indexed by the same dates as ``result.states``:
    bar returns for the volume pipeline, daily log returns for the FFD
    pipeline.
    """
    frame = pd.concat(
        [
            result.states.rename("state"),
            period_returns.rename("period_return"),
            result.feature.rename("feature"),
        ],
        axis=1,
    ).dropna()
    frame = frame.loc[frame["state"] >= 0]
    rows = []
    for state in range(n_states):
        sample = frame.loc[frame["state"].eq(state)]
        rows.append(
            {
                "state": state,
                "observations": len(sample),
                "share": len(sample) / len(frame) if len(frame) else np.nan,
                "mean_period_return": sample["period_return"].mean(),
                "std_period_return": sample["period_return"].std(),
                "mean_feature": sample["feature"].mean(),
                "positive_share": (sample["period_return"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def forward_return(
    price: pd.Series, dates, *, days: int = 90
) -> pd.Series:
    """Next-session entry to the first close at/after entry + holding days.

    The supplied market-price index is the session calendar. Unobserved
    signal dates, missing next entries, and unfinished exits are censored.
    This evaluation-only function must never enter model fitting inputs.
    """
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError("days must be a positive integer")
    index = pd.DatetimeIndex(price.index)
    values = price.to_numpy(dtype=float)
    if (index.hasnans or index.has_duplicates or not index.is_monotonic_increasing
            or index.tz is not None or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Prices require ordered unique dates and finite positive values")
    out = {}
    for date in pd.DatetimeIndex(dates):
        i = index.searchsorted(date, side="right")
        if date not in index or i >= len(index):
            out[date] = np.nan
            continue
        target = index[i] + pd.Timedelta(days=int(days))
        j = index.searchsorted(target, side="left")
        if j >= len(index):
            out[date] = np.nan
        else:
            out[date] = values[j] / values[i] - 1.0
    return pd.Series(out, name=f"forward_{days}d")


def transition_matrix(result: WalkForwardResult) -> pd.DataFrame:
    """Return the final fitted transition matrix in canonical state order."""
    if not result.fits:
        raise ValueError("The walk-forward run has no fitted models")
    fit = result.fits[-1][-1]
    order = fit.state_order
    matrix = fit.model.transmat_[np.ix_(order, order)]
    labels = [f"state_{k}" for k in range(matrix.shape[0])]
    return pd.DataFrame(matrix, index=labels, columns=labels)
