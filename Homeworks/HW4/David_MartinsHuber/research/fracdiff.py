"""Fractional differentiation following Lopez de Prado (AFML, ch. 5).

The fixed-width window fractional differencing (FFD) transform is causal:
the value at time ``t`` is a weighted sum of the current and past log prices
only.  The weights are

``w_0 = 1`` and ``w_k = -w_{k-1} * (d - k + 1) / k``,

truncated when ``|w_k|`` falls below ``threshold``.  A grid of ``d`` values is
tested with an ADF regression; the smallest ``d`` that rejects the unit-root
null is used.  This retains as much memory as possible while producing a
stationary input for the HMM.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Return the truncated FFD weights for ``d``.

    ``threshold`` controls the width.  AFML uses ``1e-5``; a larger value
    shortens the window when the sample is short.
    """
    if not np.isfinite(d) or d < 0:
        raise ValueError("d must be finite and nonnegative")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")

    weights = [1.0]
    while True:
        k = len(weights)
        nxt = -weights[-1] * (d - k + 1) / k
        if abs(nxt) <= threshold:
            break
        weights.append(nxt)
        if len(weights) > 1_000_000:
            raise ValueError("FFD weights did not truncate; increase threshold")
    return np.asarray(weights, dtype=float)


def ffd(series: Iterable[float], d: float, threshold: float = 1e-5) -> np.ndarray:
    """Apply causal FFD to a one-dimensional series.

    The first ``len(weights) - 1`` observations are ``NaN`` because the
    truncated filter has no full history there.  ``np.convolve`` computes the
    causal convolution in C; output ``t`` depends only on inputs ``<= t``.
    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("FFD input must be a nonempty finite series")
    weights = ffd_weights(d, threshold)
    if len(weights) > len(x):
        return np.full(len(x), np.nan)
    out = np.convolve(x, weights)[: len(x)]
    out[: len(weights) - 1] = np.nan
    return out


def adf_table(
    series: Iterable[float],
    d_grid: Iterable[float] | None = None,
    *,
    threshold: float = 1e-5,
) -> pd.DataFrame:
    """Run ADF tests for each ``d`` and report p-values and window widths."""
    if d_grid is None:
        grid = np.arange(0.0, 1.0001, 0.05)
    else:
        grid = np.asarray(tuple(d_grid), dtype=float)
    rows = []
    for d in grid:
        weights = ffd_weights(float(d), threshold)
        transformed = ffd(series, float(d), threshold)
        valid = transformed[np.isfinite(transformed)]
        pvalue = np.nan
        if len(valid) >= 50:
            try:
                result = adfuller(
                    valid,
                    regression="c",
                    autolag=None,
                    maxlag=20,
                    result_object=True,
                )
                pvalue = float(result.pvalue)
            except TypeError:  # Older statsmodels without result_object.
                pvalue = float(
                    adfuller(valid, regression="c", autolag="AIC")[1]
                )
            except Exception:
                pvalue = np.nan
        rows.append(
            {
                "d": float(d),
                "width": len(weights),
                "observations": len(valid),
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def select_d(
    series: Iterable[float],
    d_grid: Iterable[float] | None = None,
    *,
    threshold: float = 1e-5,
    pvalue: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Return the smallest ``d`` whose FFD series rejects the unit root.

    If no grid value rejects, the largest grid value is returned.  The table
    is returned for diagnostics and for the research notebook.
    """
    table = adf_table(series, d_grid, threshold=threshold)
    valid = table.dropna(subset=["pvalue"])
    if valid.empty:
        return float(table["d"].iloc[-1]), table
    accepted = valid.loc[valid["pvalue"] < pvalue]
    if accepted.empty:
        return float(table["d"].iloc[-1]), table
    return float(accepted["d"].iloc[0]), table
