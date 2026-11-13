"""Walk-forward Gaussian hidden Markov models.

Every fit uses observations strictly before the observation being classified.
The current observation is scored with the last fitted model, so the state
series is out-of-sample at every point.  States are relabelled by their
emission mean (0 = lowest, ``n_states - 1`` = highest) so labels are
comparable across refits.

Two independent samples are supported:

* ``walk_forward_states`` classifies a supplied series, used for the
  volume-bar log returns.
* ``walk_forward_ffd_states`` classifies the native daily FFD series, used
  for the daily fractional-differencing study.

The FFD variant re-selects ``d`` at each refit using only past daily prices.
The FFD filter itself is causal.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .fracdiff import ffd, select_d

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")


@dataclass(frozen=True)
class HMMFit:
    model: GaussianHMM
    mean: float
    scale: float
    state_order: np.ndarray
    log_likelihood: float
    seed: int


@dataclass(frozen=True)
class WalkForwardResult:
    states: pd.Series
    probabilities: pd.DataFrame
    versions: pd.Series
    feature: pd.Series
    fits: tuple
    d: pd.Series | None = None


def _standardize(x: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return mean, scale


def _repair_model(model: GaussianHMM) -> None:
    """Repair degenerate transition/start/covariance parameters.

    hmmlearn can leave a transition row with zero mass when the final
    training observation is an isolated large move. ``model.score`` then
    rejects the model. A tiny structural repair keeps the fitted state
    structure while making the parameters valid.
    """
    components = model.n_components
    trans = np.asarray(model.transmat_, dtype=float)
    if trans.shape != (components, components):
        raise ValueError("Invalid transition matrix shape")
    trans = np.where(np.isfinite(trans) & (trans >= 0), trans, 1.0)
    row_sums = trans.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0)
    if bad.any():
        trans[bad] = 1.0
        row_sums = trans.sum(axis=1, keepdims=True)
    model.transmat_ = trans / row_sums

    start = np.asarray(model.startprob_, dtype=float)
    if (start.shape != (components,) or not np.isfinite(start).all()
            or (start < 0).any()):
        start = np.full(components, 1.0 / components)
    else:
        total = float(start.sum())
        if not np.isfinite(total) or total <= 0:
            start = np.full(components, 1.0 / components)
        else:
            start = start / total
    model.startprob_ = start

    means = np.asarray(model.means_, dtype=float)
    if means.ndim != 2 or means.shape[0] != components or not np.isfinite(means).all():
        raise ValueError("HMM means are not finite")
    n_features = means.shape[1]
    covars = np.asarray(model.covars_, dtype=float)
    if covars.shape == (components, n_features, n_features):
        diagonal = np.diagonal(covars, axis1=1, axis2=2)
    elif covars.shape == (components, n_features):
        diagonal = covars
    else:
        raise ValueError("Invalid covariance shape")
    if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
        floor = 1e-9
        repaired = np.where(
            np.isfinite(diagonal) & (diagonal > 0), diagonal, floor
        )
        model.covars_ = repaired


def fit_hmm(
    x: Iterable[float],
    *,
    n_states: int = 3,
    n_init: int = 3,
    seed: int = 0,
    n_iter: int = 200,
    tol: float = 1e-4,
) -> HMMFit:
    """Fit a diagonal-covariance Gaussian HMM with several restarts."""
    values = np.asarray(x, dtype=float).ravel()
    if values.size < max(20, 5 * n_states):
        raise ValueError("Not enough observations to fit the HMM")
    if not np.isfinite(values).all():
        raise ValueError("HMM observations must be finite")
    if not isinstance(n_states, int) or n_states < 2:
        raise ValueError("n_states must be an integer of at least 2")
    if not isinstance(n_init, int) or n_init < 1:
        raise ValueError("n_init must be a positive integer")

    # Standardization uses the training sample only; inference reuses it.
    mean, scale = _standardize(values)
    standardized = ((values - mean) / scale).reshape(-1, 1)

    best: tuple[float, GaussianHMM, int] | None = None
    for offset in range(n_init):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=n_iter,
                tol=tol,
                random_state=seed + offset,
                init_params="stmc",
            )
            model.fit(standardized)
            _repair_model(model)
            likelihood = float(model.score(standardized))
        except Exception:
            continue
        if not np.isfinite(likelihood):
            continue
        if best is None or likelihood > best[0]:
            best = (likelihood, model, seed + offset)
    if best is None:
        raise RuntimeError("HMM fitting failed for every initialization")

    likelihood, model, used_seed = best
    order = np.argsort(model.means_.ravel(), kind="stable")
    return HMMFit(model, mean, scale, order, likelihood, used_seed)


def infer_state(fit: HMMFit, history: Iterable[float]) -> tuple[int, np.ndarray]:
    """Return the current state and rank-ordered posterior probabilities."""
    values = np.asarray(history, dtype=float).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("State inference needs a finite history")
    standardized = ((values - fit.mean) / fit.scale).reshape(-1, 1)
    posterior = fit.model.predict_proba(standardized)[-1]
    ranked = posterior[fit.state_order]
    return int(np.argmax(ranked)), ranked


def walk_forward_states(
    x: Iterable[float],
    dates: Iterable,
    *,
    min_train: int = 250,
    refit_every: int = 13,
    n_states: int = 3,
    n_init: int = 3,
    seed: int = 0,
    n_iter: int = 200,
) -> WalkForwardResult:
    """Classify each observation with a model fitted on earlier observations.

    ``x`` may contain leading ``NaN`` values (for example the FFD warm-up).
    They are skipped when fitting and never receive a state.
    """
    values = np.asarray(x, dtype=float).ravel()
    index = pd.DatetimeIndex(dates)
    if len(values) != len(index):
        raise ValueError("Observations and dates must have the same length")
    if not isinstance(min_train, int) or min_train < 20:
        raise ValueError("min_train must be an integer of at least 20")
    if not isinstance(refit_every, int) or refit_every < 1:
        raise ValueError("refit_every must be a positive integer")

    n = len(values)
    states = np.full(n, -1, dtype=int)
    probabilities = np.full((n, n_states), np.nan)
    versions = np.full(n, -1, dtype=int)
    fits: list[tuple] = []
    fit: HMMFit | None = None
    version = -1
    last_refit: int | None = None

    for t in range(n):
        # Fit only on observations strictly before t; t is scored, never trained on.
        train = values[:t]
        train_valid = train[np.isfinite(train)]
        if train_valid.size < min_train:
            continue
        if fit is None or (last_refit is not None and t - last_refit >= refit_every):
            fit = fit_hmm(
                train_valid,
                n_states=n_states,
                n_init=n_init,
                seed=seed,
                n_iter=n_iter,
            )
            version += 1
            last_refit = t
            fits.append((index[t], version, fit))
        if not np.isfinite(values[t]):
            continue
        history = values[: t + 1]
        history = history[np.isfinite(history)]
        state, ranked = infer_state(fit, history)
        states[t] = state
        probabilities[t] = ranked
        versions[t] = version

    return WalkForwardResult(
        states=pd.Series(states, index=index, name="state"),
        probabilities=pd.DataFrame(
            probabilities, index=index, columns=[f"p{k}" for k in range(n_states)]
        ),
        versions=pd.Series(versions, index=index, name="model_version"),
        feature=pd.Series(values, index=index, name="feature"),
        fits=tuple(fits),
    )


def walk_forward_ffd_states(
    daily_log_price: Iterable[float],
    dates: Iterable | None = None,
    *,
    min_train: int = 504,
    refit_every: int = 63,
    d_refit_every: int = 252,
    n_states: int = 3,
    n_init: int = 3,
    seed: int = 0,
    n_iter: int = 200,
    threshold: float = 1e-5,
    d_grid: Iterable[float] | None = None,
    pvalue: float = 0.05,
) -> WalkForwardResult:
    """Walk-forward HMM on the native daily FFD series.

    The sample is the daily log-price history; it is deliberately independent
    of the volume-bar sample.  ``d`` is re-selected every ``d_refit_every``
    daily observations from history strictly before the current observation,
    and the HMM is refitted every ``refit_every`` observations.  The first
    ``len(weights) - 1`` observations are the FFD warm-up and receive no
    state.
    """
    daily = np.asarray(daily_log_price, dtype=float).ravel()
    if daily.size == 0 or not np.isfinite(daily).all():
        raise ValueError("Daily log prices must be a nonempty finite series")
    if dates is None:
        index = pd.RangeIndex(daily.size)
    else:
        index = pd.DatetimeIndex(dates)
    if len(index) != daily.size:
        raise ValueError("Daily log prices and dates must have the same length")
    if not isinstance(min_train, int) or min_train < 20:
        raise ValueError("min_train must be an integer of at least 20")
    if not isinstance(refit_every, int) or refit_every < 1:
        raise ValueError("refit_every must be a positive integer")
    if not isinstance(d_refit_every, int) or d_refit_every < 1:
        raise ValueError("d_refit_every must be a positive integer")

    n = daily.size
    states = np.full(n, -1, dtype=int)
    probabilities = np.full((n, n_states), np.nan)
    versions = np.full(n, -1, dtype=int)
    ds = np.full(n, np.nan)
    used_features = np.full(n, np.nan)
    fits: list[tuple] = []
    fit: HMMFit | None = None
    version = -1
    last_refit: int | None = None
    last_d_refit: int | None = None
    current_d = np.nan
    features = np.full(n, np.nan)

    for t in range(n):
        if t == 0:
            continue
        need_d = last_d_refit is None or t - last_d_refit >= d_refit_every
        if need_d:
            # d is selected from history strictly before t; the current day is only scored.
            candidate_d, _ = select_d(
                daily[:t],
                d_grid,
                threshold=threshold,
                pvalue=pvalue,
            )
            candidate_features = ffd(daily, candidate_d, threshold)
            if candidate_d != current_d:
                fit = None  # The feature definition changed; old fit is stale.
            current_d = candidate_d
            features = candidate_features
            last_d_refit = t

        need_fit = fit is None or (
            last_refit is not None and t - last_refit >= refit_every
        )
        if need_fit:
            train = features[:t]
            train_valid = train[np.isfinite(train)]
            if train_valid.size >= min_train:
                fit = fit_hmm(
                    train_valid,
                    n_states=n_states,
                    n_init=n_init,
                    seed=seed,
                    n_iter=n_iter,
                )
                version += 1
                last_refit = t
                fits.append((index[t], version, current_d, fit))
            else:
                fit = None

        if fit is None:
            continue
        ds[t] = current_d
        if not np.isfinite(features[t]):
            continue
        used_features[t] = features[t]
        history = features[: t + 1]
        history = history[np.isfinite(history)]
        state, ranked = infer_state(fit, history)
        states[t] = state
        probabilities[t] = ranked
        versions[t] = version

    return WalkForwardResult(
        states=pd.Series(states, index=index, name="state"),
        probabilities=pd.DataFrame(
            probabilities, index=index, columns=[f"p{k}" for k in range(n_states)]
        ),
        versions=pd.Series(versions, index=index, name="model_version"),
        feature=pd.Series(used_features, index=index, name="feature"),
        fits=tuple(fits),
        d=pd.Series(ds, index=index, name="d"),
    )
