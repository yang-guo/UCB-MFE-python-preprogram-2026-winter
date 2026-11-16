"""Small preregistered model comparisons with fully completed training labels.

Lecture 7 motivates Ridge, train-only transformations and chronological folds.
Lecture 8 motivates fit-date audits and optional quarterly updates. Monthly
sampling reduces repeated, highly overlapping targets but does not eliminate
their dependence. Prediction candidates never require an observable target.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import TrainingClipper, TargetClippedRegressor
from .settings import PRICE_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple
    quarterly: bool = False
    alpha: float = 100.0
    momentum_reference: bool = False
    target_quantiles: tuple | None = None
    risk_screen: bool = False
    estimator: str = "ridge"


SPECS = [
    ModelSpec("momentum_reference", tuple(PRICE_FEATURES), momentum_reference=True),
    ModelSpec("ridge_price", tuple(PRICE_FEATURES)),
    ModelSpec("ridge_plus_roa", tuple(PRICE_FEATURES + ["operating_roa"])),
    ModelSpec(
        "ridge_plus_roa_fcf", tuple(PRICE_FEATURES + ["operating_roa", "fcf_to_assets"])
    ),
]

# Added only after the unmodified development run exposed extreme source labels.
# Each comparison changes one element; original experiments remain available.
ROBUST_SPECS = [
    ModelSpec("robust_price", tuple(PRICE_FEATURES), target_quantiles=(0.01, 0.99)),
    ModelSpec(
        "robust_price_screened",
        tuple(PRICE_FEATURES),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
    ),
    ModelSpec(
        "robust_roa_screened",
        tuple(PRICE_FEATURES + ["operating_roa"]),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
    ),
    ModelSpec(
        "robust_roa_fcf_screened",
        tuple(PRICE_FEATURES + ["operating_roa", "fcf_to_assets"]),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
    ),
]
TREE_SPECS = [
    ModelSpec(
        "tree_roa_fcf_screened",
        tuple(PRICE_FEATURES + ["operating_roa", "fcf_to_assets"]),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
        estimator="hist",
    ),
]
MOMENTUM_SPECS = [
    ModelSpec(
        "long_momentum",
        ("momentum_252_21",),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
    ),
]
ALL_SPECS = SPECS + ROBUST_SPECS + TREE_SPECS + MOMENTUM_SPECS


def make_model(spec):
    if spec.estimator == "ridge":
        estimator = Ridge(alpha=spec.alpha)
    elif spec.estimator == "hist":
        estimator = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            max_depth=3,
            min_samples_leaf=100,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=42,
        )
    else:
        raise ValueError("Unknown estimator")
    if spec.target_quantiles is not None:
        estimator = TargetClippedRegressor(estimator, spec.target_quantiles)
    return Pipeline(
        [
            ("clip", TrainingClipper()),
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler()),
            ("regression", estimator),
        ]
    )


def training_rows(panel, asof, max_signal=None):
    """Purged expanding window: a known feature with an unfinished label is unusable."""
    use = (
        panel.eligible
        & panel.date.ge("2010-01-01")
        & panel.date.lt(asof)
        & panel.label_end.lt(asof)
        & panel.target.notna()
    )
    if max_signal is not None:
        use &= panel.date.le(max_signal)
    return panel.loc[use].copy()


def predict_period(panel, calendar, spec, start, end, final_period=False):
    """Fit at prior close; save training timing, coefficients and forecast provenance.

    For final fixed fitting, training signals stop in 2015 even though the buffer
    permits their outcomes to finish. If quarterly refitting is explicitly
    selected, subsequent fits may use newly completed test-period outcomes.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if spec.risk_screen:
        panel = panel.copy()
        panel["eligible"] &= panel.risk_screen_pass
    first = calendar.searchsorted(start)
    if first == 0:
        raise ValueError("Missing prior signal date")
    initial_fit = calendar[first - 1]
    fit_dates = [initial_fit]
    if spec.quarterly:
        for i in range(first + 1, len(calendar)):
            if calendar[i] > end:
                break
            if calendar[i].to_period("Q") != calendar[i - 1].to_period("Q"):
                fit_dates.append(calendar[i - 1])
    scored, audit, fitted_models = [], [], []
    for i, asof in enumerate(fit_dates):
        next_fit = (
            fit_dates[i + 1] if i + 1 < len(fit_dates) else end + pd.Timedelta(days=1)
        )
        candidates = panel.loc[
            panel.eligible
            & panel.date.ge(asof)
            & panel.date.lt(next_fit)
            & panel.entry_date.ge(start)
            & panel.entry_date.le(end)
        ].copy()
        if candidates.empty:
            continue
        max_signal = "2015-12-31" if final_period and i == 0 else None
        training = training_rows(panel, asof, max_signal)
        if len(training) < 20:
            raise ValueError("Too few completed training examples")
        if not training.label_end.max() < asof:
            raise ValueError("Training label leakage")
        if spec.momentum_reference:
            prediction = candidates.momentum_63.to_numpy()
            fitted = None
        else:
            fitted = make_model(spec)
            fitted.fit(training[list(spec.features)], training.target)
            prediction = fitted.predict(candidates[list(spec.features)])
            fitted_models.append((asof, fitted))
        output = candidates[["date", "symbol", "target", "label_end"]].copy()
        output["score"], output["fit_date"] = prediction, asof
        scored.append(output)
        bounds = (
            getattr(fitted["regression"], "target_bounds_", [np.nan, np.nan])
            if fitted is not None
            else [np.nan, np.nan]
        )
        audit.append(
            {
                "model": spec.name,
                "fit_date": asof,
                "training_rows": len(training),
                "target_clip_lower": bounds[0],
                "target_clip_upper": bounds[1],
                "target_clip_count": getattr(fitted["regression"], "clipped_labels_", 0)
                if fitted is not None
                else 0,
                "last_training_signal": training.date.max(),
                "last_training_label": training.label_end.max(),
                "first_prediction": candidates.date.min(),
                "last_prediction": candidates.date.max(),
                "predictions": len(candidates),
                "censored_predictions": int(candidates.target.isna().sum()),
            }
        )
    if not scored:
        raise ValueError("No predictions in requested period")
    return pd.concat(scored, ignore_index=True), pd.DataFrame(audit), fitted_models


def forecast_metrics(scored, cutoff):
    """Evaluate completed targets only; report their coverage explicitly."""
    g = scored.loc[scored.target.notna() & scored.label_end.le(cutoff)].copy()
    if g.empty:
        return {"labeled_predictions": 0}
    error = g.score - g.target
    ic = []
    for _, day in g.groupby("date"):
        if len(day) >= 5 and day.score.nunique() > 1 and day.target.nunique() > 1:
            ic.append(day.score.corr(day.target, method="spearman"))
    return {
        "labeled_predictions": len(g),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(abs(error))),
        "mean_rank_IC": float(np.nanmean(ic)) if ic else np.nan,
        "last_labeled_signal": g.date.max(),
        "last_label_end": g.label_end.max(),
    }
