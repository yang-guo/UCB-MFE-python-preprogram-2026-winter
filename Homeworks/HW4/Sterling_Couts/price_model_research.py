"""Fixed three-price-feature models and independent linear-fit checks.

Reuse the frozen preprocessing/estimator factory without modifying it. The
nonlinear comparison changes only the estimator: both models see identical
training rows, features, learned transformations and clipped training targets.
"""

import numpy as np
import pandas as pd

from hw4.modeling import ModelSpec, training_rows
from hw4.settings import PRICE_FEATURES

SPECS = [
    ModelSpec(
        "ridge_three_price",
        tuple(PRICE_FEATURES),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
    ),
    ModelSpec(
        "tree_three_price",
        tuple(PRICE_FEATURES),
        target_quantiles=(0.01, 0.99),
        risk_screen=True,
        estimator="hist",
    ),
]


def audit_fit(panel, asof, fitted, spec):
    """Check train-only bounds and, for Ridge, solve the normal equations anew."""
    eligible = panel.copy()
    eligible["eligible"] &= eligible.risk_screen_pass
    train = training_rows(eligible, asof)
    assert train.label_end.lt(asof).all() and train.date.lt(asof).all()
    features = train[list(spec.features)]
    expected = np.nanquantile(features.to_numpy(), [0.01, 0.99], axis=0)
    np.testing.assert_allclose(fitted["clip"].bounds_, expected)
    target_bounds = np.quantile(train.target, [0.01, 0.99])
    np.testing.assert_allclose(fitted["regression"].target_bounds_, target_bounds)
    rows = {
        "training_rows": len(train),
        "last_training_label": train.label_end.max(),
        "last_training_signal": train.date.max(),
        "target_min": train.target.min(),
        "target_max": train.target.max(),
        "target_bounds_train_only": True,
    }
    coefficients = []
    if spec.estimator == "ridge":
        # Centering removes the intercept from the penalty, matching Ridge's
        # stated objective. solve() is numerically preferable to an explicit inverse.
        x = fitted[:-1].transform(features)
        y = np.clip(train.target.to_numpy(), *target_bounds)
        xm, ym = x.mean(axis=0), y.mean()
        xc, yc = x - xm, y - ym
        gram = xc.T @ xc + spec.alpha * np.eye(x.shape[1])
        beta = np.linalg.solve(gram, xc.T @ yc)
        intercept = ym - xm @ beta
        estimator = fitted["regression"].estimator_
        np.testing.assert_allclose(estimator.coef_, beta, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(estimator.intercept_, intercept, atol=1e-10)
        np.testing.assert_allclose(
            fitted.predict(features.iloc[:50]), x[:50] @ beta + intercept, atol=1e-10
        )
        names = list(spec.features) + [
            f"missing_{spec.features[i]}" for i in fitted["impute"].indicator_.features_
        ]
        coefficients = [
            {"feature": name, "standardized_coefficient": float(value)}
            for name, value in zip(names, beta)
        ]
        rows.update(
            normal_equation_error=float(np.max(np.abs(estimator.coef_ - beta))),
            penalized_condition_number=float(np.linalg.cond(gram)),
        )
    else:
        estimator = fitted["regression"].estimator_
        assert estimator.max_depth == 3 and estimator.max_leaf_nodes == 7
        assert estimator.min_samples_leaf == 100 and estimator.n_iter_ == 100
        assert not estimator.early_stopping
        rows["fixed_tree_complexity_checked"] = True
    return rows, pd.DataFrame(coefficients)


def response_slices(fitted, train, spec):
    """Training-supported descriptive slices, not causal or investment forecasts.

    Change long momentum between its training 5th/95th percentiles, hold recent
    momentum at its training median, and show three training volatility levels.
    Synthetic feature combinations may have weak joint support; state that limit.
    """
    feature = "momentum_252_21"
    grid = np.linspace(*train[feature].quantile([0.05, 0.95]), 61)
    parts = []
    for q in [0.25, 0.5, 0.75]:
        frame = pd.DataFrame(
            {name: np.repeat(train[name].median(), len(grid)) for name in spec.features}
        )
        frame[feature] = grid
        frame["volatility_63"] = train.volatility_63.quantile(q)
        frame["predicted_return"] = fitted.predict(frame[list(spec.features)])
        parts.append(frame.assign(volatility_quantile=q))
    return pd.concat(parts, ignore_index=True)
