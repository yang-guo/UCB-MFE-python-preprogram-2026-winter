"""Causality and independent algebra for the paired price-model comparison."""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from hw4.modeling import predict_period
from price_model_research import SPECS, audit_fit


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_future_targets_cannot_change_fits_or_candidate_predictions(spec):
    rng = np.random.default_rng(37)
    n = 500
    panel = pd.DataFrame(
        {
            "symbol": [f"T{i}" for i in range(n)],
            "date": pd.Timestamp("2011-08-31"),
            "entry_date": pd.Timestamp("2011-09-01"),
            "label_end": pd.Timestamp("2011-11-30"),
            "eligible": True,
            "risk_screen_pass": True,
            "momentum_63": rng.normal(0, 0.1, n),
            "momentum_252_21": rng.normal(0, 0.2, n),
            "volatility_63": rng.uniform(0.1, 0.5, n),
        }
    )
    panel["target"] = 0.03 - 0.1 * panel.momentum_252_21 + 0.02 * rng.normal(size=n)
    future = panel.iloc[:8].copy()
    future["date"] = pd.Timestamp("2012-12-31")
    future["entry_date"] = pd.Timestamp("2013-01-02")
    future["label_end"] = pd.Timestamp("2013-04-02")
    panel = pd.concat([panel, future], ignore_index=True)
    changed = panel.copy()
    changed.loc[changed.date.eq("2012-12-31"), "target"] = [
        np.nan,
        999,
        -999,
        0,
        2,
        3,
        4,
        5,
    ]
    calendar = pd.bdate_range("2012-12-28", "2013-01-10")
    with threadpool_limits(limits=2):
        a, aa, ma = predict_period(panel, calendar, spec, "2013-01-01", "2013-01-10")
        b, ab, mb = predict_period(changed, calendar, spec, "2013-01-01", "2013-01-10")
        np.testing.assert_allclose(a.score, b.score, atol=1e-12, rtol=0)
        assert len(a) == 8 and aa.training_rows.iloc[0] == n
        assert aa.last_training_label.iloc[0] < aa.fit_date.iloc[0]
        audit_fit(panel, ma[0][0], ma[0][1], spec)
        np.testing.assert_allclose(ma[0][1]["clip"].bounds_, mb[0][1]["clip"].bounds_)
        # Request batches also must not refit or alter the prediction transform.
        model = ma[0][1]
        np.testing.assert_allclose(
            model.predict(future[list(spec.features)])[:1],
            model.predict(future[list(spec.features)].iloc[:1]),
        )
