"""Replace each TODO failure with meaningful checks for your application."""

from __future__ import annotations

# More imports
import copy
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from model_utils import FEATURES_15

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "model_artifact.joblib"


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(ARTIFACT_PATH)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data(artifact):
    # Return a JSON-compatible dict matching your documented input contract.
    return copy.deepcopy(artifact["reference_input"])


def test_predict_valid_input(client, artifact, valid_input_data):
    # Use the real saved model. Check schema, direct/API agreement, known result.
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == artifact["model_version"]

    # Confirm exact agreement with direct pipeline inference
    ref_df = pd.DataFrame(
        [[valid_input_data[col] for col in FEATURES_15]],
        columns=FEATURES_15,
    )
    direct_pred = float(artifact["pipeline"].predict(ref_df)[0])
    assert np.isclose(body["prediction"], direct_pred, rtol=1e-12, atol=1e-12)

    # Confirm agreement with the known HW4 reference prediction
    known_pred = float(artifact["reference_prediction"])
    assert np.isclose(body["prediction"], known_pred, rtol=1e-9, atol=1e-9)


#  # Parametrize missing/extra fields, types, bounds, and feature-count cases.
@pytest.mark.parametrize(
    ("mutator", "raw_body", "description"),
    [
        (lambda d: d.pop("roa"), None, "missing_required_field"),
        (lambda d: d.clear(), None, "empty_payload_missing_all_fields"),
        (lambda d: d.update({"extra_feature": 1.23}), None, "extra_unexpected_field"),
        (lambda d: d.update({"roa": "0.01657"}), None, "string_instead_of_float"),
        (lambda d: d.update({"roa": True}), None, "boolean_instead_of_float"),
        (lambda d: d.update({"roa": None}), None, "null_value"),
        (lambda d: d.update({"ret_252d": -1.5}), None, "return_below_minus_100_pct"),
        (lambda d: d.update({"vol_63d": 0.0}), None, "zero_volatility_out_of_bounds"),
        (lambda d: d.update({"neg_universe_rank": 10.0}), None, "positive_neg_universe_rank"),
        (lambda d: d.update({"avg_hl_spread_21d": -0.01}), None, "negative_hl_spread"),
        (lambda d: d.update({"mcap_to_rev_ttm": -2.0}), None, "negative_mcap_to_rev"),
        (lambda d: d.update({"ebitda_to_assets": 500.0}), None, "ratio_above_upper_bound"),
        (None, '{"ebitda_to_assets": NaN}', "non_finite_nan_raw_json"),
    ],
)
def test_predict_invalid_input(
    client, valid_input_data, monkeypatch, mutator, raw_body, description
):
    # Confirm rejected requests do not call the model.
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Model predict should not be called on invalid input")

    monkeypatch.setattr(app.state.pipeline, "predict", fail_if_called)

    if raw_body is not None:
        response = client.post(
            "/predict",
            content=raw_body,
            headers={"Content-Type": "application/json"},
        )
    else:
        payload = copy.deepcopy(valid_input_data)
        mutator(payload)
        response = client.post("/predict", json=payload)

    assert response.status_code == 422, f"Failed case: {description}"
    assert called is False


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    # Patch predict on the loaded model to raise, send VALID input, assert 500.
    # Do not assume a JSON error body unless your application defines one.
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("Simulated inference crash")

    monkeypatch.setattr(app.state.pipeline, "predict", raise_runtime_error)

    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "Simulated inference crash" not in response.text


def test_predict_performance(client, valid_input_data):
    # Warm up, then measure >=20 successful requests with time.perf_counter.
    # Report median and p95; document an environment-specific budget if used.
    for _ in range(5):
        warm_resp = client.post("/predict", json=valid_input_data)
        assert warm_resp.status_code == 200

    n_requests = 30
    latencies_ms: list[float] = []
    for _ in range(n_requests):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=valid_input_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        latencies_ms.append(elapsed_ms)

    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    print(
        f"\nLatency over {n_requests} requests -> median: {median_ms:.2f} ms | p95: {p95_ms:.2f} ms"
    )

    assert median_ms < 50.0
    assert p95_ms < 100.0
