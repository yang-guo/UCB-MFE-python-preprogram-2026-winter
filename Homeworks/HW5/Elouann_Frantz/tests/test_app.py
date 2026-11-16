"""Automated unit, validation, failure, and latency tests for the HW5 FastAPI service."""

import statistics
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import ARTIFACT_PATH, app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def reference_artifact():
    return joblib.load(ARTIFACT_PATH)


@pytest.fixture
def valid_input_data(reference_artifact):
    return dict(reference_artifact["reference_input"])


def test_predict_valid_input(client, reference_artifact, valid_input_data):
    """Verify 200 response schema, direct pipeline match, and recorded HW4 reference value."""
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert np.isfinite(body["prediction"])
    assert body["model_version"] == reference_artifact["model_version"]

    # Direct pipeline prediction check
    feature_names = reference_artifact["feature_names"]
    df_input = pd.DataFrame([valid_input_data], columns=feature_names)
    direct_pred = float(reference_artifact["pipeline"].predict(df_input)[0])
    assert body["prediction"] == pytest.approx(direct_pred, rel=1e-12, abs=1e-12)

    # Recorded HW4 reference output check (AAPL signal 2016-03-31 -> entry 2016-04-01)
    assert body["prediction"] == pytest.approx(
        reference_artifact["reference_prediction"], rel=1e-9, abs=1e-9
    )


@pytest.mark.parametrize(
    ("raw_payload", "is_raw_json"),
    [
        ({"mom_20d": 0.08, "mom_60d": 0.04}, False),  # Missing vol_20d
        (
            {"mom_20d": 0.08, "mom_60d": 0.04, "vol_20d": 0.01, "extra_feat": 1.0},
            False,
        ),  # Extra field
        ([0.08, 0.04, 0.01], False),  # Wrong structure / list instead of object
        ({"mom_20d": "0.08", "mom_60d": 0.04, "vol_20d": 0.01}, False),  # Numeric string
        ({"mom_20d": True, "mom_60d": 0.04, "vol_20d": 0.01}, False),  # Boolean coercion
        ({"mom_20d": -1.5, "mom_60d": 0.04, "vol_20d": 0.01}, False),  # Below -100% return
        ({"mom_20d": 0.08, "mom_60d": 0.04, "vol_20d": -0.02}, False),  # Negative volatility
        ('{"mom_20d": NaN, "mom_60d": 0.04, "vol_20d": 0.01}', True),  # Non-finite NaN
        ('{"mom_20d": Infinity, "mom_60d": 0.04, "vol_20d": 0.01}', True),  # Non-finite Inf
    ],
)
def test_predict_invalid_input(client, monkeypatch, raw_payload, is_raw_json):
    """Ensure malformed or out-of-contract payloads return 422 before reaching model.predict."""
    model_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError("Pipeline predict must not be called for invalid input.")

    monkeypatch.setattr(client.app.state.pipeline, "predict", fail_if_called)

    if is_raw_json:
        response = client.post(
            "/predict",
            content=raw_payload,
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.post("/predict", json=raw_payload)

    assert response.status_code == 422
    assert model_called is False


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    """Ensure unexpected runtime errors inside model.predict return HTTP 500."""

    def broken_predict(*args, **kwargs):
        raise RuntimeError("Simulated inference crash")

    monkeypatch.setattr(failure_client.app.state.pipeline, "predict", broken_predict)

    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "Simulated inference crash" not in response.text


def test_predict_performance(client, valid_input_data):
    """Measure in-process TestClient latency across 30 requests after warm-up."""
    for _ in range(5):
        warm_resp = client.post("/predict", json=valid_input_data)
        assert warm_resp.status_code == 200

    latencies_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=valid_input_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        latencies_ms.append(elapsed_ms)

    median_ms = statistics.median(latencies_ms)
    p95_ms = float(np.percentile(latencies_ms, 95))
    print(
        f"\nIn-process TestClient latency (n=30): median={median_ms:.2f} ms, p95={p95_ms:.2f} ms"
    )

    # Generous 250 ms p95 budget for single-row Ridge inference on local Apple M3 hardware
    assert p95_ms < 250.0
