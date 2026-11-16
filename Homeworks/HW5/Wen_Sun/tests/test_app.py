"""Integration and validation tests for the HW5 FastAPI service."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from hw4_model import build_features_frame

# path
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_REQUEST = ROOT / "example_request.json"
MODEL_METADATA = ROOT / "model_metadata.json"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    if not EXAMPLE_REQUEST.is_file():
        pytest.skip("example_request.json is missing")
    return json.loads(EXAMPLE_REQUEST.read_text())


def _direct_prediction(client, payload):
    """direct pipeline call"""
    frame = build_features_frame(payload["features"], payload["sector"])
    return float(np.asarray(client.app.state.model.predict(frame)).reshape(-1)[0])


def test_predict_valid_input(client, valid_input_data):
    """Valid prediction"""
    metadata = json.loads(MODEL_METADATA.read_text())
    response = client.post("/predict", json=valid_input_data)

    # assert 200
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == client.app.state.model_version

    direct = _direct_prediction(client, valid_input_data)
    # compare the API prediction with a direct pipeline call
    assert body["prediction"] == pytest.approx(direct, rel=1e-9, abs=1e-9)
    assert body["prediction"] == pytest.approx(
        metadata["expected_prediction"], rel=1e-9, abs=1e-9
    )


def _invalid_payloads():
    """invalid cases"""
    valid = json.loads(EXAMPLE_REQUEST.read_text())
    base_features = valid["features"]

    missing_features = {"sector": valid["sector"]}
    missing_sector = {"features": base_features}
    extra_field = {**valid, "unexpected": 1}
    wrong_count = {
        "features": base_features + [0.0],
        "sector": valid["sector"],
    }
    string_features = {
        "features": ["1.0" if i == 0 else value for i, value in enumerate(base_features)],
        "sector": valid["sector"],
    }
    boolean_feature = {
        "features": [True if i == 0 else value for i, value in enumerate(base_features)],
        "sector": valid["sector"],
    }
    out_of_bounds = {
        "features": [1e13 if i == 0 else value for i, value in enumerate(base_features)],
        "sector": valid["sector"],
    }
    invalid_sector_type = {"features": base_features, "sector": 1}
    empty_sector = {"features": base_features, "sector": ""}

    return [
        missing_features,
        missing_sector,
        extra_field,
        wrong_count,
        string_features,
        boolean_feature,
        out_of_bounds,
        invalid_sector_type,
        empty_sector,
    ]


@pytest.mark.parametrize("payload", _invalid_payloads())
def test_predict_invalid_input(client, monkeypatch, payload):
    """Input validation"""
    calls = []

    def should_not_be_called(_frame):
        calls.append(_frame)
        raise AssertionError("validation should prevent inference")

    monkeypatch.setattr(client.app.state.model, "predict", should_not_be_called)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert calls == []


def _non_finite_payload(kind):
    """payloads for non finite test"""
    valid = json.loads(EXAMPLE_REQUEST.read_text())
    base_features = valid["features"]
    replacement = float("nan") if kind == "nan" else float("inf")
    return {
        "features": [replacement] + base_features[1:],
        "sector": valid["sector"],
    }


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_predict_non_finite_input(client, monkeypatch, kind):
    """Return 422 for non-finite values"""
    calls = []

    def should_not_be_called(_frame):
        calls.append(_frame)
        raise AssertionError("validation should prevent inference")

    monkeypatch.setattr(client.app.state.model, "predict", should_not_be_called)
    payload = _non_finite_payload(kind)
    content = json.dumps(payload, allow_nan=True).encode()
    response = client.post(
        "/predict",
        content=content,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert calls == []


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    """Inference failure"""

    def raise_inference_error(_frame):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(
        failure_client.app.state.model,
        "predict",
        raise_inference_error,
    )
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    """Performance"""
    request_count = 20

    # Warm up the endpoint and let the in-process server/pipeline settle.
    for _ in range(3):
        response = client.post("/predict", json=valid_input_data)
        assert response.status_code == 200

    # delta t
    latencies = []
    for _ in range(request_count):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies.append(time.perf_counter() - start)
        assert response.status_code == 200
        assert math.isfinite(response.json()["prediction"])

    median_ms = statistics.median(latencies) * 1000
    p95_ms = sorted(latencies)[int(math.ceil(0.95 * len(latencies))) - 1] * 1000
    print(
        "\nHW5 latency report: "
        f"median={median_ms:.3f} ms, p95={p95_ms:.3f} ms, "
        f"n={request_count}, method=in-process FastAPI TestClient, "
        f"platform={platform.platform()}"
    )

    # this is a sanity check that the measured
    # values are non-negative and finite.
    assert all(duration >= 0 and math.isfinite(duration) for duration in latencies)
