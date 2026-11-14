"""Contract, reference-model, failure and latency checks for the HW5 API."""

import json
import math
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import FEATURE_NAMES, MODEL_VERSION

HERE = Path(__file__).resolve().parents[1]


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
    return json.loads((HERE / "example_request.json").read_text())


@pytest.fixture
def inference_spy(client, monkeypatch):
    spy = Mock(wraps=app.state.model.predict)
    monkeypatch.setattr(app.state.model, "predict", spy)
    return spy


def test_predict_valid_input(client, valid_input_data):
    reference = json.loads((HERE / "reference_prediction.json").read_text())
    loaded_model = app.state.model
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert result["model_version"] == MODEL_VERSION
    assert math.isfinite(result["prediction"])
    direct = loaded_model.predict(pd.DataFrame([valid_input_data], columns=FEATURE_NAMES))[0]
    assert result["prediction"] == pytest.approx(float(direct), rel=1e-6, abs=1e-9)
    assert result["prediction"] == pytest.approx(
        reference["expected_prediction"], rel=1e-6, abs=1e-9
    )
    assert client.post("/predict", json=valid_input_data).status_code == 200
    assert app.state.model is loaded_model


def test_numeric_integer_and_reordered_fields(client, valid_input_data):
    request = dict(reversed(list(valid_input_data.items())))
    request["filing_age_days"] = 38
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    reference = json.loads((HERE / "reference_prediction.json").read_text())
    assert response.json()["prediction"] == pytest.approx(reference["expected_prediction"])


@pytest.mark.parametrize("missing_feature", FEATURE_NAMES)
def test_missing_required_feature(client, valid_input_data, missing_feature, inference_spy):
    request = valid_input_data.copy()
    del request[missing_feature]
    assert client.post("/predict", json=request).status_code == 422
    inference_spy.assert_not_called()


@pytest.mark.parametrize("invalid_value", ["0.1", True, None, [0.1], {"value": 0.1}])
def test_invalid_types(client, valid_input_data, invalid_value, inference_spy):
    request = valid_input_data.copy()
    request["momentum_21"] = invalid_value
    assert client.post("/predict", json=request).status_code == 422
    inference_spy.assert_not_called()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("momentum_21", -1.01),
        ("momentum_63", -1.01),
        ("momentum_252_21", -1.01),
        ("volatility_63", -0.1),
        ("drawdown_252", 0.01),
        ("drawdown_252", -1.01),
        ("filing_age_days", -1),
    ],
)
def test_invalid_bounds(client, valid_input_data, field, invalid_value, inference_spy):
    request = valid_input_data.copy()
    request[field] = invalid_value
    assert client.post("/predict", json=request).status_code == 422
    inference_spy.assert_not_called()


def test_extra_field_and_wrong_feature_count(client, valid_input_data, inference_spy):
    request = valid_input_data | {"unknown_feature": 1.0}
    assert client.post("/predict", json=request).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json=list(valid_input_data.values())).status_code == 422
    inference_spy.assert_not_called()


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input(client, valid_input_data, invalid_value, inference_spy):
    request = valid_input_data.copy()
    request["net_margin"] = invalid_value
    response = client.post(
        "/predict", content=json.dumps(request), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    inference_spy.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail_prediction(_):
        raise RuntimeError("Deliberate inference failure")

    monkeypatch.setattr(app.state.model, "predict", fail_prediction)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}


def test_nonfinite_prediction_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.model, "predict", lambda _: np.array([np.nan]))
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    durations_ms = []
    for _ in range(30):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
        assert math.isfinite(response.json()["prediction"])

    report = {
        "method": "In-process FastAPI TestClient, sequential requests",
        "warmup_requests": 5,
        "measured_requests": len(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "model_version": MODEL_VERSION,
        "timing_budget": "Reported without a hardware-independent pass/fail cutoff",
    }
    assert all(duration > 0 for duration in durations_ms)
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
