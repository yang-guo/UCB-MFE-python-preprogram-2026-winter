"""Contract, model-parity, inference-failure, and latency checks."""

import json
import statistics
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURES, MODEL_VERSION, app

SUBMISSION = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((SUBMISSION / "example_request.json").read_text())
REFERENCE = json.loads((SUBMISSION / "artifacts/reference_prediction.json").read_text())


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
    return deepcopy(EXAMPLE)


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"prediction", "model_version"}
    assert payload["model_version"] == MODEL_VERSION
    assert np.isfinite(payload["prediction"])

    artifact = joblib.load(SUBMISSION / "artifacts/hw4_ridge_pipeline.joblib")
    frame = pd.DataFrame([valid_input_data], columns=FEATURES)
    direct = float(np.asarray(artifact.predict(frame)).reshape(-1)[0])
    assert payload["prediction"] == pytest.approx(direct, rel=1e-12, abs=1e-12)
    assert payload["prediction"] == pytest.approx(REFERENCE["prediction"], rel=1e-10, abs=1e-10)


def test_predict_imputes_nullable_feature(client, valid_input_data):
    valid_input_data["net_margin"] = None
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert np.isfinite(response.json()["prediction"])


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_feature",
        "extra_feature",
        "numeric_string",
        "boolean_numeric",
        "non_finite_nan",
        "non_finite_infinity",
        "out_of_range",
        "invalid_sector_type",
        "wrong_feature_count_list",
    ],
)
def test_predict_invalid_input_does_not_reach_model(
    client, valid_input_data, monkeypatch, invalid_case
):
    model = client.app.state.model
    original_predict = model.predict
    calls = 0

    def observed_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", observed_predict)
    body = deepcopy(valid_input_data)
    if invalid_case == "missing_feature":
        del body["momentum_21"]
    elif invalid_case == "extra_feature":
        body["future_return"] = 0.1
    elif invalid_case == "numeric_string":
        body["momentum_21"] = "0.12"
    elif invalid_case == "boolean_numeric":
        body["momentum_21"] = True
    elif invalid_case == "non_finite_nan":
        body["momentum_21"] = float("nan")
    elif invalid_case == "non_finite_infinity":
        body["momentum_21"] = float("inf")
    elif invalid_case == "out_of_range":
        body["filing_age_days"] = -1
    elif invalid_case == "invalid_sector_type":
        body["sector"] = 17
    elif invalid_case == "wrong_feature_count_list":
        body = [0.0] * (len(FEATURES) - 1)

    response = client.post(
        "/predict",
        content=json.dumps(body, allow_nan=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert calls == 0


def test_predict_model_failure_returns_generic_500(
    failure_client, valid_input_data, monkeypatch
):
    def fail_prediction(_frame):
        raise RuntimeError("private internal detail")

    monkeypatch.setattr(failure_client.app.state.model, "predict", fail_prediction)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "private internal detail" not in response.text


def test_predict_performance(client, valid_input_data):
    warmup = client.post("/predict", json=valid_input_data)
    assert warmup.status_code == 200
    durations = []
    for _ in range(30):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median_ms = statistics.median(durations)
    p95_ms = float(np.percentile(durations, 95, method="linear"))
    print(
        f"In-process TestClient latency, 30 requests after one warm-up: "
        f"median={median_ms:.3f} ms, p95={p95_ms:.3f} ms"
    )
