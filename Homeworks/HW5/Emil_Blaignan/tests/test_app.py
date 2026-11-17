"""Replace each TODO failure with meaningful checks for your application."""

import json
import math
import time
from pathlib import Path
from statistics import median

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def reference():
    path = Path(__file__).resolve().parents[1] / "reference.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def valid_input_data(reference):
    # Return a JSON-compatible dict matching your documented input contract.
    return reference["input"].copy()
    # raise NotImplementedError("Supply valid input data")


def test_predict_valid_input(client, valid_input_data, reference):
    # Use the real saved model. Check schema, direct/API agreement, known result.
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200

    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert isinstance(result["prediction"], float)
    assert math.isfinite(result["prediction"])
    assert isinstance(result["model_version"], str)
    assert result["model_version"] == reference["model_version"]

    row = pd.DataFrame([valid_input_data], columns=app.state.feature_names)
    direct_prediction = float(app.state.pipeline.predict(row)[0])

    assert result["prediction"] == pytest.approx(direct_prediction, abs=1e-12)
    assert result["prediction"] == pytest.approx(reference["expected_prediction"], abs=1e-12)
    # raise NotImplementedError("Implement the valid prediction test")


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "wrong_type",
        "bool_value",
        "null_value",
        "nan_value",
        "inf_value",
        "out_of_bounds",
        "wrong_count",
        "list_body",
    ],
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, case):
    # Parametrize missing/extra fields, types, bounds, and feature-count cases.
    # Confirm rejected requests do not call the model.
    payload = valid_input_data

    if case == "missing":
        del payload["momentum_21"]
    elif case == "extra":
        payload["unknown_feature"] = 0.0
    elif case == "wrong_type":
        payload["momentum_21"] = "0.08"
    elif case == "bool_value":
        payload["momentum_21"] = True
    elif case == "null_value":
        payload["momentum_21"] = None
    elif case == "nan_value":
        payload["momentum_21"] = float("nan")
    elif case == "inf_value":
        payload["volatility_63"] = float("inf")
    elif case == "out_of_bounds":
        payload["drawdown_252"] = 0.1
    elif case == "wrong_count":
        payload = {"momentum_21": payload["momentum_21"]}
    elif case == "list_body":
        payload = list(payload.values())

    calls = []

    def unexpected_predict(row):
        calls.append(row)
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.pipeline, "predict", unexpected_predict)

    # json.dumps writes NaN/Infinity; client.post(json=...) refuses to, so send the body as text.
    response = client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert calls == []  # validation rejected the request before inference
    # raise NotImplementedError("Implement input validation tests")


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    # Patch predict on the loaded model to raise, send VALID input, assert 500.
    # Do not assume a JSON error body unless your application defines one.
    def broken_predict(row):
        raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(app.state.pipeline, "predict", broken_predict)

    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    # Details stay in the server log, and no prediction is returned.
    assert "Simulated" not in response.text
    assert "prediction" not in response.text
    # raise NotImplementedError("Implement the inference failure test")


def test_predict_performance(client, valid_input_data):
    # Warm up, then measure >=20 successful requests with time.perf_counter.
    # Report median and p95; document an environment-specific budget if used.
    for _ in range(3):
        response = client.post("/predict", json=valid_input_data)
        assert response.status_code == 200

    durations_ms = []

    for _ in range(20):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        durations_ms.append(elapsed * 1000)

    durations_ms.sort()
    p95_index = math.ceil(0.95 * len(durations_ms)) - 1
    p95 = durations_ms[p95_index]

    # Method: in-process TestClient (no network), 3 warm-up then 20 timed requests, perf_counter.
    print(f"Median: {median(durations_ms):.2f} ms, p95: {p95:.2f} ms")
    # raise NotImplementedError("Implement and report the latency measurement")
