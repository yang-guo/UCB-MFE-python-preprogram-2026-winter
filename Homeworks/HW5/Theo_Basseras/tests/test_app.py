"""Check predictions, input validation, model failures and latency."""

import json
import platform
import time
from pathlib import Path

import numpy as np
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
def valid_input_data():
    hw5_dir = Path(__file__).resolve().parent.parent
    reference = json.loads((hw5_dir / "hw4_reference.json").read_text(encoding="utf-8"))

    features = reference["features"]
    return features


def test_predict_valid_input(client, valid_input_data):
    # We use client to act like a caller sending a real request.
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200

    result = response.json()
    assert set(result) == {"prediction", "model_version"}

    # We compare with a direct call to our fitted model
    features = ["momentum_63", "volatility_63"]
    data_features = pd.DataFrame([valid_input_data])[features]
    prediction = app.state.model_full.predict(data_features)[0]

    assert result["prediction"] == pytest.approx(prediction, abs=1e-12)

    # We also compare with the reference recorded from HW4
    hw5_dir = Path(__file__).resolve().parent.parent
    reference = json.loads((hw5_dir / "hw4_reference.json").read_text(encoding="utf-8"))

    assert result["prediction"] == pytest.approx(reference["prediction"], abs=1e-12)
    assert result["model_version"] == reference["model_version"]


def test_predict_invalid_input(client, valid_input_data, monkeypatch):
    # We make the test fail if invalid input reaches the model
    def should_not_predict(*args, **kwargs):
        raise AssertionError("Invalid input reached the model")

    monkeypatch.setattr(app.state.model_full, "predict", should_not_predict)

    invalid_inputs = [
        {},
        [],
        [0.1, 0.2],
        {"momentum_63": -1, "volatility_63": 0.2},
        {"momentum_63": 0.1, "volatility_63": -0.2},
        {"momentum_63": 5, "volatility_63": 1e10},
        {"momentum_63": 0.1, "volatility_63": 0.2, "extra": 1},
    ]

    # We check missing fields and invalid values for each feature
    for feature in valid_input_data:
        missing_field = valid_input_data.copy()
        missing_field.pop(feature)
        invalid_inputs.append(missing_field)

        for value in ["0.1", True, None, float("nan"), float("inf"), float("-inf")]:
            invalid_data = valid_input_data.copy()
            invalid_data[feature] = value
            invalid_inputs.append(invalid_data)

    # We send each invalid request
    for invalid_data in invalid_inputs:
        response = client.post(
            "/predict",
            content=json.dumps(invalid_data),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, response.text


@pytest.mark.parametrize("momentum, expected_status", [(10.0, 200), (10.000001, 422)])
def test_predict_momentum_limit(client, valid_input_data, momentum, expected_status):
    # We accept the upper limit and reject values above it
    data = valid_input_data.copy()
    data["momentum_63"] = momentum
    response = client.post("/predict", json=data)
    assert response.status_code == expected_status


@pytest.mark.parametrize("volatility, expected_status", [(10.0, 200), (10.000001, 422)])
def test_predict_volatility_limit(client, valid_input_data, volatility, expected_status):
    # We accept the upper limit and reject values above it
    data = valid_input_data.copy()
    data["volatility_63"] = volatility
    response = client.post("/predict", json=data)
    assert response.status_code == expected_status


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    # We simulate an unexpected failure inside the model
    def broken_predict(*args, **kwargs):
        raise RuntimeError("Model failed")

    monkeypatch.setattr(app.state.model_full, "predict", broken_predict)

    # We send valid input so it passes validation and reaches the model
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    # We warm up the API before measuring, 7 is arbitrary but should be enough to load the model and any caches.
    for i in range(7):
        response = client.post("/predict", json=valid_input_data)
        assert response.status_code == 200

    # We measure 30 successful requests
    durations = []

    for i in range(30):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        duration = time.perf_counter() - start

        assert response.status_code == 200
        durations.append(duration * 1000)

    # We report the times in milliseconds
    median = np.median(durations)
    p95 = np.percentile(durations, 95)

    print(f"Median: {median:.3f} ms")
    print(f"P95: {p95:.3f} ms")
    print("Method: in-process TestClient, 7 warm-up requests, 30 measured requests")
    print(f"System: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
