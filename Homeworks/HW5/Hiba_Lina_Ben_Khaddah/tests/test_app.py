"""Tests for the HW5 prediction service."""

import json
import math
import statistics
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parents[1]


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
    reference = json.loads((ROOT / "artifacts/reference_prediction.json").read_text())
    return reference["input"]


def test_predict_valid_input(client, valid_input_data):
    reference = json.loads((ROOT / "artifacts/reference_prediction.json").read_text())
    columns = client.app.state.feature_order
    row = pd.DataFrame([[valid_input_data[name] for name in columns]], columns=columns)
    direct_prediction = float(client.app.state.model.predict(row)[0])

    response = client.post("/predict", json=valid_input_data)
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"prediction", "model_version"}
    assert math.isfinite(body["prediction"])
    assert body["prediction"] == pytest.approx(direct_prediction)
    assert body["prediction"] == pytest.approx(reference["expected_prediction"])
    assert body["model_version"] == reference["model_version"]


def test_predict_invalid_input(client, valid_input_data, monkeypatch):
    inference_called = False

    def track_inference(_row):
        nonlocal inference_called
        inference_called = True
        return [0.0]

    monkeypatch.setattr(client.app.state.model, "predict", track_inference)

    missing = valid_input_data.copy()
    missing.pop("momentum_21")
    extra = {**valid_input_data, "unexpected_feature": 1.0}
    numeric_string = {**valid_input_data, "momentum_21": "0.1"}
    boolean = {**valid_input_data, "momentum_21": True}
    outside_range = {**valid_input_data, "volatility_63": -0.1}
    unknown_sector = {**valid_input_data, "sector": "Not a sector"}
    nonfinite = {**valid_input_data, "momentum_21": float("nan")}

    for payload in [
        missing,
        extra,
        numeric_string,
        boolean,
        outside_range,
        unknown_sector,
        nonfinite,
    ]:
        response = client.post(
            "/predict",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    assert inference_called is False


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def raise_error(_row):
        raise RuntimeError("Deliberate test failure")

    monkeypatch.setattr(failure_client.app.state.model, "predict", raise_error)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}


def test_predict_performance(client, valid_input_data):
    assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(25):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1_000)
        assert response.status_code == 200

    median_ms = statistics.median(latencies_ms)
    p95_ms = sorted(latencies_ms)[math.ceil(0.95 * len(latencies_ms)) - 1]
    print(f"TestClient latency: median={median_ms:.2f} ms, p95={p95_ms:.2f} ms")

    assert math.isfinite(median_ms)
    assert math.isfinite(p95_ms)
