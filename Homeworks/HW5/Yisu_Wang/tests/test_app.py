"""Check the real HW4 artifact, request validation, failure handling, and latency."""

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
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
    return json.loads((ROOT / "example_request.json").read_text())


def test_predict_valid_input(client, valid_input_data):
    reference = json.loads((ROOT / "tests/reference_prediction.json").read_text())
    direct = app.state.model.predict(valid_input_data["features"])
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version", "model_scores"}
    assert result["model_version"] == reference["model_version"]
    assert result["prediction"] == pytest.approx(reference["prediction"], abs=1e-7)
    assert result["prediction"] == pytest.approx(direct["model_01"], abs=1e-7)
    for name, expected in reference["model_scores"].items():
        assert result["model_scores"][name] == pytest.approx(expected, abs=1e-7)


def test_predict_invalid_input(client, valid_input_data, monkeypatch):
    called = []
    monkeypatch.setattr(app.state.model, "predict", lambda row: called.append(row))
    values = valid_input_data["features"]
    cases = [
        {},
        {"features": values[:-1]},
        {"features": ["1.0", *values[1:]]},
        {"features": [True, *values[1:]]},
        {"features": [float("nan"), *values[1:]]},
        {"features": [float("inf"), *values[1:]]},
        {"features": values, "extra": 1},
    ]
    for case in cases:
        response = client.post(
            "/predict",
            content=json.dumps(case),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
    assert not called


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(_):
        raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(app.state.model, "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "Simulated inference failure" not in response.text


def test_predict_performance(client, valid_input_data):
    client.post("/predict", json=valid_input_data)
    durations = []
    for _ in range(20):
        started = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations.append(1000 * (time.perf_counter() - started))
        assert response.status_code == 200
    print(
        f"TestClient on {platform.platform()}, {os.cpu_count()} logical CPUs: "
        f"median {statistics.median(durations):.2f} ms, "
        f"p95 {np.percentile(durations, 95):.2f} ms (20 requests)"
    )
