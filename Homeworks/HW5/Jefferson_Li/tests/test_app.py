"""Test the HW4 prediction service."""

import json
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

EXAMPLE_REQUEST = Path(__file__).resolve().parents[1] / "example_request.json"
HW4_SKY_PREDICTION = 0.5395505648644271


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
    return json.loads(EXAMPLE_REQUEST.read_text())


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == "hw4-ridge-v1"
    assert type(body["prediction"]) in (float, int)
    assert isfinite(body["prediction"])

    features = pd.DataFrame([valid_input_data], columns=["momentum_63", "volatility_63"])
    direct = app.state.model.predict(features)[0]
    assert body["prediction"] == pytest.approx(direct, rel=1e-12, abs=1e-12)
    assert body["prediction"] == pytest.approx(HW4_SKY_PREDICTION, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="empty-object"),
        pytest.param({"volatility_63": 0.3}, id="missing-momentum"),
        pytest.param({"momentum_63": 0.2}, id="missing-volatility"),
        pytest.param({"momentum_63": 0.2, "volatility_63": 0.3, "symbol": "SKY"}, id="extra"),
        pytest.param([0.2], id="wrong-feature-count"),
        pytest.param({"momentum_63": "0.2", "volatility_63": 0.3}, id="numeric-string"),
        pytest.param({"momentum_63": 0.2, "volatility_63": True}, id="boolean"),
        pytest.param({"momentum_63": None, "volatility_63": 0.3}, id="null"),
        pytest.param({"momentum_63": -1, "volatility_63": 0.3}, id="momentum-bound"),
        pytest.param({"momentum_63": 0.2, "volatility_63": -0.1}, id="volatility-bound"),
        pytest.param({"momentum_63": float("nan"), "volatility_63": 0.3}, id="nan"),
        pytest.param({"momentum_63": 0.2, "volatility_63": float("inf")}, id="infinity"),
        pytest.param(
            {"momentum_63": float("-inf"), "volatility_63": 0.3}, id="negative-infinity"
        ),
    ],
)
def test_predict_invalid_input(client, payload, monkeypatch):
    predict = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.model, "predict", predict)
    # Send the raw body so non-finite values reach the server's validation.
    response = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    predict.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    predict = Mock(side_effect=RuntimeError("Simulated inference failure"))
    monkeypatch.setattr(app.state.model, "predict", predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    predict.assert_called_once()


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(50):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((perf_counter() - start) * 1000)
        assert response.status_code == 200

    p95 = sorted(latencies_ms)[ceil(0.95 * len(latencies_ms)) - 1]
    print(f"Latency (ms): median {median(latencies_ms):.3f}, p95 {p95:.3f}")
