import json
import platform
from statistics import median, quantiles
from time import perf_counter
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURES, MODEL_VERSION, ROOT, app


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return json.loads((ROOT / "example_request.json").read_text())


def test_predict_valid_input(client, payload):
    reference = json.loads((ROOT / "artifacts/reference.json").read_text())
    features = pd.DataFrame([payload["features"]], columns=FEATURES)
    direct = app.state.model.predict(features)[0]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert result["model_version"] == MODEL_VERSION
    assert type(result["prediction"]) is float
    assert result["prediction"] == pytest.approx(direct, abs=1e-10, rel=0)
    assert result["prediction"] == pytest.approx(
        reference["expected_prediction"], abs=1e-10, rel=0
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"features": [0] * 6, "extra": 1},
        {"features": [0] * 5},
        {"features": [0] * 7},
        {"features": ["0", 0, 0, 0, 0, 0]},
        {"features": [True, 0, 0, 0, 0, 0]},
        {"features": [-1, 0, 0, 0, 0, 0]},
        {"features": [0, 0, 0, 0, 0, -1]},
    ],
)
def test_predict_invalid_input(client, payload, monkeypatch):
    inference = Mock()
    monkeypatch.setattr(app.state.model, "predict", inference)
    assert client.post("/predict", json=payload).status_code == 422
    inference.assert_not_called()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_input(client, value, monkeypatch):
    inference = Mock()
    monkeypatch.setattr(app.state.model, "predict", inference)
    response = client.post(
        "/predict",
        content=f'{{"features":[{value},0,0,0,0,0]}}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    inference.assert_not_called()


def test_predict_model_failure(client, payload, monkeypatch):
    inference = Mock(side_effect=RuntimeError("private failure detail"))
    monkeypatch.setattr(app.state.model, "predict", inference)
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Inference failed"}
    inference.assert_called_once()


def test_predict_performance(client, payload):
    for _ in range(5):
        assert client.post("/predict", json=payload).status_code == 200
    timings = []
    for _ in range(30):
        start = perf_counter()
        response = client.post("/predict", json=payload)
        timings.append((perf_counter() - start) * 1000)
        assert response.status_code == 200
    p95 = quantiles(timings, n=100, method="inclusive")[94]
    print(f"{platform.platform()} | TestClient, n=30 | median {median(timings):.3f} ms")
    print(f"p95 {p95:.3f} ms")
