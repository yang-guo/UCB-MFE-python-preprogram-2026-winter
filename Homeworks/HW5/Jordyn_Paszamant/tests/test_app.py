"""Exercise the API against the saved HW4 pipeline."""

import json
import math
import platform
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

TEST_DIR = Path(__file__).resolve().parent


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
    return json.loads((TEST_DIR.parent / "example_request.json").read_text())


def test_predict_valid_input(client, valid_input_data):
    reference = json.loads((TEST_DIR / "hw4_reference.json").read_text())
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert result["model_version"] == "hw4-rf-2016-04-01"
    assert type(result["prediction"]) is float
    assert math.isfinite(result["prediction"])

    pipeline = app.state.model
    training_columns = pipeline.named_steps["forest"].feature_names_in_
    direct = pipeline.predict(pd.DataFrame([valid_input_data], columns=training_columns)).item()

    assert result["prediction"] == pytest.approx(direct, rel=0, abs=1e-12)
    assert result["prediction"] == pytest.approx(reference["prediction"], rel=0, abs=1e-10)
    assert round(result["prediction"], 4) == reference["notebook_prediction_4dp"]


@pytest.mark.parametrize(
    ("removed", "updates"),
    [
        ("momentum_21", {}),
        (None, {"symbol": "EGRX"}),
        (None, {"momentum_21": "0.10"}),
        (None, {"momentum_21": True}),
        (None, {"momentum_21": None}),
        (None, {"momentum_21": [0.1, 0.2]}),
        (None, {"momentum_21": -1.01}),
        (None, {"volatility_21": -0.01}),
        (None, {"ma_ratio_50": 49.01}),
        (None, {"ma_ratio_200": 199.01}),
        (None, {"momentum_21": float("nan")}),
        (None, {"momentum_21": float("inf")}),
        (None, {"momentum_21": float("-inf")}),
    ],
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, removed, updates):
    predict = Mock()
    monkeypatch.setattr(app.state.model, "predict", predict)
    if removed:
        del valid_input_data[removed]
    valid_input_data.update(updates)
    # Send raw JSON to exercise non-finite numbers the HTTP client rejects by default.
    response = client.post(
        "/predict",
        content=json.dumps(valid_input_data),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    predict.assert_not_called()


def test_predict_rejects_wrong_feature_shape(client, monkeypatch):
    predict = Mock()
    monkeypatch.setattr(app.state.model, "predict", predict)
    response = client.post("/predict", json=[0.1] * 7)
    assert response.status_code == 422
    predict.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    predict = Mock(side_effect=RuntimeError("private inference details"))
    monkeypatch.setattr(app.state.model, "predict", predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    predict.assert_called_once()


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    durations_ms = []
    for _ in range(30):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations_ms.append(1000 * (perf_counter() - start))
        assert response.status_code == 200

    p95 = quantiles(durations_ms, n=100, method="inclusive")[94]
    print(
        f"TestClient on {platform.system()} {platform.machine()}, Python "
        f"{platform.python_version()}, n={len(durations_ms)}: "
        f"median {median(durations_ms):.2f} ms, p95 {p95:.2f} ms"
    )
