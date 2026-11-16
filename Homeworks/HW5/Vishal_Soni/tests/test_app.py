"""Contract, model equivalence, error handling, and latency checks."""

import json
import math
import statistics
import time
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from model_contract import FEATURES, MODEL_VERSION

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "reference_prediction.json").read_text())


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
    return REFERENCE["features"].copy()


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert result["model_version"] == MODEL_VERSION
    assert isinstance(result["prediction"], float)
    assert math.isfinite(result["prediction"])
    direct = app.state.model.predict(pd.DataFrame([valid_input_data], columns=FEATURES))[0]
    assert result["prediction"] == pytest.approx(direct, rel=1e-12, abs=1e-12)
    assert result["prediction"] == pytest.approx(REFERENCE["prediction"], rel=1e-10, abs=1e-10)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("mom21"),
        lambda d: d.update(extra=0),
        lambda d: d.update(mom21="0.1"),
        lambda d: d.update(mom63=True),
        lambda d: d.update(mom252_21=-1.01),
        lambda d: d.update(vol63=-0.01),
        lambda d: d.update(mom21=None),
        lambda d: d.update(mom21=[0.1, 0.2]),
        lambda d: d.update(features=[0.1, 0.2, 0.3]),
    ],
)
def test_invalid_input_does_not_reach_model(client, valid_input_data, monkeypatch, change):
    fake = Mock(side_effect=AssertionError("invalid input reached inference"))
    monkeypatch.setattr(app.state.model, "predict", fake)
    change(valid_input_data)
    assert client.post("/predict", json=valid_input_data).status_code == 422
    fake.assert_not_called()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_numbers_rejected(client, valid_input_data, monkeypatch, value):
    fake = Mock(side_effect=AssertionError("invalid input reached inference"))
    monkeypatch.setattr(app.state.model, "predict", fake)
    body = json.dumps(valid_input_data).replace(str(valid_input_data["mom21"]), value)
    assert (
        client.post(
            "/predict", content=body, headers={"content-type": "application/json"}
        ).status_code
        == 422
    )
    fake.assert_not_called()


def test_model_failure(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(
        app.state.model, "predict", Mock(side_effect=RuntimeError("private detail"))
    )
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "private detail" not in response.text


def test_nonfinite_prediction_is_failure(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.model, "predict", Mock(return_value=[float("nan")]))
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    times = []
    for _ in range(30):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        times.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    times.sort()
    print(f"TestClient latency (ms): median={statistics.median(times):.3f} p95={times[28]:.3f}")
