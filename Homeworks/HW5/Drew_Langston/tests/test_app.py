"""HW5 tests, adapted from the homework starter and Lecture 9 demo."""

import json
import math
import os
import platform
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import OutputData, app

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "artifacts/reference_prediction.json").read_text())
NUMERIC_FIELDS = list(REFERENCE["input"])[:-1]


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
    return REFERENCE["input"].copy()


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    result = OutputData.model_validate(body)
    assert math.isfinite(result.prediction)
    assert result.model_version == REFERENCE["model_version"]

    bundle = app.state.bundle
    frame = pd.DataFrame([valid_input_data], columns=bundle["features"])
    direct = float(bundle["pipeline"].predict(frame)[0])

    assert result.prediction == pytest.approx(direct, rel=1e-10, abs=1e-12)
    assert result.prediction == pytest.approx(
        REFERENCE["expected_prediction"], rel=1e-10, abs=1e-12
    )


@pytest.mark.parametrize("field", list(REFERENCE["input"]))
def test_missing_fields_rejected(client, valid_input_data, monkeypatch, field):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    del valid_input_data[field]
    assert client.post("/predict", json=valid_input_data).status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize("value", ["0.2", True, [], {}])
def test_numeric_types_rejected(client, valid_input_data, monkeypatch, field, value):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    valid_input_data[field] = value
    assert client.post("/predict", json=valid_input_data).status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_rejected(client, valid_input_data, monkeypatch, field, value):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    valid_input_data[field] = value
    response = client.post(
        "/predict",
        content=json.dumps(valid_input_data),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"extra_feature": 1},
        {"momentum_126": -1.1},
        {"momentum_252_21": -1.1},
        {"volatility_63": -1},
        {"volatility_63": None},
        {"sector": 12},
        {"sector": None},
        {"sector": ""},
        {"sector": "x" * 101},
    ],
)
def test_other_invalid_inputs_rejected(client, valid_input_data, monkeypatch, changes):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    valid_input_data.update(changes)
    assert client.post("/predict", json=valid_input_data).status_code == 422
    spy.assert_not_called()


def test_wrong_input_shape_rejected(client, valid_input_data, monkeypatch):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    response = client.post("/predict", json=list(valid_input_data.values()))
    assert response.status_code == 422
    spy.assert_not_called()


def test_null_features_use_saved_imputer(client, valid_input_data):
    for field in NUMERIC_FIELDS:
        if field != "volatility_63":
            valid_input_data[field] = None

    bundle = app.state.bundle
    frame = pd.DataFrame([valid_input_data], columns=bundle["features"])
    frame[NUMERIC_FIELDS] = np.nan
    frame["volatility_63"] = valid_input_data["volatility_63"]
    expected = float(bundle["pipeline"].predict(frame)[0])

    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    spy = Mock(side_effect=RuntimeError("Simulated private inference error"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    spy.assert_called_once()


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    durations_ms = []
    for _ in range(30):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations_ms.append((perf_counter() - start) * 1000)
        assert response.status_code == 200
        assert response.json()["prediction"] == pytest.approx(
            REFERENCE["expected_prediction"], rel=1e-10, abs=1e-12
        )

    report = {
        "method": "Sequential in-process TestClient; time.perf_counter",
        "warmup_requests": 5,
        "measured_requests": len(durations_ms),
        "median_ms": float(np.median(durations_ms)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "budget": "Reported without an arbitrary latency cutoff",
    }
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nLatency report:", json.dumps(report, indent=2))
