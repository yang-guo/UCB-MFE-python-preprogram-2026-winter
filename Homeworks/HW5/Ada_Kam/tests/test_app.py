"""Replace each TODO failure with meaningful checks for your application."""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURES, app

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


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
    # Return a JSON-compatible dict matching your documented input contract.
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def metadata():
    return json.loads((ARTIFACTS / "model_metadata.json").read_text())


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Replace the loaded model's predict with a recorder, to prove it is never reached."""
    calls = []

    def spy(X):
        calls.append(X)
        raise AssertionError("invalid input reached the model")

    monkeypatch.setattr(client.app.state.bundle["pipeline"], "predict", spy)
    return calls


def test_predict_valid_input(client, valid_input_data, metadata):
    # Use the real saved model, check schema, direct/API agreement, known result
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == metadata["model_version"] == "hw4-hgb-rank-v1"

    # Call saved pipeline
    pipeline = joblib.load(ARTIFACTS / "model.joblib")["pipeline"]
    direct = float(pipeline.predict(pd.DataFrame([valid_input_data], columns=FEATURES))[0])
    assert body["prediction"] == pytest.approx(direct, rel=0, abs=1e-12)

    # Known result in export_model.py right after fitting HW4 model: 03_backtest.ipynb
    # Same library versions for deterministic behavior
    assert metadata["reference_input"] == valid_input_data
    assert body["prediction"] == pytest.approx(metadata["reference_prediction"], abs=1e-9)


def test_predict_accepts_null_fundamental(client, valid_input_data):
    # Fundamentals may be null (imputed with the training median).
    response = client.post("/predict", json={**valid_input_data, "book_to_px": None})
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def _drop(d, key):
    return {k: v for k, v in d.items() if k != key}


INVALID_CASES = {
    "missing_field": lambda d: _drop(d, "roe"),
    "extra_field": lambda d: {**d, "sector": "Tech"},
    "numeric_string": lambda d: {**d, "rsi_21": "50"},
    "boolean": lambda d: {**d, "dma_20": True},
    "null_technical": lambda d: {**d, "rsi_21": None},
    "rsi_above_100": lambda d: {**d, "rsi_21": 150.0},
    "return_below_minus_1": lambda d: {**d, "momentum_252_21": -1.5},
    "negative_volatility": lambda d: {**d, "volatility_63": -0.1},
    "too_few_features": lambda d: dict(list(d.items())[:11]),
    "too_many_features": lambda d: {**d, "feature_13": 0.0},
    "list_instead_of_object": lambda d: list(d.values()),
    "empty_body": lambda d: {},
}


@pytest.mark.parametrize("case", INVALID_CASES)
def test_predict_invalid_input(client, valid_input_data, predict_spy, case):
    # Parametrize missing/extra fields, types, bounds, and feature-count cases.
    # Confirm rejected requests do not call the model.
    response = client.post("/predict", json=INVALID_CASES[case](valid_input_data))
    assert response.status_code == 422
    assert predict_spy == []


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite(client, valid_input_data, predict_spy, value):
    # Standard JSON has no NaN/Infinity, but Python's parser accepts them, so send raw text.
    body = json.dumps(valid_input_data).replace(json.dumps(valid_input_data["dma_20"]), value)
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert predict_spy == []


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    # Patch predict on the loaded model to raise, send VALID input, assert 500.
    # Do not assume a JSON error body unless your application defines one.
    def broken_predict(X):
        raise RuntimeError("simulated model failure")

    pipeline = failure_client.app.state.bundle["pipeline"]
    monkeypatch.setattr(pipeline, "predict", broken_predict)

    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "simulated model failure" not in response.text
    assert "prediction" not in response.text


def test_predict_performance(client, valid_input_data):
    # Warm up, then measure >=20 successful requests with time.perf_counter.
    # Report median and p95; document an environment-specific budget if used.
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    timings_ms = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(timings_ms)
    p95 = statistics.quantiles(timings_ms, n=20)[-1]
    print(
        f"\nIn-process TestClient latency, n={len(timings_ms)} after 5 warm-ups: "
        f"median={median:.1f} ms, p95={p95:.1f} ms "
        f"({platform.processor() or platform.machine()}, {platform.system()}, "
        f"Python {platform.python_version()})"
    )

    assert p95 < 250
