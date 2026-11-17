"""Tests for the HW5 service. They use the real saved HW4 model, except where
a test replaces its predict method on purpose."""

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
from model_service import FEATURES

ROOT = Path(__file__).resolve().parents[1]
# The model is a 20-term sum of numbers below 1, so a different Python or BLAS
# build can only differ in the last few bits. 1e-12 is far above that and far
# below any real change in the model.
TOLERANCE = 1e-12
# 50 ms leaves room for a slow machine but still fails if the model were
# reloaded or refit on every request. See the README for the measured numbers.
MEDIAN_BUDGET_MS = 50


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
    # A real HW4 input: AAPL on 2026-03-27, the first week this fit was used.
    return json.loads((ROOT / "example_request.json").read_text())


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == "hw4-ridge-rank-fit-2026-03-27"

    # The API agrees with calling the saved model directly, as HW4 did.
    frame = pd.DataFrame([valid_input_data])[list(FEATURES)]
    direct = float(app.state.bundle["model"].predict(frame)[0])
    assert body["prediction"] == pytest.approx(direct, abs=TOLERANCE)

    # The API agrees with predictions recorded from HW4's research.fit_predict
    # by export_model.py, before the model was wrapped in the API.
    cases = json.loads((ROOT / "artifacts" / "reference_predictions.json").read_text())
    assert cases[0]["features"] == valid_input_data
    for case in cases:
        response = client.post("/predict", json=case["features"])
        assert response.status_code == 200, case["symbol"]
        assert response.json()["prediction"] == pytest.approx(
            case["hw4_prediction"], abs=TOLERANCE
        )


def test_predict_invalid_input(client, valid_input_data, monkeypatch):
    spy = Mock()
    monkeypatch.setattr(app.state.bundle["model"], "predict", spy)
    valid = valid_input_data
    payloads = {
        "missing field": {k: v for k, v in valid.items() if k != "roa"},
        "extra field": {**valid, "log_size": 0.1},
        "numeric string": {**valid, "mom_1m": "0.1"},
        "boolean": {**valid, "mom_1m": True},
        "null": {**valid, "mom_1m": None},
        "above range": {**valid, "vol_3m": 0.51},
        "below range": {**valid, "vol_3m": -0.51},
        "raw value instead of a rank": {**valid, "log_dollar_volume": 18.2},
        "empty body": {},
        "list of 20 numbers": list(valid.values()),
        "list of 19 numbers": list(valid.values())[:19],
    }
    bodies = {name: json.dumps(payload) for name, payload in payloads.items()}
    # NaN and Infinity are not legal JSON, so these bodies are written by hand.
    for token in ["NaN", "Infinity", "-Infinity", "1e999"]:
        bodies[token] = json.dumps(valid).replace(json.dumps(valid["mom_1m"]), token, 1)

    for name, body in bodies.items():
        response = client.post(
            "/predict", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422, name
    assert spy.call_count == 0, "invalid input reached the model"


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    failing_predict = Mock(side_effect=RuntimeError("simulated failure with secret detail"))
    monkeypatch.setattr(app.state.bundle["model"], "predict", failing_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert failing_predict.call_count == 1
    assert "prediction" not in response.text
    assert "secret detail" not in response.text  # details stay in the server log


def test_predict_performance(client, valid_input_data):
    # In-process TestClient timing: no network and no separate server process.
    for _ in range(10):  # warm up imports, routing and the first pandas call
        assert client.post("/predict", json=valid_input_data).status_code == 200
    latencies_ms = []
    for _ in range(200):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[-1]
    print(
        f"\nlatency over {len(latencies_ms)} in-process requests: median {median:.2f} ms, p95 {p95:.2f} ms"
    )
    assert median < MEDIAN_BUDGET_MS
