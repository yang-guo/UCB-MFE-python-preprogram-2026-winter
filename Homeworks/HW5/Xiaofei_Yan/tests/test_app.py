"""Check the saved HW4 model, request validation, failures, and local latency."""

import json
import math
import statistics
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from hw4_pipeline import FEATURES

HOMEWORK_DIR = Path(__file__).resolve().parents[1]


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
    """HW4 feature row saved before the API was implemented."""
    return json.loads((HOMEWORK_DIR / "example_request.json").read_text())


@pytest.fixture
def hw4_reference():
    """Score recorded by HW4 for the same dated symbol and feature row."""
    return json.loads((HOMEWORK_DIR / "model_reference.json").read_text())


def test_predict_valid_input(client, valid_input_data, hw4_reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert type(body["prediction"]) is float
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == hw4_reference["model_version"]
    assert hw4_reference["features"] == FEATURES
    assert len(valid_input_data["features"]) == len(FEATURES)

    frame = pd.DataFrame([valid_input_data["features"]], columns=FEATURES)
    direct = float(client.app.state.model.predict(frame)[0])
    assert math.isclose(body["prediction"], direct, rel_tol=1e-12, abs_tol=1e-12)
    # The independent reference comes from HW4's saved scores_2026-06-30.csv.
    assert math.isclose(
        body["prediction"], hw4_reference["expected_prediction"], rel_tol=1e-7, abs_tol=1e-9
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "short",
        "long",
        "string",
        "boolean",
        "null",
        "object",
        "nan",
        "infinity",
        "negative_infinity",
        "bad_drawdown",
        "bad_return",
    ],
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, case):
    features = valid_input_data["features"].copy()
    body = {"features": features}
    if case == "missing":
        body = {}
    elif case == "extra":
        body["unexpected"] = 1
    elif case == "short":
        body["features"] = features[:-1]
    elif case == "long":
        body["features"] = [*features, 0.0]
    elif case == "string":
        features[0] = "0.05"
    elif case == "boolean":
        features[0] = True
    elif case == "null":
        features[0] = None
    elif case == "object":
        features[0] = {"value": 0.05}
    elif case == "nan":
        features[0] = float("nan")
    elif case == "infinity":
        features[0] = float("inf")
    elif case == "negative_infinity":
        features[0] = float("-inf")
    elif case == "bad_drawdown":
        features[FEATURES.index("drawdown_252")] = 0.01
    elif case == "bad_return":
        features[FEATURES.index("return_5")] = -2.0

    predict_spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(client.app.state.model, "predict", predict_spy)
    # Raw content allows NaN/Infinity to reach FastAPI's validator in this test.
    response = client.post(
        "/predict",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422, (case, response.text)
    assert predict_spy.call_count == 0


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    predict_spy = Mock(side_effect=RuntimeError("Deliberate model failure"))
    monkeypatch.setattr(failure_client.app.state.model, "predict", predict_spy)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    predict_spy.assert_called_once()
    assert "Deliberate model failure" in caplog.text
    assert "Deliberate model failure" not in response.text


def test_predict_performance(client, valid_input_data):
    """Measure in-process TestClient latency, including HTTP handling."""
    for _ in range(3):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(30):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median_ms = statistics.median(latencies_ms)
    p95_ms = float(np.percentile(latencies_ms, 95))
    assert len(latencies_ms) >= 20
    assert math.isfinite(median_ms) and math.isfinite(p95_ms)
    assert 0 < median_ms <= p95_ms
    print(
        f"TestClient latency over {len(latencies_ms)} requests: median={median_ms:.2f} ms, p95={p95_ms:.2f} ms"
    )
