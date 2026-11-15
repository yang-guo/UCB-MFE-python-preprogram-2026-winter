"""HW5 tests. `python -m pytest -q -s` also prints the latency report."""

import json
import math
import os
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import MODEL_PATH, app

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = {"Content-Type": "application/json"}
# HW4 predict_scores output for example_request.json (AAPL on 2026-09-29), printed by
# export_model.py before the model was wrapped in the API.
HW4_PREDICTION = 0.014140782632866819


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
    with open(os.path.join(HERE, "..", "example_request.json")) as f:
        return json.load(f)


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == "hw4-v2-boosting-fit-2026-01-02"

    # Same library versions as HW4, so only float rounding could differ.
    direct = joblib.load(MODEL_PATH).predict(pd.DataFrame([valid_input_data]))[0]
    assert body["prediction"] == pytest.approx(direct, rel=0, abs=1e-9)
    assert body["prediction"] == pytest.approx(HW4_PREDICTION, rel=0, abs=1e-9)


INVALID_CASES = {
    "missing field (7 features)": lambda body: {k: v for k, v in body.items() if k != "roa"},
    "extra field (9 features)": lambda body: {**body, "ret_6m": 0.1},
    "numeric string": lambda body: {**body, "ret_1m": "0.1"},
    "boolean": lambda body: {**body, "ret_1m": True},
    "NaN": lambda body: {**body, "vol_3m": float("nan")},
    "infinity": lambda body: {**body, "vol_3m": float("inf")},
    "above 0.5": lambda body: {**body, "size": 0.51},
    "below -0.5": lambda body: {**body, "size": -0.51},
}


@pytest.mark.parametrize("case", INVALID_CASES)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, case):
    calls = []
    monkeypatch.setattr(app.state.model, "predict", lambda X: calls.append(X))
    body = json.dumps(INVALID_CASES[case](valid_input_data))  # writes NaN and Infinity as-is
    response = client.post("/predict", content=body, headers=JSON)
    assert response.status_code == 422
    assert calls == []  # never reached the model


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.model, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    # In-process TestClient (no network).
    for _ in range(10):
        client.post("/predict", json=valid_input_data)
    latencies_ms = []
    for _ in range(200):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median = np.median(latencies_ms)
    p95 = np.percentile(latencies_ms, 95)
    print(f"\n200 requests via TestClient: median {median:.2f} ms, p95 {p95:.2f} ms")
