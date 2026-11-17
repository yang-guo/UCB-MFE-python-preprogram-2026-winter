"""Checks for the HW4 Ridge service."""

import json
import statistics
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import course_data  # noqa: F401
from app import FEATURE_ORDER, app

HERE = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((HERE / "example_request.json").read_text())
# Direct pipeline call on this row, recorded before wrapping it in the API.
# The row is symbol A on 2016-03-01 from the HW4 test-score panel.
REFERENCE_PREDICTION = -0.20290557377034335


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
    return dict(EXAMPLE)


def test_predict_valid_input(client, valid_input_data):
    saved = joblib.load(HERE / "ridge_90d_full.joblib")
    frame = pd.DataFrame([{name: valid_input_data[name] for name in FEATURE_ORDER}])
    direct = float(saved["pipeline"].predict(frame)[0])

    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == "ridge-alpha100-top2000-2010-2015"
    assert body["prediction"] == pytest.approx(direct, rel=1e-6, abs=1e-8)
    assert body["prediction"] == pytest.approx(REFERENCE_PREDICTION, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in EXAMPLE.items() if key != "sector"},
        {**EXAMPLE, "extra": 1},
        {**EXAMPLE, "momentum_21": "0.1"},
        {**EXAMPLE, "momentum_21": True},
        {**EXAMPLE, "volatility_63": -1},
        {**EXAMPLE, "filing_age_days": 800},
        {**EXAMPLE, "sector": ""},
        {**EXAMPLE, "sector": 3},
    ],
)
def test_predict_invalid_input(client, payload, monkeypatch):
    def fail_predict(_frame):
        raise AssertionError("invalid input reached inference")

    monkeypatch.setattr(client.app.state.pipeline, "predict", fail_predict)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def explode(_frame):
        raise RuntimeError("forced inference failure")

    monkeypatch.setattr(failure_client.app.state.pipeline, "predict", explode)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    assert client.post("/predict", json=valid_input_data).status_code == 200
    samples = []
    for _ in range(20):
        started = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        samples.append(time.perf_counter() - started)
        assert response.status_code == 200
    samples.sort()
    median = statistics.median(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    print(f"in-process TestClient latency median={median:.6f}s p95={p95:.6f}s n=20")
    # This budget is for an in-process TestClient on a laptop, not a deployed host.
    assert p95 < 0.5
