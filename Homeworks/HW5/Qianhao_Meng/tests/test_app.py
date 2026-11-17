"""Tests for the HW5 service. Run `python -m pytest -q` from the submission folder."""

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parents[1]
# Recorded by export_model.py from HW4's own scoring code, before the model was wrapped in the API.
REFERENCE = json.loads((ROOT / "artifacts" / "reference_prediction.json").read_text())
EXAMPLE = json.loads((ROOT / "example_request.json").read_text())
MODEL_VERSION = "hw4-ridge-ranks-20141231-v1"
JSON = {"Content-Type": "application/json"}
# In-process TestClient budget for a four-feature Ridge on a laptop. Typical medians are a few ms.
LATENCY_BUDGET_MS = 50


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
    return dict(REFERENCE["input"])


def raw_body(**overrides):
    """JSON text of the example request, with some values replaced by raw JSON tokens."""
    fields = {**{name: json.dumps(value) for name, value in EXAMPLE.items()}, **overrides}
    return "{" + ", ".join(f'"{name}": {token}' for name, token in fields.items()) + "}"


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == MODEL_VERSION
    assert np.isfinite(body["prediction"])
    assert response.headers["X-Request-ID"]
    # Same artifact, same scikit-learn: the API must reproduce a direct call to the saved
    # estimator, made here without the service wrapper, columns in training order.
    bundle = app.state.bundle
    frame = pd.DataFrame([valid_input_data])[bundle["features"]]
    direct = float(bundle["model"].predict(frame)[0])
    assert body["prediction"] == pytest.approx(direct, abs=1e-12)
    # Compare with the recorded HW4 result, allowing small floating-point differences.
    assert body["prediction"] == pytest.approx(REFERENCE["expected_prediction"], abs=1e-9)
    assert EXAMPLE == REFERENCE["input"]


INVALID_BODIES = {
    "missing field": json.dumps(
        {k: v for k, v in EXAMPLE.items() if k != "rank_operating_roa"}
    ),
    "extra field": raw_body(surprise="1"),
    "list instead of object": json.dumps(list(EXAMPLE.values())),
    "five features in a list": json.dumps([*EXAMPLE.values(), 0.5]),
    "numeric string": raw_body(rank_momentum_252_21='"0.5"'),
    "boolean": raw_body(rank_momentum_252_21="true"),
    "null": raw_body(rank_momentum_252_21="null"),
    "above range": raw_body(rank_momentum_252_21="1.5"),
    "below range": raw_body(rank_volatility_63="-0.1"),
    "NaN": raw_body(rank_drawdown_252="NaN"),
    "Infinity": raw_body(rank_drawdown_252="Infinity"),
    "-Infinity": raw_body(rank_drawdown_252="-Infinity"),
    "overflow 1e309": raw_body(rank_drawdown_252="1e309"),
    "not JSON": "rank_momentum_252_21=0.5",
}


@pytest.mark.parametrize("body", INVALID_BODIES.values(), ids=INVALID_BODIES.keys())
def test_predict_invalid_input(client, body, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid data reached the model")

    monkeypatch.setattr(app.state.bundle["model"], "predict", unexpected_call)
    response = client.post("/predict", content=body, headers=JSON)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail and all({"loc", "msg", "type"} <= set(error) for error in detail)


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.bundle["model"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "simulated" not in response.text
    assert response.headers["X-Request-ID"]


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    timings_ms = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median = statistics.median(timings_ms)
    p95 = float(np.percentile(timings_ms, 95))
    report = "\n".join(
        [
            "# Latency report",
            "",
            f"- Requests: {len(timings_ms)} successful POST /predict after 5 warm-up requests",
            f"- Median: {median:.2f} ms",
            f"- p95: {p95:.2f} ms",
            "- Method: in-process FastAPI TestClient, time.perf_counter around each request, "
            "no network hop",
            f"- Hardware: {platform.machine()} {platform.processor() or 'cpu'}, "
            f"{os.cpu_count()} logical cores, {platform.platform()}",
            f"- Python {platform.python_version()}",
            "",
        ]
    )
    (ROOT / "latency_report.md").write_text(report)
    assert median < LATENCY_BUDGET_MS, report


def test_saved_model_scores_a_row_independently_of_the_batch(client, valid_input_data):
    bundle = app.state.bundle
    neutral = dict.fromkeys(bundle["features"], 0.5)
    alone = pd.DataFrame([valid_input_data])[bundle["features"]]
    batch = pd.DataFrame([valid_input_data, neutral])[bundle["features"]]
    assert bundle["model"].predict(alone)[0] == pytest.approx(bundle["model"].predict(batch)[0])


def test_health_reports_loaded_version(client):
    assert client.get("/health").json() == {"status": "ready", "model_version": MODEL_VERSION}
