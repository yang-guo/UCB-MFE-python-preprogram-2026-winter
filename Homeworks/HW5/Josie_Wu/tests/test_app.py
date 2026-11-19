import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURES, app

ROOT = Path(__file__).resolve().parents[1]


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
    return json.loads((ROOT / "artifacts/example_request.json").read_text(encoding="utf-8"))


@pytest.fixture
def reference_data():
    return json.loads((ROOT / "artifacts/model_reference.json").read_text(encoding="utf-8"))


def test_predict_valid_input_matches_hw4_reference(client, valid_input_data, reference_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == reference_data["model_version"]

    pipeline = app.state.bundle["pipeline"]
    row = pd.DataFrame([[valid_input_data[name] for name in FEATURES]], columns=FEATURES)
    direct_prediction = float(pipeline.predict(row)[0])
    assert body["prediction"] == pytest.approx(direct_prediction, rel=1e-12, abs=1e-12)
    assert body["prediction"] == pytest.approx(
        reference_data["reference_prediction"], rel=1e-12, abs=1e-12
    )
    assert response.headers.get("X-Request-ID")


def test_invalid_input_never_reaches_model(client, valid_input_data, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)
    missing = valid_input_data.copy()
    missing.pop(FEATURES[-1])
    extra = {**valid_input_data, "unexpected": 1.0}
    numeric_string = {**valid_input_data, FEATURES[0]: str(valid_input_data[FEATURES[0]])}
    boolean = {**valid_input_data, FEATURES[0]: True}
    below_return_bound = {**valid_input_data, FEATURES[0]: -1.0}
    negative_volatility = {**valid_input_data, "volatility_63": -0.01}
    invalid_drawdown = {**valid_input_data, "drawdown_252": 0.01}
    nonfinite = {**valid_input_data, FEATURES[0]: float("nan")}
    payloads = [
        missing,
        extra,
        numeric_string,
        boolean,
        below_return_bound,
        negative_volatility,
        invalid_drawdown,
        nonfinite,
    ]

    for payload in payloads:
        response = client.post(
            "/predict",
            content=json.dumps(payload, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


def test_predict_inference_failure_returns_500(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert response.headers.get("X-Request-ID")


def test_predict_performance(client, valid_input_data):
    warmup = client.post("/predict", json=valid_input_data)
    assert warmup.status_code == 200

    elapsed_ms = []
    for _ in range(20):
        started = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed_ms.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200

    median_ms = statistics.median(elapsed_ms)
    p95_ms = float(np.percentile(elapsed_ms, 95))
    print(
        f"TestClient latency over 20 requests: median={median_ms:.3f} ms, p95={p95_ms:.3f} ms"
    )


def test_health_reports_loaded_model(client, reference_data):
    assert client.get("/health").json() == {
        "status": "ready",
        "model_version": reference_data["model_version"],
    }
