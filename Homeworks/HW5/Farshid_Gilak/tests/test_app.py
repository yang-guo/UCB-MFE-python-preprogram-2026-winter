"""API tests for the served HW4 model (patterns adapted from Lectures/Lecture 9/demo/tests)."""

import json
import math
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
NULLABLE = json.loads((ROOT / "artifacts/model_metadata.json").read_text())["nullable_features"]
P95_BUDGET_SECONDS = 0.050


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
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def hw4_reference():
    return json.loads((ROOT / "tests/hw4_reference.json").read_text())


def post_json(client, payload):
    """Send raw JSON so NaN and Infinity tokens reach the server unchanged."""
    return client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )


def direct_prediction(payload):
    pipeline = app.state.pipeline
    frame = pd.DataFrame([payload], columns=list(pipeline.feature_names_in_))
    numeric = {name: "float64" for name in frame.columns if name != "sector"}
    return float(pipeline.predict(frame.astype(numeric))[0])


def test_predict_valid_input(client, valid_input_data, hw4_reference):
    response = post_json(client, valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == hw4_reference["model_version"]
    assert body["prediction"] == pytest.approx(direct_prediction(valid_input_data), abs=1e-12)
    assert body["prediction"] == pytest.approx(hw4_reference["prediction"], abs=1e-9)


def test_predict_missing_features_are_imputed(client, valid_input_data):
    payload = {**valid_input_data, **{name: None for name in NULLABLE}}
    response = post_json(client, payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(direct_prediction(payload), abs=1e-12)


INVALID_PAYLOADS = {
    "empty body": lambda p: {},
    "body is a list": lambda p: [p],
    "one feature too few": lambda p: {k: v for k, v in p.items() if k != "vol_63"},
    "one feature too many": lambda p: {**p, "dividend_yield": 0.5},
    "fixed-length list instead of named features": lambda p: {
        "features": [v for k, v in p.items() if k != "sector"]
    },
    "numeric string": lambda p: {**p, "mom_21": "0.5"},
    "boolean": lambda p: {**p, "mom_21": True},
    "null for a required feature": lambda p: {**p, "mom_21": None},
    "NaN": lambda p: {**p, "mom_21": float("nan")},
    "Infinity": lambda p: {**p, "roa": float("inf")},
    "-Infinity": lambda p: {**p, "roa": float("-inf")},
    "rank of zero": lambda p: {**p, "vol_63": 0.0},
    "rank above one": lambda p: {**p, "vol_63": 1.5},
    "negative rank": lambda p: {**p, "beta_252": -0.2},
    "raw log market cap instead of rank": lambda p: {**p, "log_mcap": 23.5},
    "sector not seen in training": lambda p: {**p, "sector": "Unknown"},
    "sector with different case": lambda p: {**p, "sector": "financial services"},
    "sector as a number": lambda p: {**p, "sector": 3},
    "missing sector": lambda p: {k: v for k, v in p.items() if k != "sector"},
}


@pytest.mark.parametrize("make_payload", INVALID_PAYLOADS.values(), ids=INVALID_PAYLOADS.keys())
def test_predict_invalid_input(client, valid_input_data, make_payload, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.pipeline, "predict", unexpected_call)
    response = post_json(client, make_payload(valid_input_data))
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "broken_predict",
    [
        pytest.param(lambda frame: 1 / 0, id="predict raises"),
        pytest.param(lambda frame: np.array([np.nan]), id="non-finite prediction"),
    ],
)
def test_predict_model_failure(failure_client, valid_input_data, broken_predict, monkeypatch):
    monkeypatch.setattr(app.state.pipeline, "predict", broken_predict)
    response = post_json(failure_client, valid_input_data)
    assert response.status_code == 500
    assert "model_version" not in response.text


def test_predict_performance(client, valid_input_data, capsys):
    for _ in range(5):
        assert post_json(client, valid_input_data).status_code == 200
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        response = post_json(client, valid_input_data)
        latencies.append(time.perf_counter() - start)
        assert response.status_code == 200
    median = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    with capsys.disabled():
        print(
            f"\nIn-process TestClient latency over {len(latencies)} requests on "
            f"{platform.platform()} ({platform.machine()}), Python {platform.python_version()}: "
            f"median {median * 1e3:.2f} ms, p95 {p95 * 1e3:.2f} ms"
        )
    assert p95 < P95_BUDGET_SECONDS
