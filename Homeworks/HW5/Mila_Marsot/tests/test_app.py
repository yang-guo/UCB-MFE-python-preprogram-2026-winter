"""The 4 groups of tests. They use my real HW4 model."""

import json
import math
import statistics
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import ARTIFACT_DIR, app
from hw4_model import FEATURES

# 2 real HW4 rows, with the prediction of my HW4 model.
# export_hw4_model.py saved them before the API existed.
REFERENCES = json.loads((ARTIFACT_DIR / "reference_predictions.json").read_text())


@pytest.fixture
def client():
    # "with" loads the model, like when the server starts.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    # A server error gives a 500 answer (and does not stop the test).
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    return dict(REFERENCES[0]["features"])  # NVDA


# 1. Good input


@pytest.mark.parametrize("reference", REFERENCES, ids=[r["name"] for r in REFERENCES])
def test_predict_valid_input(client, reference):
    response = client.post("/predict", json=reference["features"])

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == "hw4-ridge-alpha100-trained-before-2016-03-31"

    # Same as a direct call to the model.
    row = pd.DataFrame([reference["features"]])[FEATURES]
    direct = app.state.bundle["pipeline"].predict(row)[0]
    assert body["prediction"] == pytest.approx(direct, rel=0, abs=1e-12)

    # Same as the HW4 prediction. The last digit can change on another computer.
    assert body["prediction"] == pytest.approx(reference["hw4_prediction"], rel=0, abs=1e-10)


# 2. Bad input


def without(data, key):
    return {k: v for k, v in data.items() if k != key}


BAD_INPUTS = {
    "12 features": lambda d: without(d, "momentum_63"),
    "no sector": lambda d: without(d, "sector"),
    "14 features": lambda d: {**d, "dividend_yield": 0.02},
    "number as text": lambda d: {**d, "momentum_21": "0.05"},
    "true instead of a number": lambda d: {**d, "momentum_21": True},
    "null not allowed": lambda d: {**d, "volatility_63": None},
    "NaN": lambda d: {**d, "momentum_21": float("nan")},
    "Infinity": lambda d: {**d, "net_margin": float("inf")},
    "return below -100%": lambda d: {**d, "momentum_63": -1.5},
    "negative volatility": lambda d: {**d, "volatility_63": -0.2},
    "positive drawdown": lambda d: {**d, "drawdown_252": 0.1},
    "unknown sector": lambda d: {**d, "sector": "Crypto"},
    "list instead of names": lambda d: list(d.values()),
    "empty body": lambda d: {},
}


@pytest.mark.parametrize("make_input", BAD_INPUTS.values(), ids=BAD_INPUTS.keys())
def test_predict_invalid_input(client, valid_input_data, make_input, monkeypatch):
    # If the model runs, the test fails.
    def model_must_not_run(*args, **kwargs):
        pytest.fail("Bad input reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", model_must_not_run)

    # json.dumps can write NaN and Infinity, so I can send them.
    body = json.dumps(make_input(valid_input_data))
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# 3. The model fails


def predict_raises(X):
    raise RuntimeError("simulated failure")


def predict_returns_nan(X):
    return np.array([np.nan])


@pytest.mark.parametrize("broken_predict", [predict_raises, predict_returns_nan])
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, broken_predict):
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "simulated failure" not in response.text  # the error stays in the log


# 4. Performance


def test_predict_performance(client, valid_input_data):
    for _ in range(10):  # warm-up
        assert client.post("/predict", json=valid_input_data).status_code == 200

    times_ms = []
    for _ in range(100):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        times_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(times_ms)
    p95 = np.percentile(times_ms, 95)
    print(f"\nTestClient, 100 requests: median {median:.2f} ms, p95 {p95:.2f} ms")
    # I measure about 3 ms. 100 ms is OK for a slower computer.
    assert p95 < 100
