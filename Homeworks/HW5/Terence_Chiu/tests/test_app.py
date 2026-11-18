"""Tests for the HW5 prediction service, run in-process with FastAPI's TestClient."""

import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURES, MODEL_VERSION, app

# BOOT on 2016-03-31 (the first HW4 signal date) and the saved HW4 pipeline's score
# for it, recorded from the HW4 code before the model was wrapped in this API.
REFERENCE = json.loads((Path(__file__).parent / "reference_example.json").read_text())

DROP = object()  # marker: remove this field from the payload


@pytest.fixture
def client():
    # Entering the context runs the app's lifespan, so the real model is loaded.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_input():
    # A fresh copy for every test, so one test's edits cannot leak into another.
    return dict(REFERENCE["input"])


@pytest.fixture
def predict_calls(client, monkeypatch):
    """Replace the model's predict with a recorder; tests assert it was never called."""
    calls = []
    monkeypatch.setattr(client.app.state.pipeline, "predict", lambda X: calls.append(X))
    return calls


def test_predict_valid_input(client, valid_input):
    response = client.post("/predict", json=valid_input)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == MODEL_VERSION
    assert math.isfinite(body["prediction"])
    assert response.headers["X-Request-ID"]

    # The API must agree with calling the saved pipeline directly...
    row = pd.DataFrame([valid_input], columns=FEATURES)
    direct = client.app.state.pipeline.predict(row)[0]
    assert body["prediction"] == pytest.approx(direct, rel=1e-12)
    # ...and with the prediction recorded from the HW4 code.
    assert body["prediction"] == pytest.approx(REFERENCE["expected_prediction"], rel=1e-9)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sector", DROP),  # missing required field
        ("symbol", "BOOT"),  # extra field
        ("momentum_21", "0.02"),  # numeric string must not become a number
        ("momentum_21", True),  # boolean must not become a number
        ("net_margin", None),  # null is not a finite number
        ("sector", "Crypto"),  # sector not seen in training
        ("momentum_63", -1.0),  # a return cannot reach -100%
        ("volatility_63", -0.1),  # volatility cannot be negative
        ("drawdown_252", 0.2),  # price cannot be above its own high
        ("filing_age_days", -1.0),  # filing cannot be in the future
    ],
    ids=[
        "missing-field",
        "extra-field",
        "numeric-string",
        "boolean",
        "null",
        "unknown-sector",
        "momentum-at-minus-100pct",
        "negative-volatility",
        "positive-drawdown",
        "negative-filing-age",
    ],
)
def test_predict_invalid_input(client, valid_input, predict_calls, field, value):
    payload = dict(valid_input)
    if value is DROP:
        del payload[field]
    else:
        payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert predict_calls == []  # validation stopped the request before inference


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_predict_rejects_non_finite(client, valid_input, predict_calls, value):
    # Standard JSON has no NaN/Infinity, so send Python's extended JSON text directly.
    payload = json.dumps({**valid_input, "net_margin": value})

    response = client.post(
        "/predict", content=payload, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert predict_calls == []


def test_predict_rejects_wrong_feature_count(client, valid_input, predict_calls):
    # A bare list of values (any length) is not the named-field contract.
    response = client.post("/predict", json=list(valid_input.values())[:-1])

    assert response.status_code == 422
    assert predict_calls == []


def test_predict_model_failure(client, valid_input, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(client.app.state.pipeline, "predict", broken_predict)

    response = client.post("/predict", json=valid_input)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "simulated" not in response.text  # internal details stay in the server log
    assert response.headers["X-Request-ID"]


def test_predict_performance(client, valid_input):
    for _ in range(5):  # warm-up: the first calls pay one-time costs
        assert client.post("/predict", json=valid_input).status_code == 200

    durations = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input)
        durations.append(time.perf_counter() - start)
        assert response.status_code == 200

    median_ms = statistics.median(durations) * 1000
    p95_ms = float(np.percentile(durations, 95)) * 1000
    print(
        f"\nIn-process TestClient latency over {len(durations)} requests: "
        f"median {median_ms:.2f} ms, p95 {p95_ms:.2f} ms"
    )
    # Generous local budget: catches regressions such as reloading the model per request.
    assert p95_ms < 500
