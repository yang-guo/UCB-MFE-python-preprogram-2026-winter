"""Tests of the HW5 API: valid prediction, input validation, inference failure, latency.

Fixtures and the failure pattern are adapted from the HW5 template and the Lesson 9 demo tests.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import predict_one

ROOT = Path(__file__).resolve().parents[1]
JSON_HEADERS = {"Content-Type": "application/json"}

# A real HW4 row: AAPL at the 2026-08-31 signal, with the score HW4 gave it.
# export_model.py wrote it from the output of hw4_model.walk_forward, before any API existed.
REFERENCE = json.loads((ROOT / "artifacts" / "reference_prediction.json").read_text())
METADATA = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
VALID = json.loads((ROOT / "example_request.json").read_text())


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
    return dict(VALID)


def loaded_model():
    """The model that the running app loaded at startup."""
    return app.state.bundle["model"]


# 1. valid prediction, with the real saved model


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == "hw4-elastic-net-fit-2026-06-30"

    # same number as calling the saved model directly, without the API
    features = app.state.bundle["features"]
    direct = loaded_model().predict(pd.DataFrame([valid_input_data])[features])[0]
    assert body["prediction"] == pytest.approx(direct, rel=1e-12)

    # same number as the score HW4 gave to this row.
    # Tolerance 1e-9: same model and same arithmetic, so only float rounding can differ,
    # and 1e-9 on a return is far below anything that matters (the scores are around 1e-2).
    assert valid_input_data == REFERENCE["input"]
    assert body["prediction"] == pytest.approx(REFERENCE["expected_prediction"], abs=1e-9)

    # same number by hand: intercept + sum of coefficient x rank
    by_hand = METADATA["intercept"] + sum(
        METADATA["coefficients"][name] * rank for name, rank in valid_input_data.items()
    )
    assert body["prediction"] == pytest.approx(by_hand, abs=1e-9)


def test_middle_stock_gets_the_intercept(client):
    # all ranks at 0 = the average stock (or all missing): the prediction is the intercept.
    # 0 is sent as an integer here, a JSON integer is a valid number.
    response = client.post("/predict", json={name: 0 for name in VALID})
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(METADATA["intercept"], abs=1e-12)


def test_key_order_in_the_request_does_not_matter(client, valid_input_data):
    # the features are put back in the training order before predicting
    reversed_input = dict(reversed(list(valid_input_data.items())))
    first = client.post("/predict", json=valid_input_data).json()["prediction"]
    second = client.post("/predict", json=reversed_input).json()["prediction"]
    assert first == second
    # same check on the model function alone, without the API
    assert predict_one(app.state.bundle, reversed_input) == first


# 2. input validation: 422, and the model is never called


def without(name):
    return {k: v for k, v in VALID.items() if k != name}


def replaced(name, value):
    return {**VALID, name: value}


INVALID_REQUESTS = {
    "missing one feature (17 instead of 18)": without("z_beta_252"),
    "extra field (19 instead of 18)": {**VALID, "z_surprise": 0.1},
    "empty object": {},
    "list of 18 numbers instead of named fields": list(VALID.values()),
    "number as a string": replaced("z_beta_252", "0.1"),
    "boolean": replaced("z_beta_252", True),
    "null": replaced("z_beta_252", None),
    "list as a value": replaced("z_beta_252", [0.1]),
    "above the range": replaced("z_beta_252", 0.51),
    "below the range": replaced("z_beta_252", -0.51),
    "NaN": replaced("z_beta_252", float("nan")),
    "infinity": replaced("z_beta_252", float("inf")),
    "minus infinity": replaced("z_beta_252", float("-inf")),
    "number too big for a float": json.dumps(VALID).replace(str(VALID["z_beta_252"]), "1e999"),
    "not JSON": "hello",
}


@pytest.mark.parametrize("payload", INVALID_REQUESTS.values(), ids=INVALID_REQUESTS.keys())
def test_predict_invalid_input(client, payload, monkeypatch):
    calls = []
    monkeypatch.setattr(loaded_model(), "predict", lambda *args, **kwargs: calls.append(args))

    # we send the raw text ourselves, because NaN and Infinity are not standard JSON
    body = payload if isinstance(payload, str) else json.dumps(payload)
    response = client.post("/predict", content=body, headers=JSON_HEADERS)

    assert response.status_code == 422
    assert calls == []  # the invalid request never reached the model


# 3. inference failure: 500, never a fake prediction


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(loaded_model(), "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}  # the contract of my app
    assert "simulated inference failure" not in response.text  # no details for the client
    assert "simulated inference failure" in caplog.text  # the details are in the server logs


def test_non_finite_prediction_is_a_failure(failure_client, valid_input_data, monkeypatch):
    # if the model ever returns NaN, the API must say 500, not 200 with a NaN
    monkeypatch.setattr(loaded_model(), "predict", lambda *args, **kwargs: np.array([np.nan]))
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


# 4. performance


def test_predict_performance(client, valid_input_data):
    # In-process measurement with TestClient: no network, no real HTTP server.
    for _ in range(10):  # warm-up
        client.post("/predict", json=valid_input_data)

    latencies_ms = []
    for _ in range(200):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median, p95 = np.percentile(latencies_ms, [50, 95])
    print(f"\nlatency over 200 in-process requests: median {median:.2f} ms, p95 {p95:.2f} ms")

    # Budget: the model is a dot product of 18 numbers, and I measured a p95 of a few ms.
    # 100 ms leaves a big margin for a slow machine, but it still catches a real problem,
    # for example reloading the model at each request.
    assert p95 < 100
