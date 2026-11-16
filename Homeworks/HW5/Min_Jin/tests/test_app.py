"""Tests for the HW5 API, using the real saved HW4 artifact except in the failure test"""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from factor_model import FACTOR_FEATURES, to_frame

HERE = Path(__file__).resolve().parents[1]
# Recorded by export_model.py from HW4's own model.FactorScore
REFERENCE = json.loads((HERE / "reference_example.json").read_text())


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
    # Return a JSON-compatible dict matching your documented input contract.
    return dict(REFERENCE["input"])


@pytest.fixture
def predict_calls(client, monkeypatch):
    """Wrap the loaded model's predict to count number of requests reach inference"""
    model = client.app.state.model
    real_predict = model.predict
    calls = []

    def counting_predict(X):
        calls.append(X)
        return real_predict(X)

    # swap it in for this test only
    monkeypatch.setattr(model, "predict", counting_predict)
    return calls


def test_predict_valid_input(client, valid_input_data):
    # Use the real saved model. Check schema, direct/API agreement, known result.
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert body["model_version"] == "hw4-factor-score-v1"

    # Same answer as calling the loaded model directly with the training columns
    direct = client.app.state.model.predict(to_frame([valid_input_data]))[0]
    assert body["prediction"] == pytest.approx(direct, rel=0, abs=1e-12)

    # Same answer HW4's code gave before the model was wrapped in the API
    assert body["prediction"] == pytest.approx(
        REFERENCE["expected_prediction"], rel=0, abs=1e-12
    )

    # Hand check of the formula: mean of the 4 ranks and 1 - volatility rank
    by_hand = np.mean(
        [valid_input_data[f] for f in FACTOR_FEATURES[:4]]
        + [1 - valid_input_data["rank_volatility_63"]]
    )
    assert body["prediction"] == pytest.approx(by_hand, abs=1e-12)


def test_predict_feature_order_does_not_matter(client, valid_input_data):
    """JSON key order must not change which value goes to which column."""
    reversed_input = dict(reversed(list(valid_input_data.items())))
    response = client.post("/predict", json=reversed_input)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(REFERENCE["expected_prediction"])


def test_predict_accepts_rank_bounds(client, valid_input_data):
    """0 and 1 are valid ranks."""
    for value in (0, 1, 0.0, 1.0):
        response = client.post(
            "/predict", json={**valid_input_data, "rank_volatility_63": value}
        )
        assert response.status_code == 200


# helper for building the bad request body
def _replace(base, **changes):
    """returns a copy of base with some keys set to new values"""
    return {**base, **changes}


def _drop(base, key):
    """return a copy of base without one key"""
    return {k: v for k, v in base.items() if k != key}


# list of bad request bodies
VALID = REFERENCE["input"]
INVALID_CASES = {
    "missing field (only 4 features)": _drop(VALID, "rank_volatility_63"),
    "empty body": {},
    "extra field (6 features)": _replace(VALID, rank_size=0.5),
    "misspelled field": _replace(_drop(VALID, "rank_operating_roa"), rank_operating_ROA=0.5),
    "numeric string": _replace(VALID, rank_operating_roa="0.5"),
    "boolean": _replace(VALID, rank_fcf_to_assets=True),
    "null": _replace(VALID, rank_drawdown_252=None),
    "list instead of number": _replace(VALID, rank_drawdown_252=[0.5]),
    "below 0": _replace(VALID, rank_momentum_252_21=-0.01),
    "above 1 (raw value, not a rank)": _replace(VALID, rank_momentum_252_21=1.5),
    "feature list instead of named fields": list(VALID.values()),
}


@pytest.mark.parametrize("payload", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_predict_invalid_input(client, predict_calls, payload):
    # Parametrize missing/extra fields, types, bounds, and feature-count cases.
    # Confirm rejected requests do not call the model.
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    # Rejected requests never reach the model
    assert predict_calls == []


def test_valid_request_reaches_model_once(client, predict_calls, valid_input_data):
    """checks the counter can see the request"""
    assert client.post("/predict", json=valid_input_data).status_code == 200
    assert len(predict_calls) == 1
    assert list(predict_calls[0].columns) == FACTOR_FEATURES


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    # Patch predict on the loaded model to raise, send VALID input, assert 500.
    # Do not assume a JSON error body unless your application defines one.
    def broken_predict(X):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(failure_client.app.state.model, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    # The exception text stays in the server log
    assert "simulated" not in response.text


def test_predict_non_finite_model_output_is_500(failure_client, valid_input_data, monkeypatch):
    """A NaN score is an inference failure instead of a successful prediction."""
    monkeypatch.setattr(
        failure_client.app.state.model, "predict", lambda X: np.array([float("nan")])
    )
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


# settings for the performance test
P95_BUDGET_MS = 100
N_WARMUP = 5
N_MEASURED = 50


def test_predict_performance(client, valid_input_data, capsys):
    # send a few untimed ones before measuring for warming up
    for _ in range(N_WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    # Time each request on its own with perf_counter and convert from seconds to ms.
    latencies_ms = []
    for _ in range(N_MEASURED):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        # only successful requests count
        assert response.status_code == 200

    # median = a typical request
    # 95% of requests were faster than this
    median = statistics.median(latencies_ms)
    p95 = float(np.percentile(latencies_ms, 95))
    # show report in the terminal
    with capsys.disabled():
        print(
            f"\nLatency (in-process TestClient, {N_MEASURED} requests after {N_WARMUP} warm-up, "
            f"{platform.machine()} {platform.system()}, Python {platform.python_version()}): "
            f"median {median:.2f} ms, p95 {p95:.2f} ms"
        )
    # Fail only on a large slowdown
    assert p95 < P95_BUDGET_MS
