"""Tests for the HW5 service, run against the real exported HW4 model."""

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import InputData, app
from model_service import FEATURE_ORDER, NULLABLE_FEATURES, to_matrix

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "reference_predictions.json").read_text())
EXAMPLE = json.loads((HERE.parent / "example_request.json").read_text())

# The API and the HW4 code feed the same float32 row to the same pipeline, so
# they agree to rounding: 1e-9 on a prediction of order 0.5.
TOLERANCE = 1e-9


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
    # example_request.json: AAPL on 2026-09-29, from the HW4 scoring cross-section.
    return dict(EXAMPLE)


@pytest.fixture
def forbid_inference(monkeypatch):
    """Make any call to the loaded model fail the test."""

    def unexpected(*args, **kwargs):
        pytest.fail("invalid input reached the model")

    monkeypatch.setattr(app.state.model.pipeline, "predict", unexpected)


# 1. Valid prediction


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == REFERENCE["model_version"]

    direct = app.state.model.pipeline.predict(to_matrix(valid_input_data))[0]
    assert body["prediction"] == pytest.approx(float(direct), abs=TOLERANCE)
    assert body["prediction"] == pytest.approx(
        REFERENCE["cases"][0]["hw4_prediction"], abs=TOLERANCE
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["symbol"])
def test_predict_matches_hw4_reference(client, case):
    # The second case has missing fundamentals (null fields).
    response = client.post("/predict", json=case["features"])
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(case["hw4_prediction"], abs=TOLERANCE)


def test_contract_matches_exported_model(client):
    metadata = app.state.model.metadata
    assert tuple(InputData.model_fields) == FEATURE_ORDER == tuple(metadata["features"])
    assert all(field.is_required() for field in InputData.model_fields.values())
    nullable = {
        name
        for name, field in InputData.model_fields.items()
        if type(None) in getattr(field.annotation, "__args__", ())
    }
    assert nullable == NULLABLE_FEATURES == set(metadata["nullable_features"])


# 2. Input validation


def with_change(**changes):
    payload = dict(EXAMPLE)
    for name, value in changes.items():
        if value is ...:
            del payload[name]
        else:
            payload[name] = value
    return payload


def first_n(n):
    return {name: 0.5 for name in FEATURE_ORDER[:n]}


INVALID = {
    "missing field": with_change(r_roa=...),
    "27 features": first_n(27),
    "29 features": {**first_n(28), "r_extra": 0.5},
    "extra field": with_change(symbol="AAPL"),
    "numeric string": with_change(r_mom_21="0.5"),
    "boolean": with_change(r_mom_21=True),
    "null in a never-missing feature": with_change(r_universe_rank_pct=None),
    "rank of 0": with_change(r_vol_63=0.0),
    "rank above 1": with_change(r_vol_63=1.5),
    "negative rank": with_change(r_roa=-0.2),
    "list instead of object": [0.5] * 28,
    "empty body": {},
}


@pytest.mark.parametrize("payload", INVALID.values(), ids=INVALID.keys())
def test_predict_invalid_input(client, forbid_inference, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_predict_non_finite_input(client, forbid_inference, token):
    # Standard JSON has no NaN, but Python's parser accepts these tokens.
    body = json.dumps(with_change(r_mom_63=float(token.replace("Infinity", "inf"))))
    assert token in body
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# 3. Inference failure


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.model.pipeline, "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "simulated" not in response.text  # details stay in the server log


def test_predict_non_finite_output(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.model.pipeline, "predict", lambda X: [float("nan")])
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


# 4. Performance


def test_predict_performance(client, valid_input_data):
    """In-process TestClient latency (no network, no server), reported with `pytest -s`.

    One request is a one-row Ridge prediction plus FastAPI validation, a few
    milliseconds on a laptop. The default budget (p95 < 100 ms) only catches gross
    regressions, such as fitting or reloading the model per request.
    Override it with LATENCY_BUDGET_MS on a slower machine.
    """
    for _ in range(5):
        client.post("/predict", json=valid_input_data)
    latencies_ms, predictions = [], set()
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
        predictions.add(response.json()["prediction"])
    median = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[-1]
    print(
        f"\nIn-process TestClient, POST /predict, 5 warm-up + 50 timed requests "
        f"(time.perf_counter): median {median:.2f} ms, p95 {p95:.2f} ms "
        f"on {platform.platform()}, {platform.machine()}, {os.cpu_count()} CPUs"
    )
    assert len(predictions) == 1
    assert p95 < float(os.environ.get("LATENCY_BUDGET_MS", 100))
