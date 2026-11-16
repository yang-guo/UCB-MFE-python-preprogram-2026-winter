"""Tests for the HW5 service around the saved HW4 model."""

import json
import logging
import math
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from hw4_model import FEATURES, MODEL_VERSION, to_frame

REFERENCE = json.loads((Path(__file__).parent / "hw4_reference.json").read_text())

# HW4 fit Ridge on float32 ranks, so coef_ and predict() work in float32
# An absolute tolerance of 1e-7 allows for float32 rounding but would catch a wrong feature order, scaling, or model.

TOLERANCE = 1e-7


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
    # Return a JSON-compatible dict matching documented input contract.
    # AAPL on 2026-09-25 (same payload as example_request.json).
    return dict(REFERENCE[0]["features"])


@pytest.fixture
def model_must_not_be_called(client, monkeypatch):
    """Make any call to the loaded model fail the test."""

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(client.app.state.bundle["pipeline"], "predict", unexpected_call)
    

@pytest.mark.parametrize("reference", REFERENCE, ids=[r["symbol"] for r in REFERENCE])
def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data["features"])

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == MODEL_VERSION
    assert response.headers["X-Request-ID"]

    pipeline = client.app.state.bundle["pipeline"]
    direct = float(pipeline.predict(to_frame(valid_input_data["features"]))[0])
    assert body["prediction"] == pytest.approx(direct, abs=TOLERANCE)
    assert body["prediction"] == pytest.approx(valid_input_data["hw4_recorded_score"], abs=TOLERANCE)


def test_example_request_file_matches_contract(client):
    payload = json.loads((Path(__file__).parents[1] / "example_request.json").read_text())
    assert list(payload) == FEATURES
    assert client.post("/predict", json=payload).status_code == 200


def test_prediction_does_not_depend_on_other_rows(client, valid_input_data):
    # The saved preprocessing is fixed; nothing is re-learned from a batch.
    pipeline = client.app.state.bundle["pipeline"]
    alone = pipeline.predict(to_frame(valid_input_data))[0]
    rows = [to_frame(valid_input_data)] + [to_frame(r["features"]) for r in REFERENCE[1:]]
    batch = pipeline.predict(pd.concat(rows, ignore_index=True))
    assert batch[0] == pytest.approx(alone, abs=TOLERANCE)


def test_health_reports_loaded_version(client):
    assert client.get("/health").json() == {"status": "ready", "model_version": MODEL_VERSION}


def _without(field):
    return lambda d: {k: v for k, v in d.items() if k != field}


def _with(field, value):
    return lambda d: {**d, field: value}


INVALID_JSON_CASES = {
    "missing_field": _without("r_mom_252_21"),
    "empty_object": lambda d: {},
    "extra_field": _with("r_mom_126", 0.1),  # 8 features: wrong count
    "numeric_string": _with("r_earnings_yield", "0.1"),
    "boolean": _with("r_earnings_yield", True),
    "list_value": _with("r_earnings_yield", [0.1]),
    "above_range": _with("r_fcf_to_assets", 0.51),
    "below_range": _with("r_fcf_to_assets", -0.51),
    "raw_value_not_rank": _with("r_mom_252_21", 3.2),
    "fixed_list_7": lambda d: list(d.values()),
    "fixed_list_6": lambda d: list(d.values())[:6],
    "null_body": lambda d: None,
}


@pytest.mark.parametrize("make_payload", INVALID_JSON_CASES.values(), ids=INVALID_JSON_CASES)
@pytest.mark.usefixtures("model_must_not_be_called")
def test_predict_invalid_input(client, valid_input_data, make_payload):
    response = client.post("/predict", json=make_payload(valid_input_data))
    assert response.status_code == 422

NON_FINITE = ["NaN", "Infinity", "-Infinity", "1e999"]


@pytest.mark.parametrize("literal", NON_FINITE)
@pytest.mark.usefixtures("model_must_not_be_called")
def test_predict_rejects_non_finite(client, valid_input_data, literal):
    body = json.dumps(valid_input_data).replace(
        str(valid_input_data["r_operating_roa"]), literal
    )
    assert literal in body
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "r_operating_roa"]


@pytest.mark.usefixtures("model_must_not_be_called")
def test_predict_rejects_malformed_json(client):
    response = client.post(
        "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_null_feature_is_imputed_as_median(client, valid_input_data):
    # null is allowed (feature unavailable) and must equal sending the median rank 0.
    with_null = client.post("/predict", json={**valid_input_data, "r_sales_growth_yoy": None})
    with_zero = client.post("/predict", json={**valid_input_data, "r_sales_growth_yoy": 0.0})
    assert with_null.status_code == with_zero.status_code == 200
    assert with_null.json()["prediction"] == pytest.approx(with_zero.json()["prediction"])



def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated inference failure: secret internal detail")

    monkeypatch.setattr(failure_client.app.state.bundle["pipeline"], "predict", broken_predict)
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "prediction" not in response.text
    assert "secret internal detail" not in response.text
    assert response.headers["X-Request-ID"]
    assert "inference_failed" in caplog.text and "secret internal detail" in caplog.text


def test_non_finite_model_output_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(
        failure_client.app.state.bundle["pipeline"], "predict", lambda X: [float("nan")]
    )
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


WARMUP, N_REQUESTS = 5, 200
P95_BUDGET_MS = 50.0


def test_predict_performance(client, valid_input_data):
    for _ in range(WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(N_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=100, method="inclusive")[94]
    print(
        f"\n[latency] in-process TestClient, {N_REQUESTS} sequential requests after "
        f"{WARMUP} warm-ups, {platform.machine()} {platform.system()} Python "
        f"{platform.python_version()}: median {median:.2f} ms, p95 {p95:.2f} ms"
    )
    assert p95 < P95_BUDGET_MS
