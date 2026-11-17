"""Tests for the HW5 service: valid predictions, validation, failures, and latency.

The positive tests use the real saved HW4 model.  Only the failure tests replace the
loaded estimator's ``predict``, as the assignment asks.
"""

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import EXAMPLE, InputData, OutputData, app
from model_service import FEATURES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = json.loads((HERE / "data" / "hw4_reference.json").read_text())
MODEL_VERSION = "hw4-P1_top50__R1-f06421a6"

# HW4 scored a whole date at once (an (n, 7) @ (7,) product); the API scores one row.
# BLAS may sum the seven terms in a different order, which moves the last bit (the
# AAPL case differs by one ulp, 1.4e-17).  1e-12 allows that and is still ~1e8 times smaller
# than the score gap between neighbouring stocks (~1e-4), so a real change fails.
ABS_TOL = 1e-12


# ----------------------------------------------------------------------------- fixtures
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
    """The AAPL row HW4 scored on 2015-12-31 (same values as example_request.json)."""
    return dict(REFERENCE["cases"][0]["features"])


@pytest.fixture
def model_must_not_run(client, monkeypatch):
    """Replace the loaded model's predict with a spy that records any call."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Invalid input reached the model")

    monkeypatch.setattr(client.app.state.model.estimator, "predict", spy)
    return calls


def post_raw(client, body: str):
    """Send a body exactly as written (httpx's json= refuses NaN, so build it here)."""
    return client.post("/predict", content=body, headers={"Content-Type": "application/json"})


# ------------------------------------------------------------------- 1. valid prediction
def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    OutputData.model_validate(body)  # schema: finite float + string
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == MODEL_VERSION
    assert response.headers["X-Request-ID"]

    # Direct call on the loaded estimator, with HW4's representation: float64 array,
    # columns in training order.
    estimator = client.app.state.model.estimator
    row = np.array([[valid_input_data[name] for name in FEATURES]], dtype=float)
    assert body["prediction"] == pytest.approx(estimator.predict(row)[0], rel=0, abs=ABS_TOL)

    # The score HW4 itself recorded for this row before any API existed.
    assert body["prediction"] == pytest.approx(
        REFERENCE["cases"][0]["hw4_score"], rel=0, abs=ABS_TOL
    )


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["symbol"])
def test_api_reproduces_hw4_reference_scores(client, case):
    """Each case: inputs and score recorded by HW4's walk_forward_scores (see README)."""
    response = client.post("/predict", json=case["features"])
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(case["hw4_score"], rel=0, abs=ABS_TOL)


def test_example_request_file_matches_reference(valid_input_data):
    assert json.loads((ROOT / "example_request.json").read_text()) == valid_input_data
    assert EXAMPLE == valid_input_data  # the example shown in /docs


def test_field_order_in_json_does_not_change_the_prediction(client, valid_input_data):
    reversed_body = dict(reversed(list(valid_input_data.items())))
    first = client.post("/predict", json=valid_input_data).json()["prediction"]
    assert client.post("/predict", json=reversed_body).json()["prediction"] == first


def test_null_rank_counts_as_median_like_hw4(client, valid_input_data):
    """HW4 passed a missing rank as NaN, which the composite treats as 0 (the median)."""
    with_null = {**valid_input_data, "mom_252_21_csr": None}
    with_median = {**valid_input_data, "mom_252_21_csr": 0.0}
    a = client.post("/predict", json=with_null)
    b = client.post("/predict", json=with_median)
    assert a.status_code == b.status_code == 200
    assert a.json()["prediction"] == pytest.approx(b.json()["prediction"], rel=0, abs=ABS_TOL)


def test_range_bounds_are_accepted(client, valid_input_data):
    for value in (-0.5, 0.5, 0):  # 0 is a JSON integer; ints are valid numbers
        response = client.post("/predict", json={**valid_input_data, "vol_63_csr": value})
        assert response.status_code == 200


def test_model_is_loaded_once_per_process(monkeypatch, valid_input_data):
    loads = []
    real_load = app_module.load_model

    def counting_load(*args, **kwargs):
        loads.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(app_module, "load_model", counting_load)
    with TestClient(app) as test_client:
        for _ in range(5):
            assert test_client.post("/predict", json=valid_input_data).status_code == 200
    assert loads == [1]


def test_health_reports_loaded_version(client):
    assert client.get("/health").json() == {"status": "ready", "model_version": MODEL_VERSION}


# ------------------------------------------------------------------- 2. input validation
def _without(payload, name):
    return {k: v for k, v in payload.items() if k != name}


INVALID_JSON_CASES = {
    # feature count / field names
    "missing field (6 features)": lambda p: _without(p, "ret_21_csr"),
    "extra field (8 features)": lambda p: {**p, "ret_63_csr": 0.1},
    "misspelled field": lambda p: {**_without(p, "vol_63_csr"), "vol63_csr": 0.1},
    "empty object": lambda p: {},
    "fixed-length list instead of named fields": lambda p: list(p.values()),
    "list with 6 values": lambda p: list(p.values())[:6],
    # types
    "numeric string": lambda p: {**p, "vol_63_csr": "0.1"},
    "boolean": lambda p: {**p, "vol_63_csr": True},
    "nested object": lambda p: {**p, "vol_63_csr": {"value": 0.1}},
    "list as a value": lambda p: {**p, "vol_63_csr": [0.1]},
    "null body": lambda p: None,
    # ranges
    "above 0.5": lambda p: {**p, "ret_21_csr": 0.5000001},
    "below -0.5": lambda p: {**p, "ret_21_csr": -0.51},
    "raw ratio instead of a rank": lambda p: {**p, "book_to_market_csr": 1.7},
}


@pytest.mark.parametrize("make_payload", INVALID_JSON_CASES.values(), ids=INVALID_JSON_CASES)
def test_predict_invalid_input(client, model_must_not_run, valid_input_data, make_payload):
    response = client.post("/predict", json=make_payload(valid_input_data))
    assert response.status_code == 422
    assert model_must_not_run == []  # rejected before inference


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_are_rejected(client, model_must_not_run, valid_input_data, token):
    # Python's json.dumps writes float("nan") as the bare token NaN, which Starlette parses.
    body = json.dumps({**valid_input_data, "ret_21_csr": float(token)})
    assert f'"ret_21_csr": {token}' in body
    response = post_raw(client, body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "ret_21_csr"]
    assert model_must_not_run == []


@pytest.mark.parametrize("body", ["", "not json", '{"gross_profitability_csr": 0.1,'])
def test_malformed_json_is_rejected(client, model_must_not_run, body):
    assert post_raw(client, body).status_code == 422
    assert model_must_not_run == []


def test_validation_error_names_the_offending_field(client, valid_input_data):
    response = client.post("/predict", json={**valid_input_data, "operating_roa_csr": "high"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "operating_roa_csr"]


def test_contract_fields_match_training_order():
    assert tuple(InputData.model_fields) == FEATURES
    schema = InputData.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(FEATURES)


# ------------------------------------------------------------------ 3. inference failure
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated inference failure: secret internal detail")

    monkeypatch.setattr(failure_client.app.state.model.estimator, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}  # our documented body
    assert "secret internal detail" not in response.text
    request_id = response.headers["X-Request-ID"]
    # The detail stays in the server log, tagged with the same request ID.
    assert any(
        request_id in r.getMessage()
        and r.exc_info
        and "secret internal detail" in str(r.exc_info[1])
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "bad_output",
    [np.array([np.nan]), np.array([np.inf]), np.array([0.1, 0.2]), np.array([])],
    ids=["nan", "inf", "two-predictions", "no-prediction"],
)
def test_bad_model_output_is_an_error_not_a_prediction(
    failure_client, valid_input_data, monkeypatch, bad_output
):
    monkeypatch.setattr(
        failure_client.app.state.model.estimator, "predict", lambda X: bad_output
    )
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "prediction" not in response.json()


# ------------------------------------------------------------------------ 4. performance
# Budget: the HW4 strategy scores at most 2,000 stocks (its top-2,000 universe) once a
# week after the close and trades at the next session's close.  Even one request per
# stock at 50 ms is 100 s, far inside that window, so a 50 ms p95 is a generous but
# meaningful ceiling for this in-process measurement (typical here: ~3 ms).
P95_BUDGET_MS = 50.0
WARMUP_REQUESTS = 10
MEASURED_REQUESTS = 200


def test_predict_performance(client, valid_input_data, capsys):
    for _ in range(WARMUP_REQUESTS):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(MEASURED_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = float(np.percentile(latencies_ms, 95))
    with capsys.disabled():  # print the report even under pytest -q
        print(
            f"\n[latency] in-process FastAPI TestClient (no network), {MEASURED_REQUESTS} "
            f"sequential POST /predict after {WARMUP_REQUESTS} warm-up requests, "
            f"time.perf_counter per request: median {median:.2f} ms, p95 {p95:.2f} ms, "
            f"max {max(latencies_ms):.2f} ms | {platform.platform()}, "
            f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
            f"Python {platform.python_version()}"
        )
    assert p95 < P95_BUDGET_MS
