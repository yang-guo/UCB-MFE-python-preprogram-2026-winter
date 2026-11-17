"""Tests for the HW4 model service. Run ``python -m pytest -q`` from the submission folder.

The four groups of the assignment are marked by section comments: valid prediction, input
validation, inference failure and performance. A fifth section covers model loading.
"""

import dataclasses
import json
import logging
import math
import platform
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import inference
from app import InputData, OutputData, app

ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
# Real rows of my HW4 panel with the prediction HW4 made for them. export_hw4_model.py wrote
# them straight from the output of HW4's walk_forward_predict, before any API existed.
HW4_CASES = json.loads((ROOT / "tests" / "data" / "hw4_reference.json").read_text())["cases"]

MODEL_VERSION = "hw4-ridge_rank-2026-01-05"
JSON_BODY = {"Content-Type": "application/json"}

# HW4 scored all 79,925 rows of this fit as one float32 batch; the service scores one float32
# row at a time. The two differ by at most 1.5e-8 over those rows (recorded in
# model_metadata.json). 1e-6 leaves room for another BLAS in the Linux container and is far
# below any real mistake: on the first case the previous year's fit moves the prediction by
# 1.3e-3 and swapping two features moves it by 4e-4.
HW4_TOLERANCE = 1e-6

# Not a service-level target: HW4 decides after the close and trades at the next close, so
# nobody waits on one request. This is a tripwire, roughly 20 to 50 times the 1 to 3 ms p95
# I measure in-process: loose enough for a slow CI machine, and it fails if a request starts
# taking tens of milliseconds. It cannot see cheap extra work; loading or fitting during a
# request is tested directly in the model-loading section.
P95_BUDGET_MS = 50
WARMUP_REQUESTS, MEASURED_REQUESTS = 20, 200


@pytest.fixture
def client():
    # The context manager runs the lifespan, so the model is loaded as in a running server.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    # Returns an HTTP 500 to the test instead of re-raising the server's exception.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    """NVDA on 2026-09-28, the last HW4 signal date: the body saved in example_request.json."""
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Count the calls that reach the loaded model, without changing what it returns."""
    pipeline = app.state.model.pipeline
    spy = Mock(wraps=pipeline.predict)
    monkeypatch.setattr(pipeline, "predict", spy)
    return spy


# ----------------------------------------------------------------------------- valid prediction
def test_predict_valid_input(client, valid_input_data, predict_spy):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == MODEL_VERSION
    assert response.headers["X-Request-ID"]
    assert predict_spy.call_count == 1  # the spy does see inference (used by the 422 tests)

    # Direct call to the saved pipeline, with the row written out here independently of the
    # service: training order, float32. The API adds no arithmetic, so the match is exact.
    in_training_order = [valid_input_data[name] for name in METADATA["features"]]
    row = np.array([in_training_order], dtype=np.float32)
    assert body["prediction"] == float(app.state.model.pipeline.predict(row)[0])

    # The number HW4 itself produced for this stock and date.
    nvda = HW4_CASES[0]
    assert valid_input_data == nvda["features"]
    assert body["prediction"] == pytest.approx(nvda["hw4_prediction"], abs=HW4_TOLERANCE)


@pytest.mark.parametrize("case", HW4_CASES, ids=lambda case: case["symbol"])
def test_predict_reproduces_hw4(client, case):
    """Three HW4 rows; two have missing fundamentals, sent as null."""
    response = client.post("/predict", json=case["features"])
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction == pytest.approx(case["hw4_prediction"], abs=HW4_TOLERANCE)


def test_null_is_scored_as_the_middle_rank(client, valid_input_data):
    """HW4 set a missing rank to 0.0 before the Ridge; null must give exactly that."""
    original = client.post("/predict", json=valid_input_data).json()["prediction"]
    with_null = client.post("/predict", json=valid_input_data | {"revenue_growth": None})
    with_zero = client.post("/predict", json=valid_input_data | {"revenue_growth": 0.0})
    assert with_null.status_code == with_zero.status_code == 200
    assert with_null.json()["prediction"] == with_zero.json()["prediction"] != original


@pytest.mark.parametrize("name", METADATA["features"])
def test_null_is_accepted_only_where_hw4_had_missing_values(client, valid_input_data, name):
    response = client.post("/predict", json=valid_input_data | {name: None})
    never_missing_in_hw4 = name in METADATA["never_missing_features"]
    assert response.status_code == (422 if never_missing_in_hw4 else 200)


def test_prediction_does_not_depend_on_json_key_order(client, valid_input_data):
    """The contract says key order is free: fields are matched by name."""
    reordered = dict(reversed(valid_input_data.items()))
    expected = client.post("/predict", json=valid_input_data).json()["prediction"]
    assert client.post("/predict", json=reordered).json()["prediction"] == expected


@pytest.mark.parametrize("rank", [-0.5, 0, 0.5])
def test_rank_bounds_are_inclusive(client, valid_input_data, rank):
    """0.5 is the top-ranked stock, so it must pass; 0 is a JSON integer and still a number."""
    assert client.post("/predict", json=valid_input_data | {"mom_21": rank}).status_code == 200


def test_health_reports_the_loaded_version(client):
    assert client.get("/health").json() == {"status": "ready", "model_version": MODEL_VERSION}


# ----------------------------------------------------------------------------- input validation
INVALID_BODIES = {
    "23 features: a required field is missing": lambda body: dict(list(body.items())[:-1]),
    "25 fields: one extra": lambda body: body | {"ticker": "NVDA"},
    "numeric string": lambda body: body | {"mom_21": "0.32"},
    "boolean": lambda body: body | {"mom_21": True},
    "null where HW4 never had a missing value": lambda body: body | {"vol_63": None},
    "list in place of a number": lambda body: body | {"mom_21": [0.32]},
    "just above the highest rank": lambda body: body | {"mom_21": 0.5001},
    "just below the lowest rank": lambda body: body | {"mom_21": -0.5001},
    "raw dollar volume instead of its rank": lambda body: body | {"dollar_vol_63": 2.5e9},
    "a list of 24 numbers instead of named fields": lambda body: list(body.values()),
    "features nested under a key": lambda body: {"features": body},
    "empty object": lambda body: {},
}


@pytest.mark.parametrize("make_invalid", INVALID_BODIES.values(), ids=INVALID_BODIES.keys())
def test_predict_invalid_input(client, valid_input_data, predict_spy, make_invalid):
    response = client.post("/predict", json=make_invalid(valid_input_data))
    assert response.status_code == 422
    assert predict_spy.call_count == 0  # rejected before inference


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_predict_rejects_non_finite_numbers(client, valid_input_data, predict_spy, token):
    """Python's JSON parser accepts these tokens, so they arrive as real non-finite floats.

    They need a raw body: the test client refuses to serialize them, as a strict client would.
    """
    body = json.dumps(valid_input_data | {"mom_63": "TOKEN"}).replace('"TOKEN"', token)
    response = client.post("/predict", content=body, headers=JSON_BODY)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "mom_63"]
    assert predict_spy.call_count == 0


def test_predict_rejects_malformed_json(client, predict_spy):
    response = client.post("/predict", content='{"mom_21": 0.1,', headers=JSON_BODY)
    assert response.status_code == 422
    assert predict_spy.call_count == 0


@pytest.mark.parametrize(
    "body",
    [b'{"mom_21": "\xff"}', b'{"mom_21": ' + b"1" * 5000 + b"}"],
    ids=["invalid UTF-8", "5,000-digit integer"],
)
def test_predict_rejects_a_body_that_cannot_be_decoded(client, predict_spy, body):
    """FastAPI answers 400 rather than 422 when it cannot turn the body into JSON values."""
    response = client.post("/predict", content=body, headers=JSON_BODY)
    assert response.status_code == 400
    assert predict_spy.call_count == 0


def test_validation_error_names_the_field_and_not_the_value(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data | {"mom_21": "0.32"})
    (problem,) = response.json()["detail"]
    assert problem["loc"] == ["body", "mom_21"] and problem["type"] == "float_type"
    assert set(problem) == {"loc", "msg", "type"}


# ---------------------------------------------------------------------------- inference failure
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated failure with an internal detail")

    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        patch.setattr(app.state.model.pipeline, "predict", broken_predict)
        response = failure_client.post("/predict", json=valid_input_data)

    # A visible error with a generic body: no prediction, nothing internal.
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "internal detail" not in response.text
    # The traceback is in the server log, under the id the client received.
    (record,) = [r for r in caplog.records if "inference_failed" in r.getMessage()]
    assert response.headers["X-Request-ID"] in record.getMessage()
    assert "internal detail" in str(record.exc_info[1])
    # One failed request does not take the service down.
    assert failure_client.post("/predict", json=valid_input_data).status_code == 200


def test_non_finite_prediction_is_an_error(failure_client, valid_input_data, monkeypatch):
    """A NaN from the model must become a 500, not a 200 carrying a useless number."""
    not_a_number = Mock(return_value=np.array([np.nan], dtype=np.float32))
    monkeypatch.setattr(app.state.model.pipeline, "predict", not_a_number)
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500
    assert not_a_number.call_count == 1


def test_each_layer_refuses_a_non_finite_prediction(valid_input_data, monkeypatch):
    """The model logic and the response schema each stop a NaN on their own."""
    model = inference.load_model()
    monkeypatch.setattr(model.pipeline, "predict", Mock(return_value=np.array([np.nan])))
    with pytest.raises(ValueError, match="non-finite"):
        inference.predict_one(model, valid_input_data)
    with pytest.raises(ValidationError):
        OutputData(prediction=float("nan"), model_version=MODEL_VERSION)


# -------------------------------------------------------------------------------- model loading
def test_request_fields_are_the_hw4_features_in_training_order(client):
    assert tuple(InputData.model_fields) == app.state.model.features
    assert list(app.state.model.features) == METADATA["features"]
    assert len(app.state.model.features) == 24


def test_model_is_loaded_once_and_requests_never_load_or_fit(valid_input_data, monkeypatch):
    loader = Mock(wraps=inference.load_model)
    monkeypatch.setattr("app.load_model", loader)  # the name the lifespan calls
    forbidden = Mock(side_effect=AssertionError("a request loaded or fitted a model"))
    with TestClient(app) as test_client:
        # Startup is over. From here on, loading or fitting anything raises.
        pipeline = app.state.model.pipeline
        monkeypatch.setattr(inference.joblib, "load", forbidden)
        for estimator in (pipeline, *pipeline.named_steps.values()):
            monkeypatch.setattr(type(estimator), "fit", forbidden)
        for _ in range(3):
            assert test_client.post("/predict", json=valid_input_data).status_code == 200
    assert loader.call_count == 1
    assert forbidden.call_count == 0


def test_startup_fails_if_request_schema_and_model_disagree(monkeypatch):
    reordered = dataclasses.replace(inference.load_model(), features=("mom_63", "mom_21"))
    monkeypatch.setattr("app.load_model", lambda: reordered)
    with pytest.raises(RuntimeError, match="InputData"), TestClient(app):
        pass


def test_load_refuses_a_file_that_is_not_the_exported_artifact(tmp_path):
    altered = tmp_path / "model.joblib"
    altered.write_bytes(inference.MODEL_PATH.read_bytes() + b"\x00")
    with pytest.raises(RuntimeError, match="not the artifact"):
        inference.load_model(model_path=altered)


def test_load_refuses_another_scikit_learn_version(monkeypatch):
    monkeypatch.setattr(inference.sklearn, "__version__", "1.7.2")
    with pytest.raises(RuntimeError, match="fitted with scikit-learn"):
        inference.load_model()


# ---------------------------------------------------------------------------------- performance
def test_predict_performance(client, valid_input_data, capsys):
    """Latency of POST /predict through the in-process TestClient (no socket, no network)."""
    for _ in range(WARMUP_REQUESTS):
        client.post("/predict", json=valid_input_data)

    latencies_ms = []
    for _ in range(MEASURED_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median, p95 = np.percentile(latencies_ms, [50, 95])
    with capsys.disabled():  # show the report even without ``pytest -s``
        print(
            f"\n[latency] POST /predict, in-process TestClient, {MEASURED_REQUESTS} requests "
            f"after {WARMUP_REQUESTS} warm-up: median {median:.2f} ms, p95 {p95:.2f} ms "
            f"({platform.machine()}, {platform.system()}, Python {platform.python_version()})"
        )
    assert p95 < P95_BUDGET_MS
