"""API tests for the HW4 model service (the four HW5 test groups).

Reference results: ``tests/hw4_reference_predictions.json`` holds three stocks
from the last HW4 signal date (2026-09-28) with the scores that my HW4 notebook
wrote to ``outputs/test_predictions.parquet`` (quarterly walk-forward of
``RidgeRanker("y_rank", 1000)`` on the full database), before any HW5 code
existed. ``scripts/export_hw4_model.py`` copied them out unchanged.
"""

import copy
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

from app import InputData, app
from model_pipeline import INPUT_COLUMNS, to_model_frame

REFERENCE = json.loads(
    (Path(__file__).parent / "hw4_reference_predictions.json").read_text(encoding="utf-8")
)
# Same float64 linear algebra on the same inputs. Batch scores matched HW4 exactly
# (model_metadata.json "export_checks"); a single-row call differs by ~1 ulp (~1e-17)
# because BLAS sums in a different order. 1e-9 allows for other machines/BLAS builds and
# is 45,000x smaller than the smallest gap between the 10th and 11th scores (HW4's buy
# cutoff) in any week this fit scored (4.5e-5), so it cannot change which stocks HW4 buys.
TOLERANCE = REFERENCE["tolerance_abs"]


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
    # AAPL on 2026-09-28: all 28 features present (same body as example_request.json).
    return copy.deepcopy(REFERENCE["cases"][0]["request"])


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Wrap the loaded pipeline's predict so a test can count model calls."""
    pipeline = app.state.model.pipeline
    real_predict = pipeline.predict
    calls = []

    def spy(frame):
        calls.append(frame)
        return real_predict(frame)

    monkeypatch.setattr(pipeline, "predict", spy)
    return calls


def post_raw(client, text):
    return client.post("/predict", content=text, headers={"Content-Type": "application/json"})


# ------------------------------------------------------------ 1. valid prediction
def test_predict_valid_input(client, valid_input_data, predict_spy):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == app.state.model.metadata["model_version"]
    assert response.headers["X-Request-ID"]
    assert len(predict_spy) == 1  # one model call per request, never a refit

    # Direct call of the same saved pipeline, outside the API, on the training representation.
    direct = app.state.model.pipeline.predict(to_model_frame(valid_input_data))
    assert body["prediction"] == pytest.approx(float(direct[0]), abs=TOLERANCE)

    # Known result recorded by my HW4 code before the API existed.
    expected = REFERENCE["cases"][0]["expected_prediction"]
    assert body["prediction"] == pytest.approx(expected, abs=TOLERANCE)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: f"{c['case']}-{c['symbol']}")
def test_predictions_match_recorded_hw4_scores(client, case):
    # Includes ALMR, whose request has null features (HW4's missing-value path).
    response = client.post("/predict", json=case["request"])
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(
        case["expected_prediction"], abs=TOLERANCE
    )


def test_missing_feature_values_follow_hw4_imputation(client, valid_input_data):
    """null for a rank = HW4's NaN: rank imputed at 0.5 plus a missing flag (not plain 0.5)."""
    as_null = dict(valid_input_data, r_roe=None)
    as_median = dict(valid_input_data, r_roe=0.5)
    null_pred = client.post("/predict", json=as_null).json()["prediction"]
    median_pred = client.post("/predict", json=as_median).json()["prediction"]

    ridge = app.state.model.pipeline.named_steps["ridge"]
    names = list(app.state.model.pipeline.named_steps["design"].get_feature_names_out())
    missing_coef = ridge.coef_[names.index("r_roe_missing")]
    assert null_pred - median_pred == pytest.approx(missing_coef, abs=TOLERANCE)


def test_integer_json_number_is_an_exact_float(client, valid_input_data):
    """JSON 1 and 1.0 are the same number; only strings/booleans are refused."""
    as_int = client.post("/predict", json=dict(valid_input_data, r_log_mcap=1))
    as_float = client.post("/predict", json=dict(valid_input_data, r_log_mcap=1.0))
    assert as_int.status_code == as_float.status_code == 200
    assert as_int.json() == as_float.json()


def test_contract_matches_saved_artifact(client):
    design = app.state.model.pipeline.named_steps["design"]
    assert list(InputData.model_fields) == INPUT_COLUMNS
    assert list(design.feature_names_in_) == INPUT_COLUMNS
    assert app.state.model.metadata["input_columns"] == INPUT_COLUMNS


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": REFERENCE["model_version"],
    }


# ------------------------------------------------------------ 2. input validation
def drop(key):
    return lambda body: {k: v for k, v in body.items() if k != key}


def setv(key, value):
    return lambda body: {**body, key: value}


INVALID_JSON_BODIES = {
    "missing_feature_27_of_28": drop("r_roe"),
    "missing_sector": drop("sector"),
    "extra_field": setv("symbol", "AAPL"),
    "extra_feature_29_of_28": setv("r_extra_feature", 0.5),
    "numeric_string": setv("r_mom_5", "0.67"),
    "boolean_true": setv("r_mom_5", True),
    "boolean_false": setv("r_mom_5", False),
    "list_value": setv("r_mom_5", [0.67]),
    "object_value": setv("r_mom_5", {"value": 0.67}),
    "rank_zero": setv("r_mom_5", 0.0),
    "rank_negative": setv("r_mom_5", -0.1),
    "rank_above_one": setv("r_mom_5", 1.5),
    "raw_return_instead_of_rank": setv("r_mom_21", 12.0),
    "sector_unknown_label": setv("sector", "Tech"),
    "sector_wrong_case": setv("sector", "technology"),
    "sector_number": setv("sector", 7),
    "sector_null": setv("sector", None),
    "empty_object": lambda body: {},
    "fixed_length_list_body": lambda body: list(body.values()),
    "list_body_wrong_length": lambda body: list(body.values())[:-2],
}


@pytest.mark.parametrize("make_body", INVALID_JSON_BODIES.values(), ids=INVALID_JSON_BODIES)
def test_predict_invalid_input(client, valid_input_data, predict_spy, make_body):
    response = client.post("/predict", json=make_body(valid_input_data))
    assert response.status_code == 422
    assert response.json()["detail"]  # FastAPI-style list of field errors
    assert predict_spy == []  # rejected before inference: the model was never called


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_numbers_are_rejected(client, valid_input_data, predict_spy, token):
    # Python's JSON parser turns these tokens into nan/inf, so they reach validation.
    text = json.dumps(dict(valid_input_data, r_mom_5="__X__")).replace('"__X__"', token)
    response = post_raw(client, text)
    assert response.status_code == 422  # not 500: the 422 body must not echo the NaN back
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "r_mom_5"]
    # Rejected by the finiteness rule itself, not only as a side effect of the (0, 1] range.
    assert error["type"] == "finite_number"
    assert predict_spy == []


@pytest.mark.parametrize("text", ["", "not json", '{"r_mom_5": 0.5,'])
def test_malformed_body_is_rejected(client, predict_spy, text):
    assert post_raw(client, text).status_code == 422
    assert predict_spy == []


# ------------------------------------------------------------ 3. inference failure
def raise_error(frame):
    raise RuntimeError("simulated failure: secret internal detail")


INFERENCE_FAILURES = {
    "predict_raises": raise_error,
    "predict_returns_nan": lambda frame: np.array([np.nan]),
    "predict_returns_wrong_shape": lambda frame: np.array([0.1, 0.2]),
}


@pytest.mark.parametrize("broken_predict", INFERENCE_FAILURES.values(), ids=INFERENCE_FAILURES)
def test_predict_model_failure(
    failure_client, valid_input_data, monkeypatch, caplog, broken_predict
):
    monkeypatch.setattr(app.state.model.pipeline, "predict", broken_predict)

    with caplog.at_level("ERROR", logger="uvicorn.error"):
        response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "prediction" not in response.text  # no fallback/default prediction
    assert "secret internal detail" not in response.text  # details stay in the log
    request_id = response.headers["X-Request-ID"]
    assert f"inference_failed request_id={request_id}" in caplog.text
    assert caplog.records[-1].exc_info is not None  # traceback logged server-side


# ------------------------------------------------------------ 4. performance
WARMUP_REQUESTS = 5
MEASURED_REQUESTS = 50
# Budget: on my laptop the in-process median is about 2 ms (see README). A p95 of
# 250 ms leaves >50x headroom for slower CI runners or Docker on a VM, but still
# fails if the service does per-request work it should not (reloading/refitting
# the model, or reading the 1.4M-row training panel).
P95_BUDGET_SECONDS = 0.250


def test_predict_performance(client, valid_input_data, capsys):
    """In-process TestClient measurement (no network): ASGI + validation + inference.

    Real localhost HTTP against uvicorn or the container is measured separately with
    ``scripts/measure_latency.py`` (results in README.md).
    """
    for _ in range(WARMUP_REQUESTS):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies = []
    for _ in range(MEASURED_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies.append(time.perf_counter() - start)
        assert response.status_code == 200

    median = statistics.median(latencies)
    p95 = float(np.percentile(latencies, 95))
    report = (
        f"\n[latency] in-process TestClient, {MEASURED_REQUESTS} requests after "
        f"{WARMUP_REQUESTS} warm-up: median {median * 1000:.2f} ms, p95 {p95 * 1000:.2f} ms "
        f"(Python {platform.python_version()}, {platform.system()} {platform.release()}, "
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs)"
    )
    with capsys.disabled():  # show the report even under `pytest -q`
        print(report)
    assert len(latencies) >= 20
    assert p95 < P95_BUDGET_SECONDS
