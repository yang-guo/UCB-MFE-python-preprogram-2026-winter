"""Tests for the HW5 service serving my HW4 model.

Reference results come from artifacts/reference_prediction.json, written by export_model.py:
real HW4 final-test observations scored by the HW4 pipeline (built with hw4.fit) before any
API code existed. The positive tests use the real saved model; only the failure tests
replace its predict method.
"""

import json
import logging
import math
import platform
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model as inference
from app import app

REFERENCE = json.loads(
    (Path(inference.MODEL_PATH).parent / "reference_prediction.json").read_text()
)
FEATURES = REFERENCE["feature_order"]
EXAMPLES = [REFERENCE["reference"], *REFERENCE["additional_examples"]]
COMPLETE = REFERENCE["reference"]["features"]
# The saved model and the HW4 pipeline do the same float64 arithmetic, so they should agree
# exactly; 1e-12 allows only last-bit noise. Predictions differ across stocks by about 1e-4,
# so a looser tolerance could pass for the wrong input.
TOLERANCE = 1e-12


# Fixtures


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
    """The complete reference input: stock A on 2026-08-31, all eight features present."""
    return dict(COMPLETE)


@pytest.fixture(scope="module")
def artifact_version():
    """The version stored in the saved artifact itself, independent of the running app."""
    return joblib.load(inference.MODEL_PATH)["model_version"]


@pytest.fixture
def predict_calls(client, monkeypatch):
    """Spy on the loaded pipeline: record each input frame, then run the real predict."""
    pipeline = client.app.state.model.pipeline
    real_predict = pipeline.predict
    calls = []

    def spy(frame):
        calls.append(frame.copy())
        return real_predict(frame)

    monkeypatch.setattr(pipeline, "predict", spy)
    return calls


def post_raw(client, content):
    """POST a raw body, so NaN, Infinity, and malformed JSON reach the server unchanged."""
    return client.post(
        "/predict", content=content, headers={"Content-Type": "application/json"}
    )


def with_values(**changes):
    return json.dumps({**COMPLETE, **changes})


def without(name):
    return json.dumps({k: v for k, v in COMPLETE.items() if k != name})


def with_literal(name, token):
    """Replace one value with a raw JSON token such as NaN that json.dumps would not write."""
    return with_values(**{name: "__TOKEN__"}).replace('"__TOKEN__"', token)


# 1. Valid predictions


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["category"] for e in EXAMPLES])
def test_predict_valid_input(client, predict_calls, artifact_version, example):
    response = client.post("/predict", json=example["features"])

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == artifact_version == REFERENCE["model_version"]

    # Known result recorded from HW4 code before the API existed.
    assert body["prediction"] == pytest.approx(example["prediction"], abs=TOLERANCE)

    # Direct pipeline call on an independently built frame (the class method bypasses the spy).
    pipeline = client.app.state.model.pipeline
    direct_frame = pd.DataFrame([example["features"]], columns=FEATURES, dtype=float)
    direct = float(type(pipeline).predict(pipeline, direct_frame)[0])
    assert body["prediction"] == pytest.approx(direct, abs=TOLERANCE)

    # The model saw one named row in HW4's order, with nulls passed on as NaN.
    assert len(predict_calls) == 1
    seen = predict_calls[0]
    assert isinstance(seen, pd.DataFrame) and seen.shape == (1, len(FEATURES))
    assert list(seen.columns) == FEATURES
    assert list(seen.columns[seen.isna().iloc[0]]) == example["missing_features"]


def test_reference_examples_cover_null_handling():
    """The real examples include missing price and missing financial features."""
    categories = {e["category"]: e["missing_features"] for e in EXAMPLES}
    assert categories["complete"] == []
    assert categories["missing_price_feature"] == ["mom_252_21", "drawdown_252"]
    assert categories["missing_financial_feature"] == [
        "operating_roa",
        "accruals_to_assets",
        "debt_to_assets",
    ]


# 2. Input validation: 422, no echo of the input, and the model is never called

# Each case: raw body, and submitted text that must not come back in the response.
INVALID_INPUTS = {
    "missing_feature": (without("log_cap"), []),
    "extra_feature": (with_values(sector=0.123456789), ["0.123456789"]),
    "empty_object": ("{}", []),
    "empty_body": ("", []),
    "boolean_true": (with_values(mom_21=True), ["true"]),
    "boolean_false": (with_values(log_cap=False), ["false"]),
    "numeric_string": (with_values(vol_63="0.4321"), ["0.4321"]),
    "text_string": (with_values(vol_63="not-a-number"), ["not-a-number"]),
    "list_value": (with_values(mom_21=[0.987654321]), ["0.987654321"]),
    "object_value": (with_values(mom_21={"v": 0.55555}), ["0.55555"]),
    "below_zero": (with_values(operating_roa=-0.0123456), ["0.0123456"]),
    "above_one": (with_values(drawdown_252=1.0000001), ["1.0000001"]),
    "nan": (with_literal("mom_21", "NaN"), ["NaN", "nan"]),
    "infinity": (with_literal("vol_63", "Infinity"), ["Infinity"]),
    "negative_infinity": (with_literal("log_cap", "-Infinity"), ["Infinity"]),
    "overflow_1e400": (with_literal("debt_to_assets", "1e400"), ["1e400", "Infinity"]),
    "all_null": (json.dumps({name: None for name in FEATURES}), []),
    "array_not_object": (json.dumps([0.5] * len(FEATURES)), []),
    "malformed_json": ('{"mom_21": 0.5,', []),
}


@pytest.mark.parametrize("content, submitted", INVALID_INPUTS.values(), ids=INVALID_INPUTS)
def test_predict_invalid_input(client, predict_calls, content, submitted):
    response = post_raw(client, content)

    assert response.status_code == 422
    assert predict_calls == [], "a rejected request reached the model"
    # Only safe fields come back: no "input" or "ctx" echoing what was sent.
    for error in response.json()["detail"]:
        assert set(error) == {"type", "loc", "msg"}
    # None of the submitted values appear, including the valid ones in the same body.
    for text in [*submitted, repr(COMPLETE["vol_63"]), repr(COMPLETE["log_cap"])]:
        assert text not in response.text


@pytest.mark.parametrize("name", FEATURES)
def test_each_feature_accepts_null(client, predict_calls, name):
    response = client.post("/predict", json={**COMPLETE, name: None})

    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])
    assert list(predict_calls[0].columns[predict_calls[0].isna().iloc[0]]) == [name]


@pytest.mark.parametrize("name", FEATURES)
def test_each_feature_is_required(client, predict_calls, name):
    response = post_raw(client, without(name))

    assert response.status_code == 422
    assert response.json() == {
        "detail": [{"type": "missing", "loc": ["body", name], "msg": "Field required"}]
    }
    assert predict_calls == []


# 3. Inference failures stay errors and do not leak details


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    secret = "SECRET-TOKEN in /Users/someone/private/model_store.py line 42"

    def broken_predict(frame):
        raise RuntimeError(secret)

    monkeypatch.setattr(failure_client.app.state.model.pipeline, "predict", broken_predict)
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    for leak in ("SECRET-TOKEN", "/Users", "model_store.py", "RuntimeError", "Traceback"):
        assert leak not in response.text
    # The full exception is in the server log, tied to the id the client received.
    request_id = response.headers["X-Request-ID"]
    logged = [r for r in caplog.records if r.exc_info and secret in str(r.exc_info[1])]
    assert len(logged) == 1
    assert request_id in logged[0].getMessage()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_prediction_is_an_error(
    failure_client, valid_input_data, monkeypatch, bad_value
):
    monkeypatch.setattr(
        failure_client.app.state.model.pipeline, "predict", lambda frame: np.array([bad_value])
    )
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}


# 4. Lifecycle and health


def test_health_reports_loaded_model(client, artifact_version):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": artifact_version}


def test_model_loads_once_per_lifespan(monkeypatch, valid_input_data):
    loads = []
    real_load = inference.StockReturnModel.load

    def counting_load(cls, *args, **kwargs):
        loads.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(inference.StockReturnModel, "load", classmethod(counting_load))
    with TestClient(app) as test_client:
        first_model = test_client.app.state.model
        for _ in range(5):
            assert test_client.post("/predict", json=valid_input_data).status_code == 200
        assert test_client.app.state.model is first_model
    assert len(loads) == 1
    with TestClient(app):  # a new lifespan (a server restart) loads it again, once
        pass
    assert len(loads) == 2


# 5. Latency

WARMUP_REQUESTS = 5
TIMED_REQUESTS = 50
# Budgets for an in-process TestClient call (no network, no uvicorn). On an Apple M2 with
# 8 GB RAM (macOS, Python 3.12.9) this measured about 1.2 ms median and 1.5 ms p95: FastAPI
# validation, a one-row DataFrame, and an 8-coefficient Ridge. The budgets are about 40x
# and 130x those numbers, so a slower CI runner, Docker, or a busy laptop should not fail,
# while gross regressions still do: a blocking call, heavy per-request work such as
# rebuilding features, or an order-of-magnitude slowdown. (Reloading the 2 KB artifact per
# request would be too fast to see here; test_model_loads_once_per_lifespan covers that.)
MEDIAN_BUDGET_MS = 50.0
P95_BUDGET_MS = 200.0


def test_predict_performance(client, valid_input_data):
    for _ in range(WARMUP_REQUESTS):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    timings_ms = []
    for _ in range(TIMED_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median_ms = statistics.median(timings_ms)
    p95_ms = float(np.percentile(timings_ms, 95))
    print(
        f"\nLatency, in-process FastAPI TestClient (not real HTTP), {TIMED_REQUESTS} requests "
        f"after {WARMUP_REQUESTS} warm-ups, time.perf_counter: median {median_ms:.2f} ms, "
        f"p95 {p95_ms:.2f} ms, max {max(timings_ms):.2f} ms on {platform.platform()} "
        f"({platform.machine()}), Python {platform.python_version()}"
    )
    assert median_ms < MEDIAN_BUDGET_MS
    assert p95_ms < P95_BUDGET_MS
