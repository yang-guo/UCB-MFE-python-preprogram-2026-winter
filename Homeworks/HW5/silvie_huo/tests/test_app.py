"""Tests for POST /predict, run against the real saved HW4 model.

Reference result (test_predict_matches_hw4_reference): tests/reference_case.json is written in pre-processing. 
Its input is a real row from the HW4 panel, and its expected value is the fitted
pipeline's own .predict on that row, recorded before any API code existed. Tolerance: rel=1e-6,
abs=1e-9. The artifact and sklearn version are identical, 1e-6 is orders of magnitude 
below any economically meaningful difference in a 90-day return, and abs=1e-9 handles predictions near zero.

"""
import json
import logging
import math
import os
import platform
import statistics
import time
from pathlib import Path

import fastapi
import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

import app as app_module
from app import app

CONTRACT = app_module.CONTRACT
FEATURES = CONTRACT["features"]
NAMES = app_module.FEATURE_NAMES                       # training column order
NUMERIC = [f for f in FEATURES if f["kind"] == "numeric"]
CATEGORICAL = [f for f in FEATURES if f["kind"] == "categorical"]

HERE = Path(__file__).resolve().parent
REFERENCE_FILE = HERE / "reference_case.json"
PERF_REPORT = HERE / "perf_report.json"

REF_TOL = dict(rel=1e-6, abs=1e-9)                     # recorded reference vs API
DIRECT_TOL = dict(rel=1e-9, abs=1e-12)                 # same artifact, same code path

# Loose regression guard, not a benchmark: its job is to catch accidental per-request model
# loading/fitting (hundreds of ms to seconds), while a one-row in-process predict is a few ms.
# After you have measured your machine, set the env var to roughly 5-10x the observed p95.
P95_BUDGET_MS = float(os.environ.get("PREDICT_P95_BUDGET_MS", "250"))
WARMUP, N_TIMED = 5, 30


# ---------- fixtures ----------

@pytest.fixture
def client():
    with TestClient(app) as test_client:               # context manager runs startup/shutdown
        yield test_client


@pytest.fixture
def failure_client():
    # Returns the HTTP 500 instead of re-raising the server exception in the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    """Deterministic valid payload: midpoint of each numeric range, first allowed category."""
    return {f["name"]: ((f["min"] + f["max"]) / 2 if f["kind"] == "numeric" else f["categories"][0])
            for f in FEATURES}


@pytest.fixture(scope="module")
def saved_model():
    """Independent copy of the saved artifact for direct pipeline calls."""
    return joblib.load(app_module.ARTIFACTS / CONTRACT["model_file"])


@pytest.fixture(scope="module")
def reference_case():
    if not REFERENCE_FILE.exists():
        pytest.fail("tests/reference_case.json is missing; run `python export_model.py`")
    ref = json.loads(REFERENCE_FILE.read_text())
    if ref["model_sha256"] != CONTRACT["model_sha256"]:
        pytest.fail("reference_case.json belongs to a different model; re-run `python export_model.py`")
    return ref


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Replace the loaded model's predict with a recorder that returns a harmless value."""
    calls = []

    def spy(X):
        calls.append(X)
        return np.array([0.0])

    monkeypatch.setattr(app.state.model, "predict", spy)
    return calls


def direct_predict(model, row):
    """Same representation as training: one-row DataFrame, training column names and order."""
    X = pd.DataFrame([row], columns=NAMES).astype({f["name"]: "float64" for f in NUMERIC})
    return float(model.predict(X)[0])


def post_raw(client, body: str):
    # Raw text guarantees NaN/Infinity literals reach the server unchanged.
    return client.post("/predict", content=body, headers={"content-type": "application/json"})


# ---------- 1. valid prediction ----------

def test_predict_valid_input(client, valid_input_data, saved_model):
    r = client.post("/predict", json=valid_input_data)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()

    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == CONTRACT["model_version"]

    # API result equals a direct pipeline call on an independently loaded artifact
    assert body["prediction"] == pytest.approx(direct_predict(saved_model, valid_input_data), **DIRECT_TOL)

    # key order in the JSON must not matter: server rebuilds the row in training order
    reversed_payload = dict(reversed(list(valid_input_data.items())))
    r2 = client.post("/predict", json=reversed_payload)
    assert r2.status_code == 200 and r2.json()["prediction"] == body["prediction"]


def test_predict_matches_hw4_reference(client, reference_case):
    """Recorded HW4 input -> recorded HW4 prediction, sent through the API."""
    r = client.post("/predict", json=reference_case["input"])
    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx(reference_case["expected_prediction"], **REF_TOL)


def test_reference_matches_direct_call(saved_model, reference_case):
    """The saved artifact still reproduces the number recorded at export time."""
    assert direct_predict(saved_model, reference_case["input"]) == pytest.approx(
        reference_case["expected_prediction"], **REF_TOL)


def test_spy_sees_training_representation(client, valid_input_data, predict_spy):
    """Control for the validation tests (the spy does fire on valid input) and a representation check."""
    assert client.post("/predict", json=valid_input_data).status_code == 200
    assert len(predict_spy) == 1
    X = predict_spy[0]
    assert isinstance(X, pd.DataFrame) and list(X.columns) == NAMES and len(X) == 1


def test_openapi_documents_contract(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["InputData"]
    assert list(schema["properties"]) == NAMES
    assert set(schema["required"]) == set(NAMES)
    assert schema.get("additionalProperties") is False
    assert all(p.get("description") for p in schema["properties"].values())


# ---------- 2. input validation (422, model never called) ----------

def _drop(name):
    return lambda v: json.dumps({k: x for k, x in v.items() if k != name})


def _set(name, value):
    return lambda v: json.dumps({**v, name: value})        # NaN/Infinity serialize as literals


CASES = [
    ("missing_first_field", _drop(NAMES[0])),
    ("missing_last_field", _drop(NAMES[-1])),
    ("empty_object", lambda v: "{}"),
    ("extra_field", _set("unexpected_feature", 1.0)),
    ("list_body_exact_width", lambda v: json.dumps([0.0] * len(NAMES))),
    ("list_body_too_short", lambda v: json.dumps([0.0] * (len(NAMES) - 1))),
    ("list_body_too_long", lambda v: json.dumps([0.0] * (len(NAMES) + 1))),
    ("malformed_json", lambda v: "{not json"),
    ("null_body", lambda v: "null"),
]
if NUMERIC:
    n = NUMERIC[0]
    CASES += [
        ("numeric_as_string", _set(n["name"], "1.0")),
        ("numeric_null", _set(n["name"], None)),
        ("numeric_bool", _set(n["name"], True)),
        ("numeric_list", _set(n["name"], [1.0])),
        ("numeric_nan", _set(n["name"], float("nan"))),
        ("numeric_inf", _set(n["name"], float("inf"))),
        ("numeric_neg_inf", _set(n["name"], float("-inf"))),
        ("numeric_below_min", _set(n["name"], n["min"] - abs(n["min"]) - 1)),
        ("numeric_above_max", _set(n["name"], n["max"] + abs(n["max"]) + 1)),
    ]
if CATEGORICAL:
    c = CATEGORICAL[0]
    CASES += [
        ("category_not_allowed", _set(c["name"], "__not_a_category__")),
        ("category_as_number", _set(c["name"], 1.0)),
        ("category_null", _set(c["name"], None)),
    ]


@pytest.mark.parametrize("build_body", [c[1] for c in CASES], ids=[c[0] for c in CASES])
def test_predict_invalid_input(client, valid_input_data, predict_spy, build_body):
    r = post_raw(client, build_body(valid_input_data))
    assert r.status_code == 422
    assert "detail" in r.json()                        # the app's own 422 handler defines this body
    assert predict_spy == [], "rejected requests must not reach inference"


# ---------- 3. inference failure (500) ----------

@pytest.mark.parametrize("failure", ["raises", "nan", "inf"])
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog, failure):
    def broken(X):
        if failure == "raises":
            raise ValueError("secret internal detail")
        return np.array([float("nan") if failure == "nan" else float("inf")])

    monkeypatch.setattr(app.state.model, "predict", broken)   # mock only here; positive tests use the real model

    with caplog.at_level(logging.ERROR, logger=app_module.log.name):
        r = failure_client.post("/predict", json=valid_input_data)

    assert r.status_code == 500
    # No JSON body is assumed; only that nothing sensitive or successful leaks.
    assert "secret internal detail" not in r.text
    assert "Traceback" not in r.text
    assert "model_version" not in r.text
    assert any(rec.levelno >= logging.ERROR and rec.exc_info for rec in caplog.records)


# ---------- 4. performance ----------

def hardware_description():
    return {"platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "fastapi": fastapi.__version__}


def test_predict_performance(client, valid_input_data):
    """In-process TestClient timing (no network, no uvicorn) of POST /predict, one-row input.

    Method: WARMUP untimed requests, then N_TIMED requests, each timed with time.perf_counter()
    around client.post(...). Every timed request must return 200. p95 uses nearest-rank on the
    sorted samples. The budget is a regression guard (see P95_BUDGET_MS), not a benchmark.
    """
    for _ in range(WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    times_ms = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        r = client.post("/predict", json=valid_input_data)
        times_ms.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200

    times_ms.sort()
    result = {"measurement": "in-process fastapi TestClient (httpx->ASGI), not localhost HTTP",
              "timer": "time.perf_counter", "warmup_requests": WARMUP, "timed_requests": N_TIMED,
              "median_ms": statistics.median(times_ms),
              "p95_ms": times_ms[math.ceil(0.95 * len(times_ms)) - 1],
              "min_ms": times_ms[0], "max_ms": times_ms[-1],
              "p95_budget_ms": P95_BUDGET_MS, "hardware": hardware_description()}
    PERF_REPORT.write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result, indent=2))
    assert result["p95_ms"] < P95_BUDGET_MS