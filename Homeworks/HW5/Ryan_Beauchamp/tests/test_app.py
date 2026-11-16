"""Tests for the HW5 prediction service (app.py) and its model logic (model_service.py).

Reference results come from two independent places:
* hw4_reference.json: scores READ FROM HW4's recorded scores.parquet (written by the HW4
  walk-forward run) before this API existed. The exported model reproduced all 365,938
  HW4 scores for its fit with a max abs diff of 0.0.
* a direct call on a separately loaded pipeline (load_artifact + predict_one), so the API
  is compared with the plain model call, not with itself.
"""

import json
import logging
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import sklearn
from fastapi.testclient import TestClient

from app import InputData, Sector, app
from model_service import METADATA_FILE, MODEL_FILE, load_artifact, predict_one

ARTIFACT_DIR = Path(__file__).resolve().parents[1]  # the submission folder
EXAMPLE = json.loads((ARTIFACT_DIR / "example_request.json").read_text())
REFERENCES = json.loads((ARTIFACT_DIR / "hw4_reference.json").read_text())

# The documented input bounds (README contract), written out on purpose rather than
# imported from app.py, so a change to app.BOUNDS cannot silently change what is tested.
EXPECTED_BOUNDS = {
    "momentum_21": (-1.0, 1000.0),
    "momentum_63": (-1.0, 1000.0),
    "momentum_252_21": (-1.0, 1000.0),
    "volatility_63": (0.0, 100.0),
    "drawdown_252": (-1.0, 0.0),
    "net_margin": (-1000.0, 1000.0),
    "operating_roa": (-1000.0, 1000.0),
    "debt_to_assets": (0.0, 1000.0),
    "cash_to_assets": (-1000.0, 1000.0),
    "fcf_to_assets": (-1000.0, 1000.0),
    "accruals_to_assets": (-1000.0, 1000.0),
    "filing_age_days": (0.0, 1095.0),
}

# Agreement required with the direct call and with HW4's recorded score. The observed diff
# on this machine was 0.0; the slack allows for different BLAS builds / CPU instruction
# sets in the Linux container, while staying far below any meaningful change in a score
# that lives on a -0.5..0.5 scale.
TOLERANCE = 1e-9

# p95 latency budget for one in-process request. Measured first: p95 was about 30 ms
# (median 22-25 ms) on the development laptop (Windows 11, 8-CPU Intel). 10x headroom
# allows for slower machines, CI runners and containers, while still catching a gross
# regression such as reloading the model on every request.
P95_BUDGET_MS = 300.0

SECRET = "secret internal detail"


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture
def client():
    # Real app with its lifespan: loads and verifies the saved model once.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    # Same app, but server exceptions become HTTP 500 responses instead of test errors.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    # The AAPL 2026-06-30 request from Stage 1 (a copy, so tests can modify it).
    return dict(EXAMPLE)


@pytest.fixture
def metadata():
    return json.loads((ARTIFACT_DIR / METADATA_FILE).read_text())


@pytest.fixture(scope="session")
def direct_pipeline():
    # Loaded independently of the app, so it is not the object behind the API.
    pipeline, _ = load_artifact(ARTIFACT_DIR)
    return pipeline


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Wrap the loaded pipeline's predict: record each call, then delegate to the original."""
    pipeline = client.app.state.pipeline
    original = pipeline.predict
    calls = []

    def spy(X):
        calls.append(len(X))  # number of rows passed to the model
        return original(X)

    monkeypatch.setattr(pipeline, "predict", spy)
    return calls


def with_value(**changes):
    """The example request with some fields replaced."""
    body = dict(EXAMPLE)
    body.update(changes)
    return body


def without(name):
    """The example request with one field removed."""
    body = dict(EXAMPLE)
    del body[name]
    return body


def send(client, payload):
    """POST a payload: a str is sent as raw JSON text, anything else via json=."""
    if isinstance(payload, str):
        # Raw text lets us send NaN/Infinity, which httpx's json= refuses to encode.
        return client.post(
            "/predict", content=payload, headers={"content-type": "application/json"}
        )
    return client.post("/predict", json=payload)


# ---------------------------------------------------------------- valid predictions


@pytest.mark.parametrize("reference", REFERENCES, ids=[r["symbol"] for r in REFERENCES])
def test_predict_valid_input(client, metadata, direct_pipeline, reference):
    # WHY: the core contract - real saved model, correct schema, and the same number as
    # both the plain pipeline call and the score HW4 itself recorded for this row.
    response = client.post("/predict", json=reference["features"])
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == metadata["model_version"]
    assert isinstance(body["prediction"], float)
    assert math.isfinite(body["prediction"])

    direct = predict_one(direct_pipeline, metadata["feature_order"], reference["features"])
    assert body["prediction"] == pytest.approx(direct, abs=TOLERANCE)
    assert body["prediction"] == pytest.approx(reference["hw4_score"], abs=TOLERANCE)


def test_integer_json_value_accepted(client, valid_input_data):
    # WHY: JSON has one number type; clients may send 59 instead of 59.0 for a day count.
    # The contract accepts whole numbers for float fields and must score them identically.
    assert valid_input_data["filing_age_days"] == 59.0
    as_float = client.post("/predict", json=valid_input_data)
    as_int = client.post("/predict", json=with_value(filing_age_days=59))
    assert as_float.status_code == as_int.status_code == 200
    assert as_int.json()["prediction"] == as_float.json()["prediction"]


# ---------------------------------------------------------------- invalid input -> 422


def bound_cases():
    """Each feature just outside its range: the nearest float below lower / above upper."""
    cases = []
    for name, (lower, upper) in EXPECTED_BOUNDS.items():
        below = math.nextafter(lower, -math.inf)
        above = math.nextafter(upper, math.inf)
        cases.append(pytest.param(with_value(**{name: below}), id=f"{name}-below-lower"))
        cases.append(pytest.param(with_value(**{name: above}), id=f"{name}-above-upper"))
    return cases


INVALID_CASES = [
    # Every required field missing, one at a time (12 numeric + sector).
    *[pytest.param(without(name), id=f"missing-{name}") for name in EXAMPLE],
    pytest.param(with_value(foo=1.0), id="extra-field"),
    pytest.param(with_value(momentum_5=0.1), id="extra-13th-numeric-field"),
    pytest.param(with_value(momentum_21="0.1"), id="numeric-string"),
    pytest.param(with_value(momentum_21=True), id="boolean"),
    pytest.param(with_value(momentum_21=None), id="null"),
    # Non-finite numbers only exist as raw JSON text (Python's json writes NaN/Infinity).
    pytest.param(json.dumps(with_value(momentum_21=math.nan)), id="nan"),
    pytest.param(json.dumps(with_value(momentum_21=math.inf)), id="infinity"),
    pytest.param(json.dumps(with_value(momentum_21=-math.inf)), id="minus-infinity"),
    pytest.param(with_value(sector="Crypto"), id="unknown-sector"),
    pytest.param(with_value(sector="technology"), id="wrong-case-sector"),
    pytest.param(with_value(sector=1), id="sector-number"),
    pytest.param(with_value(sector=None), id="sector-null"),
    pytest.param(with_value(sector=["Technology"]), id="sector-list"),
    *bound_cases(),
    pytest.param({}, id="empty-object"),
    # Wrong shape: a bare list of the 12 numeric values instead of the named-field object.
    pytest.param([EXAMPLE[name] for name in EXPECTED_BOUNDS], id="array-of-12-numbers"),
]


@pytest.mark.parametrize("payload", INVALID_CASES)
def test_predict_invalid_input(client, predict_spy, payload):
    # WHY: every contract violation must be rejected as a client error (422) BEFORE the
    # model runs - never scored with a guessed value and never turned into a 500.
    response = send(client, payload)
    assert response.status_code == 422
    assert predict_spy == []  # the model was never called


def test_spy_counts_valid_call(client, predict_spy, valid_input_data):
    # WHY: proves the spy really sees model calls, so "predict_spy == []" above means
    # something. A valid request makes exactly one call with one row.
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert predict_spy == [1]


AT_BOUND_CASES = [
    pytest.param(name, value, id=f"{name}-at-{side}")
    for name, (lower, upper) in EXPECTED_BOUNDS.items()
    for side, value in (("lower", lower), ("upper", upper))
]


@pytest.mark.parametrize(("name", "value"), AT_BOUND_CASES)
def test_predict_at_bounds_accepted(client, name, value):
    # WHY: bounds are inclusive; a value exactly on a limit is valid and must be scored.
    response = client.post("/predict", json=with_value(**{name: value}))
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def test_422_body_omits_input(client):
    # WHY: pins the custom validation handler. FastAPI's default 422 body echoes "input";
    # JSON cannot encode NaN, so echoing it would crash the error response into a 500.
    response = send(client, json.dumps(with_value(momentum_21=math.nan)))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    for item in detail:
        assert {"loc", "msg", "type"} <= set(item)
        assert "input" not in item


# ---------------------------------------------------------------- inference failure -> 500


def raise_secret(X):
    raise RuntimeError(SECRET)


def return_nan(X):
    return np.array([np.nan])


@pytest.mark.parametrize("broken_predict", [raise_secret, return_nan], ids=["raises", "nan"])
def test_predict_model_failure(
    failure_client, valid_input_data, monkeypatch, caplog, broken_predict
):
    # WHY: an inference failure on VALID input must be a server error (500) with a generic
    # message - no fallback prediction, no internal details leaked - and it must be logged.
    monkeypatch.setattr(failure_client.app.state.pipeline, "predict", broken_predict)
    with caplog.at_level(logging.ERROR, logger="app"):
        response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "prediction" not in response.json()
    assert SECRET not in response.text
    assert response.json() == {"detail": "Prediction failed"}
    assert "Prediction failed" in caplog.text
    if broken_predict is raise_secret:
        # The full detail goes to the server log (with the traceback), and only there.
        assert SECRET in caplog.text


# ---------------------------------------------------------------- performance


def test_predict_performance(client, valid_input_data, capsys):
    # WHY: documents single-request latency and catches gross regressions
    # (e.g. reloading the model on every request).
    for _ in range(5):  # warm-up: first calls pay one-off import/allocation costs
        assert client.post("/predict", json=valid_input_data).status_code == 200
    durations_ms = []
    for _ in range(100):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median_ms = float(np.median(durations_ms))
    p95_ms = float(np.percentile(durations_ms, 95))
    with capsys.disabled():  # show the report even under pytest -q
        print(
            "\n--- /predict latency report ---\n"
            f"requests: {len(durations_ms)} (after 5 warm-up)\n"
            f"median: {median_ms:.3f} ms | p95: {p95_ms:.3f} ms "
            f"(budget {P95_BUDGET_MS:g} ms)\n"
            "method: in-process TestClient, includes validation, JSON, model.predict; "
            "excludes network and uvicorn\n"
            f"platform: {platform.platform()} | processor: {platform.processor()} | "
            f"cpus: {os.cpu_count()}\n"
            f"python {platform.python_version()} | sklearn {sklearn.__version__} | "
            f"numpy {np.__version__}"
        )
    assert p95_ms < P95_BUDGET_MS


# ---------------------------------------------------------------- contract and artifact checks


def test_input_contract_matches_metadata(metadata):
    # WHY: the model was trained on columns in this exact order and saw exactly these
    # sectors; the API contract must not drift from the saved artifact.
    assert list(InputData.model_fields) == metadata["feature_order"]
    assert list(get_args(Sector)) == metadata["sector_categories"]


def copy_artifact(destination):
    """Copy only the two files load_artifact reads into a scratch folder."""
    for name in (MODEL_FILE, METADATA_FILE):
        shutil.copy2(ARTIFACT_DIR / name, destination / name)


def test_load_artifact_rejects_modified_model(tmp_path):
    # WHY: a corrupted or swapped model file must fail at startup, not serve wrong scores.
    copy_artifact(tmp_path)
    load_artifact(tmp_path)  # control: the untouched copy loads fine
    data = bytearray((tmp_path / MODEL_FILE).read_bytes())
    data[len(data) // 2] ^= 0xFF  # flip one byte
    (tmp_path / MODEL_FILE).write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="sha256"):
        load_artifact(tmp_path)


def test_load_artifact_rejects_library_version_mismatch(tmp_path):
    # WHY: sklearn pickles are only reliable with the training versions; a mismatch must
    # fail fast with a clear error instead of loading silently.
    copy_artifact(tmp_path)
    metadata_path = tmp_path / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata["library_versions"]["sklearn"] = "0.0.0"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="sklearn"):
        load_artifact(tmp_path)
