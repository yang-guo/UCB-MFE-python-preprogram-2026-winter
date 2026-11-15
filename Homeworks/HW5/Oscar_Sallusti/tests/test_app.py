"""Tests for the HW5 service. All of them use the real saved HW4 model."""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import InputData, app
from model_service import FEATURES, SECTORS

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((Path(__file__).parent / "reference_prediction.json").read_text())
METADATA = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())

# Predictions are ranks on a 0-1 scale and span about 0.29-0.59 across stocks on
# one date. 1e-9 is far below any difference that matters, but loose enough for
# floating-point differences between platforms (e.g. macOS vs the Linux image).
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
    """The request shipped as example_request.json: AAPL on the last HW4 signal date."""
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def spy(client, monkeypatch):
    """Count calls to the real pipeline's predict."""
    pipeline = app.state.model.pipeline
    real_predict = pipeline.predict
    calls = []

    def counting_predict(X):
        calls.append(X)
        return real_predict(X)

    monkeypatch.setattr(pipeline, "predict", counting_predict)
    return calls


def direct_prediction(payload: dict) -> float:
    """Call the loaded pipeline without the API or the service's own frame builder."""
    frame = pd.DataFrame([payload])[[*FEATURES, "sector"]]
    frame[list(FEATURES)] = frame[list(FEATURES)].astype("float32")
    return float(app.state.model.pipeline.predict(frame)[0])


# ---------------------------------------------------------------- 1. valid input


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == METADATA["model_version"] == "hw4-ridge-fit-2026-04-01"
    assert response.headers["X-Request-ID"]

    # Same number as calling the pipeline directly.
    assert body["prediction"] == pytest.approx(
        direct_prediction(valid_input_data), abs=TOLERANCE
    )

    # Same number as HW4's own predict() gave for this row before any API existed
    # (recorded by export_model.py; see REFERENCE["source"]).
    assert valid_input_data == REFERENCE["input"]
    assert body["prediction"] == pytest.approx(REFERENCE["expected_prediction"], abs=TOLERANCE)


def test_predict_accepts_null_for_features_missing_in_training(client, valid_input_data):
    payload = {**valid_input_data, "roe": None, "mom_12_1": None}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction == pytest.approx(direct_prediction(payload), abs=TOLERANCE)
    # Missing values change the answer: they are imputed and flagged, not ignored.
    assert prediction != pytest.approx(REFERENCE["expected_prediction"], abs=TOLERANCE)


def test_predict_accepts_boundary_rank_of_one(client, valid_input_data):
    # The top-ranked stock has rank exactly 1; JSON may carry that as the integer 1.
    response = client.post("/predict", json={**valid_input_data, "log_mcap": 1})
    assert response.status_code == 200


def test_schema_matches_the_fitted_artifact(client):
    """The documented contract is the one the saved model was trained with."""
    model = app.state.model
    fields = list(InputData.model_fields)
    assert fields == [*FEATURES, "sector"]
    assert list(model.pipeline.feature_names_in_) == fields

    nullable = {
        name
        for name, field in InputData.model_fields.items()
        if type(None) in getattr(field.annotation, "__args__", ())
    }
    assert nullable == set(model.nullable_features)

    encoder = model.pipeline.named_steps["pre"].named_transformers_["sec"]
    assert set(encoder.categories_[0]) <= set(SECTORS)
    assert set(InputData.model_fields["sector"].annotation.__args__) == set(SECTORS)


def test_health_reports_loaded_version(client):
    assert client.get("/health").json() == {
        "status": "ready",
        "model_version": METADATA["model_version"],
    }


# ---------------------------------------------------------------- 2. invalid input


def without(key):
    return lambda payload: {k: v for k, v in payload.items() if k != key}


def replace(key, value):
    return lambda payload: {**payload, key: value}


INVALID_CASES = {
    "missing numeric feature": without("mom_1m"),
    "missing sector": without("sector"),
    "extra field": replace("pe_ratio", 0.5),
    "numeric string": replace("vol_63", "0.35"),
    "boolean": replace("vol_63", True),
    "list instead of number": replace("vol_63", [0.35]),
    "rank of zero": replace("mom_3m", 0),
    "negative rank": replace("mom_3m", -0.1),
    "rank above one": replace("mom_3m", 1.01),
    "raw value instead of rank": replace("filing_age_days", 45),
    "null for a never-missing feature": replace("vol_63", None),
    "unknown sector": replace("sector", "Crypto"),
    "sector with wrong case": replace("sector", "technology"),
    "numeric sector": replace("sector", 3),
    "null sector": replace("sector", None),
    "empty object": lambda payload: {},
    "feature list (wrong shape)": lambda payload: [payload[name] for name in FEATURES],
    "too few features in a list": lambda payload: [0.5] * 5,
}


@pytest.mark.parametrize("make_invalid", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_predict_invalid_input(client, valid_input_data, spy, make_invalid):
    response = client.post("/predict", json=make_invalid(valid_input_data))

    assert response.status_code == 422
    assert response.json()["detail"]  # says which field failed and why
    assert spy == []  # the model was never called


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_numbers(client, valid_input_data, spy, token):
    # NaN/Infinity are not valid JSON, but Python's parser accepts them, so the
    # body has to be written by hand.
    body = json.dumps({**valid_input_data, "vol_63": "@"}).replace('"@"', token)
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert spy == []


def test_predict_rejects_malformed_json(client, spy):
    response = client.post(
        "/predict", content='{"mom_1m": ', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert spy == []


# ---------------------------------------------------------------- 3. inference failure


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated failure with internal detail /secret/path")

    monkeypatch.setattr(app.state.model.pipeline, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "secret" not in response.text  # the exception text stays in the server log
    assert response.headers["X-Request-ID"]


def test_non_finite_prediction_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.model.pipeline, "predict", lambda X: [float("nan")])
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


# ---------------------------------------------------------------- 4. performance


def test_predict_performance(client, valid_input_data):
    """In-process latency: TestClient calls the app directly, with no network.

    The budget is a regression guard, not a service-level target: on the laptop
    used for development the p95 is a few milliseconds, so 250 ms leaves room
    for a slow shared CI runner while still catching, say, the model being
    reloaded on every request. Run with ``pytest -s`` to see the numbers.
    """
    for _ in range(10):  # warm up imports, caches and the first-call paths
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(200):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[-1]
    print(
        f"\nLatency over {len(latencies_ms)} in-process TestClient requests "
        f"({platform.machine()}, {platform.system()}, Python {platform.python_version()}): "
        f"median {median:.2f} ms, p95 {p95:.2f} ms, max {max(latencies_ms):.2f} ms"
    )
    assert p95 < 250
