"""Checks for the served HW4 ridge-rank model."""

import json
import math
import os
import platform
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

from app import app
from contract import FEATURE_NAMES, MODEL_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example_request.json"
REFERENCE = ROOT / "artifacts" / "reference_prediction.json"
METADATA = ROOT / "artifacts" / "model_metadata.json"
ARTIFACT = ROOT / "artifacts" / "model.joblib"


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
    return json.loads(EXAMPLE.read_text())


@pytest.fixture
def reference_prediction():
    return json.loads(REFERENCE.read_text())


def test_artifact_matches_training_environment():
    metadata = json.loads(METADATA.read_text())
    assert metadata["sklearn_version"] == sklearn.__version__ == "1.8.0"
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["features"] == FEATURE_NAMES


def test_predict_valid_input(client, valid_input_data, reference_prediction):
    """The reference was saved by train_artifact.py before this API existed."""
    assert valid_input_data == reference_prediction["features"]
    assert reference_prediction["source"].startswith("Direct pipeline.predict")
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == MODEL_VERSION
    assert math.isfinite(body["prediction"])
    assert body["prediction"] == pytest.approx(reference_prediction["prediction"], abs=1e-6)

    pipeline = joblib.load(ARTIFACT)["pipeline"]
    frame = pd.DataFrame([valid_input_data])[FEATURE_NAMES]
    direct = float(pipeline.predict(frame)[0])
    assert body["prediction"] == pytest.approx(direct, abs=1e-6)


def _invalid_payloads(valid: dict) -> list[tuple[str, object]]:
    missing = dict(valid)
    missing.pop("filing_age_days")
    extra = dict(valid)
    extra["sector"] = "Technology"
    numeric_string = dict(valid)
    numeric_string["momentum_21"] = "0.01"
    boolean = dict(valid)
    boolean["volatility_63"] = True
    not_a_number = dict(valid)
    not_a_number["momentum_63"] = float("nan")
    positive_infinity = dict(valid)
    positive_infinity["operating_roa"] = float("inf")
    negative_infinity = dict(valid)
    negative_infinity["net_margin"] = float("-inf")
    momentum_at_minus_one = dict(valid)
    momentum_at_minus_one["momentum_21"] = -1
    momentum_too_large = dict(valid)
    momentum_too_large["momentum_252_21"] = 101
    negative_volatility = dict(valid)
    negative_volatility["volatility_63"] = -0.01
    positive_drawdown = dict(valid)
    positive_drawdown["drawdown_252"] = 0.01
    negative_filing_age = dict(valid)
    negative_filing_age["filing_age_days"] = -1
    null_feature = dict(valid)
    null_feature["cash_to_assets"] = None
    eleven_features = [0.01] * 11
    twelve_features = [0.0] * 12
    return [
        ("missing_field", missing),
        ("extra_field", extra),
        ("numeric_string", numeric_string),
        ("boolean", boolean),
        ("nan", not_a_number),
        ("inf", positive_infinity),
        ("negative_inf", negative_infinity),
        ("momentum_below_domain", momentum_at_minus_one),
        ("momentum_above_domain", momentum_too_large),
        ("negative_volatility", negative_volatility),
        ("positive_drawdown", positive_drawdown),
        ("negative_filing_age", negative_filing_age),
        ("null_feature", null_feature),
        ("empty_object", {}),
        ("eleven_value_list", eleven_features),
        ("twelve_value_list", twelve_features),
    ]


def test_predict_invalid_input(client, valid_input_data, monkeypatch):
    called = {"value": False}

    def mark(*args, **kwargs):
        called["value"] = True
        raise RuntimeError("invalid input reached inference")

    monkeypatch.setattr(app.state.pipeline, "predict", mark)
    for name, payload in _invalid_payloads(valid_input_data):
        response = client.post(
            "/predict",
            content=json.dumps(payload, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, name
        assert called["value"] is False, name


def test_integer_filing_age_is_accepted(client, valid_input_data):
    payload = dict(valid_input_data)
    payload["filing_age_days"] = int(payload["filing_age_days"])
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.pipeline, "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "simulated inference failure" not in response.text


def test_predict_performance(client, valid_input_data):
    """In-process TestClient timing, not a real socket.

    Median under one second is a local guard for this 12-feature Ridge. It is
    not a serving SLO. One warmup request is excluded from the 20 samples.
    Percentiles use the nearest rank of the sorted samples.
    """
    warmup = client.post("/predict", json=valid_input_data)
    assert warmup.status_code == 200
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed = time.perf_counter() - start
        assert response.status_code == 200
        assert math.isfinite(response.json()["prediction"])
        samples.append(elapsed)
    samples.sort()
    median = (samples[9] + samples[10]) / 2
    p95 = samples[math.ceil(0.95 * len(samples)) - 1]
    report = ROOT / "evidence" / "latency_report.txt"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "method: in-process Starlette TestClient (not localhost HTTP)",
                "requests: 1 warmup excluded, then 20 successful POST /predict",
                "timer: time.perf_counter around the client call",
                "median: average of the 10th and 11th sorted samples",
                "p95: nearest rank, index ceil(0.95 * 20) - 1",
                f"median_seconds: {median:.6f}",
                f"p95_seconds: {p95:.6f}",
                f"min_seconds: {samples[0]:.6f}",
                f"max_seconds: {samples[-1]:.6f}",
                f"platform: {platform.platform()}",
                f"machine: {platform.machine()}",
                f"cpu_count: {os.cpu_count()}",
                "budget: median < 1 second on this machine for a local guard only",
                "",
            ]
        )
    )
    assert median < 1.0
