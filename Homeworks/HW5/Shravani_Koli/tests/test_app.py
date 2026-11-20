"""Tests for the HW5 stock-return prediction API."""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURES, app

ROOT = Path(__file__).resolve().parents[1]


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
    return json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200

    result = response.json()

    assert set(result) == {"prediction", "model_version"}
    assert isinstance(result["prediction"], (int, float))
    assert math.isfinite(result["prediction"])
    assert result["model_version"] == "hw4-ridge-alpha100-v1"

    artifact = app.state.artifact

    frame = pd.DataFrame([valid_input_data], columns=FEATURES)

    clipped = frame.clip(
        lower=artifact["feature_lower"],
        upper=artifact["feature_upper"],
        axis=1,
    )

    direct_prediction = float(artifact["model"].predict(clipped)[0])

    assert result["prediction"] == pytest.approx(
        direct_prediction,
        abs=1e-12,
    )

    reference = json.loads((ROOT / "reference_prediction.json").read_text(encoding="utf-8"))

    assert valid_input_data == reference["input"]

    assert result["prediction"] == pytest.approx(
        reference["expected_prediction"],
        abs=1e-10,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"return_20d": None},
        {"return_20d": "0.1"},
        {"return_20d": True},
        {"return_20d": -1.1},
        {"return_60d": 10.1},
        {"volatility_20d": -0.1},
        {"volatility_20d": "NaN"},
        {"extra_field": 1.0},
    ],
)
def test_predict_invalid_input(
    client,
    valid_input_data,
    changes,
    monkeypatch,
):
    def should_not_predict(_):
        raise AssertionError("Invalid input reached inference")

    monkeypatch.setattr(
        app.state.artifact["model"],
        "predict",
        should_not_predict,
    )

    payload = {**valid_input_data, **changes}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_feature(client, valid_input_data, monkeypatch):
    def should_not_predict(_):
        raise AssertionError("Missing feature reached inference")

    monkeypatch.setattr(
        app.state.artifact["model"],
        "predict",
        should_not_predict,
    )

    valid_input_data.pop("return_20d")

    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 422


def test_wrong_feature_count(client):
    response = client.post(
        "/predict",
        json={"features": [0.1, 0.2]},
    )

    assert response.status_code == 422


def test_nonfinite_input(client, valid_input_data):
    for value in ["NaN", "Infinity", "-Infinity"]:
        payload = {
            **valid_input_data,
            "return_20d": value,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422


def test_predict_model_failure(
    failure_client,
    valid_input_data,
    monkeypatch,
):
    def broken_predict(_):
        raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(
        app.state.artifact["model"],
        "predict",
        broken_predict,
    )

    response = failure_client.post(
        "/predict",
        json=valid_input_data,
    )

    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        response = client.post(
            "/predict",
            json=valid_input_data,
        )
        assert response.status_code == 200

    timings = []

    for _ in range(30):
        start = time.perf_counter()

        response = client.post(
            "/predict",
            json=valid_input_data,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200

        timings.append(elapsed_ms)

    timings.sort()

    median_ms = statistics.median(timings)

    p95_index = math.ceil(0.95 * len(timings)) - 1
    p95_ms = timings[p95_index]

    report = (
        f"Hardware: {platform.platform()}, "
        f"{platform.machine()}\n"
        "Method: in-process FastAPI TestClient\n"
        "Warmup requests: 5\n"
        "Measured requests: 30\n"
        f"Median latency: {median_ms:.3f} ms\n"
        f"P95 latency: {p95_ms:.3f} ms\n"
        "No universal latency cutoff asserted.\n"
    )

    (ROOT / "latency_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + report)
