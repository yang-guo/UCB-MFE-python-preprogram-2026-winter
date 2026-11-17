"""Tests for the HW5 service. They use the real saved HW4 model, except the failure test."""

import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

HERE = Path(__file__).resolve().parents[1]
JSON_HEADERS = {"Content-Type": "application/json"}


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
    # AAPL on 2016-03-31, the first signal date of the HW4 test period.
    return json.loads((HERE / "example_request.json").read_text())


@pytest.fixture
def reference():
    # Input and prediction that export_model.py recorded from the HW4 code path
    # (course_data.make_pipeline, the call in hw4.ipynb) before the API existed.
    return json.loads((HERE / "artifacts" / "model_metadata.json").read_text())["reference"]


@pytest.fixture
def predict_calls(client, monkeypatch):
    """Replace predict on the loaded model with a recorder. It must stay empty."""
    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        return [0.0]

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", record)
    return calls


def changed(payload, **changes):
    return {**payload, **changes}


def test_predict_valid_input(client, valid_input_data, reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()

    # Schema: exactly two fields, a finite float and the version text.
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert body["model_version"] == "hw4-round1-ridge-fit-2016-03-31"

    # The API result equals a direct call of the saved pipeline.
    bundle = app.state.bundle
    frame = pd.DataFrame([valid_input_data], columns=bundle["features"])
    direct = float(bundle["pipeline"].predict(frame)[0])
    assert body["prediction"] == pytest.approx(direct, rel=1e-12, abs=1e-12)

    # The API result equals the prediction recorded from the HW4 code. The tolerance is
    # 1e-9 because the record can come from a different CPU or BLAS library (the last
    # digits of a float can change), and a real model error is much larger than 1e-9.
    assert valid_input_data == reference["input"]
    assert body["prediction"] == pytest.approx(reference["prediction"], rel=1e-9, abs=1e-9)


def test_integer_json_number_is_accepted(client, valid_input_data):
    # 63 and 63.0 are the same JSON number for a day count. Only strings/booleans are refused.
    response = client.post("/predict", json=changed(valid_input_data, filing_age_days=63))
    assert response.status_code == 200


INVALID_CASES = {
    "missing field (11 features)": lambda p: {k: v for k, v in p.items() if k != "net_margin"},
    "extra field (13 features)": lambda p: changed(p, volatility_63=0.3),
    "empty object": lambda p: {},
    "feature list, not named fields": lambda p: list(p.values()),
    "numeric string": lambda p: changed(p, momentum_21="0.08"),
    "boolean": lambda p: changed(p, momentum_21=True),
    "null": lambda p: changed(p, net_margin=None),
    "nested value": lambda p: changed(p, momentum_21=[0.08]),
    "below the screen limit": lambda p: changed(p, momentum_63=-0.9),
    "above the screen limit": lambda p: changed(p, momentum_63=5.5),
    "return below -100%": lambda p: changed(p, momentum_21=-1.5),
    "percent, not fraction": lambda p: changed(p, momentum_252_21=1500.0),
    "negative debt": lambda p: changed(p, debt_to_assets=-0.1),
    "negative filing age": lambda p: changed(p, filing_age_days=-1.0),
    "ratio too large": lambda p: changed(p, operating_roa=1e9),
}


@pytest.mark.parametrize("case", INVALID_CASES)
def test_predict_invalid_input(client, valid_input_data, predict_calls, case):
    response = client.post("/predict", json=INVALID_CASES[case](valid_input_data))
    assert response.status_code == 422
    assert predict_calls == [], "Invalid input reached the model"


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_predict_non_finite_input(client, valid_input_data, predict_calls, token):
    # Standard JSON has no NaN or infinity, so the body is sent as raw text.
    body = json.dumps(valid_input_data).replace(
        json.dumps(valid_input_data["momentum_21"]), token, 1
    )
    assert token in body
    response = client.post("/predict", content=body, headers=JSON_HEADERS)
    assert response.status_code == 422
    assert predict_calls == [], "Non-finite input reached the model"


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    # No default prediction, and no internal error text in the response.
    assert "prediction" not in response.text.replace("Prediction unavailable", "")
    assert "simulated" not in response.text


def test_non_finite_prediction_is_a_failure(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", lambda *a, **k: [float("nan")])
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_health_reports_the_loaded_version(client):
    assert client.get("/health").json() == {
        "status": "ready",
        "model_version": "hw4-round1-ridge-fit-2016-03-31",
    }


def hardware():
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        found = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        chip = found.stdout.strip() or chip
    return f"{chip}, {os.cpu_count()} logical CPUs, {platform.platform()}"


def test_predict_performance(client, valid_input_data):
    for _ in range(10):  # Warm-up: the first calls import code paths and fill caches.
        assert client.post("/predict", json=valid_input_data).status_code == 200

    samples_ms = []
    for _ in range(200):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        samples_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(samples_ms)
    p95 = statistics.quantiles(samples_ms, n=20)[18]
    report = "\n".join(
        [
            "Latency of POST /predict",
            "Method: in-process Starlette TestClient (no network, no real HTTP socket),",
            "  10 warm-up requests, then 200 successful requests in sequence,",
            "  each timed with time.perf_counter around client.post.",
            f"Hardware: {hardware()}",
            f"Python {platform.python_version()}",
            f"median {median:.2f} ms | p95 {p95:.2f} ms | max {max(samples_ms):.2f} ms",
            "",
        ]
    )
    (HERE / "latency_report.txt").write_text(report)
    print("\n" + report)

    # Budget for my laptop only. One Ridge prediction on one row takes a few milliseconds.
    # 250 ms is far above that. It fails only for a real regression, for example if the
    # service loaded the model again for each request.
    assert p95 < 250
