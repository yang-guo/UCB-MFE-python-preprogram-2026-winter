"""Checks for the served HW4 pipeline.

The reference prediction was recorded before this API existed. After fitting
``hw4-hgb-deep-static-2016-04-01``, a direct ``pipeline.predict`` on symbol A
for 2016-04-01 returned -0.09155196684529192. That input is
``example_request.json`` and the number is also stored in
``artifacts/model_metadata.json``. The model is a deterministic tree, so the
test tolerance is only float round-trip slack.
"""

import json
import math
import os
import platform
import statistics
import subprocess
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import OutputData, app
from scoring import EXAMPLE_PATH, FEATURE_COLUMNS, MODEL_VERSION

# Direct pipeline.predict result saved at serialization, before the service.
REFERENCE_PREDICTION = -0.09155196684529192
# Absolute slack for JSON float round-trip. An ulp near 0.09 is about 1e-17.
PREDICTION_TOLERANCE = 1e-12
# In-process TestClient. One 22-feature histogram-boost predict is a few
# milliseconds here; 250ms fails a hang or an accidental refit without
# treating a universal 500ms rule as the requirement.
P95_BUDGET_SECONDS = 0.25
MEASURED_REQUESTS = 30


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
    return json.loads(EXAMPLE_PATH.read_text())


def test_predict_valid_input(client, valid_input_data):
    frame = pd.DataFrame([valid_input_data], columns=FEATURE_COLUMNS)
    direct = float(app.state.bundle["pipeline"].predict(frame)[0])

    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    parsed = OutputData.model_validate(body)
    assert set(body) == {"prediction", "model_version"}
    assert parsed.model_version == MODEL_VERSION
    assert math.isfinite(parsed.prediction)
    assert parsed.prediction == pytest.approx(direct, abs=PREDICTION_TOLERANCE)
    assert parsed.prediction == pytest.approx(REFERENCE_PREDICTION, abs=PREDICTION_TOLERANCE)


@pytest.mark.parametrize(
    "case",
    [
        "missing_field",
        "extra_field",
        "numeric_string",
        "boolean",
        "out_of_range",
        "unknown_sector",
        "null_feature",
        "list_width_21",
        "list_width_22",
        "list_width_23",
        "nan",
        "positive_infinity",
        "negative_infinity",
    ],
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, case):
    def unexpected_call(*args, **kwargs):
        pytest.fail("invalid input reached inference")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)
    payload = dict(valid_input_data)
    raw = None
    if case == "missing_field":
        del payload["momentum_5"]
    elif case == "extra_field":
        payload["symbol"] = "A"
    elif case == "numeric_string":
        payload["volatility_21"] = "0.20"
    elif case == "boolean":
        payload["large_cap_score"] = True
    elif case == "out_of_range":
        payload["volatility_21"] = -0.01
    elif case == "unknown_sector":
        payload["sector"] = "Unknown"
    elif case == "null_feature":
        payload["filing_age_days"] = None
    elif case == "list_width_21":
        payload = [0.1] * 21
    elif case == "list_width_22":
        payload = [0.1] * 22
    elif case == "list_width_23":
        payload = [0.1] * 23
    elif case in {"nan", "positive_infinity", "negative_infinity"}:
        token = {
            "nan": "NaN",
            "positive_infinity": "Infinity",
            "negative_infinity": "-Infinity",
        }
        encoded = json.dumps(valid_input_data)
        needle = '"momentum_5": ' + json.dumps(valid_input_data["momentum_5"])
        raw = encoded.replace(needle, f'"momentum_5": {token[case]}', 1)
        assert raw != encoded
    else:
        raise AssertionError(f"unhandled case {case}")

    if raw is None:
        response = client.post("/predict", json=payload)
    else:
        response = client.post(
            "/predict", content=raw, headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 422


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "simulated inference failure" not in response.text
    assert response.json() == {"detail": "Prediction unavailable"}


def _hardware() -> str:
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    memory_gb = pages * page_size / 1024**3
    return f"{cpu}, {memory_gb:.1f} GB RAM, {platform.platform()}"


def test_predict_performance(client, valid_input_data):
    warmup = client.post("/predict", json=valid_input_data)
    assert warmup.status_code == 200

    samples = []
    for _ in range(MEASURED_REQUESTS):
        started = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        samples.append(time.perf_counter() - started)
        assert response.status_code == 200
        assert response.json()["prediction"] == pytest.approx(
            REFERENCE_PREDICTION, abs=PREDICTION_TOLERANCE
        )

    samples.sort()
    median = statistics.median(samples)
    # Nearest rank: the smallest sample at or above the 95th percentile.
    p95 = samples[math.ceil(0.95 * len(samples)) - 1]
    print(
        "\nlatency in-process TestClient, not localhost HTTP; "
        f"hardware {_hardware()}; warmup 1; n={len(samples)}; "
        f"median_ms={median * 1e3:.3f}; p95_ms={p95 * 1e3:.3f}"
    )
    assert p95 < P95_BUDGET_SECONDS
