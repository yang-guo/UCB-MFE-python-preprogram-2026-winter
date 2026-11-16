"""Tests for the HW5 service around the real HW4 artifact.

The reference predictions in ``hw4_reference.json`` were produced by HW4's own code
(``RidgeOnRanks.predict`` on ``model.cross_section`` rows for 2026-09-28) inside
``export_hw4_model.py``, before any API wrapping. They are independent of the
serialized sklearn pipeline the service loads.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import InputData, app
from ranks_model import FEATURES

REFERENCE = json.loads((Path(__file__).parent / "hw4_reference.json").read_text())
# Requests carry float64 values that HW4 held as float32 ranks; the only difference
# is rounding in ``rank - 0.5``. Predictions are ranks in (0, 1), so 1e-6 is far
# below any economically meaningful change and far above float noise (~1e-8).
ATOL = 1e-6


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
    """AAPL's ranked features on 2026-09-28, as scored by HW4 (no missing values)."""
    return dict(REFERENCE["cases"][0]["features"])


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Wrap the loaded pipeline's predict to count calls while still using it."""
    pipeline = client.app.state.bundle["pipeline"]
    calls = []
    real_predict = pipeline.predict

    def spy(X, *args, **kwargs):
        calls.append(X)
        return real_predict(X, *args, **kwargs)

    monkeypatch.setattr(pipeline, "predict", spy)
    return calls


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["symbol"])
def test_predict_valid_input(client, case):
    response = client.post("/predict", json=case["features"])

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and np.isfinite(body["prediction"])
    assert body["model_version"] == REFERENCE["model_version"]

    # 1. Agrees with a direct call to the loaded pipeline on the same representation.
    direct = client.app.state.bundle["pipeline"].predict(
        InputData(**case["features"]).to_frame()
    )[0]
    assert body["prediction"] == pytest.approx(direct, abs=1e-12)
    # 2. Agrees with the prediction recorded from HW4's code before wrapping.
    assert body["prediction"] == pytest.approx(case["prediction"], abs=ATOL)


def test_reference_includes_missing_feature_case():
    """null handling is covered by a real HW4 row, not a made-up one."""
    assert any(None in c["features"].values() for c in REFERENCE["cases"])


def _without(data, key):
    return {k: v for k, v in data.items() if k != key}


INVALID_CASES = {
    "missing_field": lambda d: _without(d, "roe"),
    "extra_field": lambda d: {**d, "skew_180": 0.5},
    "too_few_features": lambda d: {k: d[k] for k in FEATURES[:19]},
    "numeric_string": lambda d: {**d, "mom_5": "0.5"},
    "boolean": lambda d: {**d, "mom_5": True},
    "text": lambda d: {**d, "vol_63": "high"},
    "list_value": lambda d: {**d, "vol_63": [0.5]},
    "zero_rank": lambda d: {**d, "mom_21": 0.0},
    "negative_rank": lambda d: {**d, "mom_21": -0.2},
    "rank_above_one": lambda d: {**d, "mom_21": 1.5},
    "raw_unranked_value": lambda d: {**d, "filing_age_days": 45},
    "fixed_length_list": lambda d: list(d.values()),
    "empty_body": lambda d: {},
}


@pytest.mark.parametrize("make_invalid", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_predict_invalid_input(client, valid_input_data, predict_spy, make_invalid):
    response = client.post("/predict", json=make_invalid(valid_input_data))

    assert response.status_code == 422
    assert predict_spy == []  # rejected before inference


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite(client, valid_input_data, predict_spy, bad):
    # Standard JSON cannot encode NaN/inf, so send the raw tokens Python's json emits.
    body = json.dumps(valid_input_data).replace(json.dumps(valid_input_data["mom_5"]), bad, 1)
    assert bad in body
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert predict_spy == []


def test_predict_spy_sees_valid_requests(client, valid_input_data, predict_spy):
    """Control for the invalid tests: the spy does record calls that reach the model."""
    assert client.post("/predict", json=valid_input_data).status_code == 200
    assert len(predict_spy) == 1
    assert list(predict_spy[0].columns) == FEATURES


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(failure_client.app.state.bundle["pipeline"], "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "simulated inference failure" not in response.text  # details stay in logs


def test_predict_non_finite_model_output_is_500(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(
        failure_client.app.state.bundle["pipeline"], "predict", lambda X: np.array([np.nan])
    )
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    """In-process TestClient latency (no network, no uvicorn); see README for results.

    Budget: the median must stay under 50 ms. On the development laptop it is a few
    milliseconds (one 20x40 matrix product plus FastAPI/Pydantic overhead), so 50 ms
    leaves room for slower CI machines while still catching e.g. accidental
    reloading of the artifact or refitting on each request (~hundreds of ms).
    """
    for _ in range(5):  # warm-up: imports, first-call caches
        assert client.post("/predict", json=valid_input_data).status_code == 200

    timings_ms = []
    for _ in range(100):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(timings_ms)
    p95 = statistics.quantiles(timings_ms, n=20)[-1]
    print(
        f"\nTestClient in-process latency over {len(timings_ms)} requests: "
        f"median {median:.2f} ms, p95 {p95:.2f} ms "
        f"({platform.machine()}, {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()})"
    )
    assert median < 50
