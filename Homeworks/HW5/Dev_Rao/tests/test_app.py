"""Tests for the HW5 service. Run: python -m pytest -q  (add -s to see the latency report)."""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from composite import FEATURES
from model_service import MODEL_PATH

HERE = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((HERE / "reference.json").read_text())
EVIDENCE = HERE / "evidence"
# HW4 stored and scored the ranks in float32; the service uses float64 -> ~1e-8 differences
REFERENCE_TOL = 1e-7


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
    # KLAC, 2016-03-31 signal: a complete row from the HW4 final test
    return dict(REFERENCE["cases"][0]["features"])


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Wrap the loaded pipeline's predict so a test can count inference calls."""
    pipeline = client.app.state.bundle.pipeline
    real = pipeline.predict
    calls = []

    def spy(X):
        calls.append(X)
        return real(X)

    monkeypatch.setattr(pipeline, "predict", spy)
    return calls


def direct_prediction(features: dict) -> float:
    """Load the artifact independently of the app and call the pipeline directly."""
    pipeline = joblib.load(MODEL_PATH)
    row = pd.DataFrame([features], columns=FEATURES, dtype="float64")
    return float(pipeline.predict(row)[0])


# 1. valid prediction -------------------------------------------------------------
def test_predict_valid_input(client, valid_input_data, predict_spy):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == "hw4-composite-frozen-2026-11-13"
    assert -0.5 <= body["prediction"] <= 0.5
    # API == direct pipeline call on the same artifact (same float64 path -> exact)
    assert body["prediction"] == pytest.approx(direct_prediction(valid_input_data), abs=1e-12)
    # the model saw one row with the training column names and order
    assert len(predict_spy) == 1 and list(predict_spy[0].columns) == FEATURES
    assert predict_spy[0].shape == (1, 15)


@pytest.mark.parametrize(
    "case", REFERENCE["cases"], ids=lambda c: f"{c['symbol']}-{c['signal_date']}"
)
def test_predict_matches_hw4_reference(client, case):
    """Expected values are HW4's own final-test scores (outputs/final_scores.csv),
    recorded before this API existed; includes rows with missing (null) features."""
    response = client.post("/predict", json=case["features"])
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(case["hw4_score"], abs=REFERENCE_TOL)


def test_feature_order_does_not_matter_in_json(client, valid_input_data):
    shuffled = dict(reversed(list(valid_input_data.items())))
    a = client.post("/predict", json=valid_input_data).json()["prediction"]
    b = client.post("/predict", json=shuffled).json()["prediction"]
    assert a == b


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": "hw4-composite-frozen-2026-11-13",
    }


# 2. input validation -------------------------------------------------------------
def _without(d, key):
    return {k: v for k, v in d.items() if k != key}


INVALID = {
    "missing field": lambda d: _without(d, "r_roe"),
    "missing required non-null field": lambda d: _without(d, "r_sector_mom_126"),
    "extra field": lambda d: {**d, "sector": "Technology"},
    "16 features (extra rank)": lambda d: {**d, "r_mom_63": 0.5},
    "empty body": lambda d: {},
    "list instead of object": lambda d: list(d.values()),
    "numeric string": lambda d: {**d, "r_ey": "0.5"},
    "boolean": lambda d: {**d, "r_bm": True},
    "text": lambda d: {**d, "r_sp": "high"},
    "nested value": lambda d: {**d, "r_fcfy": [0.5]},
    "null where not allowed": lambda d: {**d, "r_sector_mom_126": None},
    "rank above 1": lambda d: {**d, "r_gp_a": 1.01},
    "rank equal 0": lambda d: {**d, "r_roa": 0.0},
    "negative rank": lambda d: {**d, "r_ivol_63": -0.2},
    "raw value not a rank": lambda d: {**d, "r_mom_12_1": 35.0},
}


@pytest.mark.parametrize("name", list(INVALID))
def test_predict_invalid_input(client, valid_input_data, predict_spy, name):
    response = client.post("/predict", json=INVALID[name](valid_input_data))
    assert response.status_code == 422, response.text
    assert predict_spy == []  # rejected before inference


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_predict_non_finite_rejected(client, valid_input_data, predict_spy, token):
    # json.dumps cannot emit these by default, so send the raw JSON text
    body = json.dumps(valid_input_data).replace(str(valid_input_data["r_ey"]), token)
    assert token in body
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert predict_spy == []


def test_predict_malformed_json(client, predict_spy):
    response = client.post(
        "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert predict_spy == []


def test_integer_one_is_a_valid_rank(client, valid_input_data):
    # the top-ranked stock has rank exactly 1; JSON may send it as an integer
    response = client.post("/predict", json={**valid_input_data, "r_roe": 1})
    assert response.status_code == 200


def test_null_fundamentals_are_accepted(client, valid_input_data, predict_spy):
    data = {**valid_input_data, "r_ey": None, "r_roe": None}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert predict_spy[0][["r_ey", "r_roe"]].isna().all(axis=None)


# 3. inference failure ------------------------------------------------------------
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(X):
        raise RuntimeError("simulated inference failure with secret detail")

    monkeypatch.setattr(failure_client.app.state.bundle.pipeline, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "secret detail" not in response.text  # details stay in the server log
    assert response.headers.get("X-Request-ID")
    assert any("inference_failed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.9])
def test_invalid_model_output_is_500(failure_client, valid_input_data, monkeypatch, bad):
    # a model that returns NaN/inf or an impossible score must not produce a 200
    pipeline = failure_client.app.state.bundle.pipeline
    monkeypatch.setattr(pipeline, "predict", lambda X: [bad])
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


# 4. performance ------------------------------------------------------------------
def test_predict_performance(client, valid_input_data):
    """In-process TestClient latency (no network, no uvicorn): full FastAPI request path
    incl. validation, DataFrame build and pipeline.predict. Real localhost HTTP against
    the container is measured separately by measure_http_latency.py (see README)."""
    for _ in range(10):  # warm-up
        assert client.post("/predict", json=valid_input_data).status_code == 200
    times_ms = []
    for _ in range(200):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        times_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    times_ms.sort()
    report = {
        "method": "in-process fastapi.testclient.TestClient, sequential, 10 warm-up + 200 timed",
        "n": len(times_ms),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(times_ms[int(0.95 * len(times_ms)) - 1], 3),
        "max_ms": round(times_ms[-1], 3),
        "machine": f"{platform.processor()} | {platform.system()} {platform.release()} | "
        f"Python {platform.python_version()}",
    }
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / "latency_testclient.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nlatency report:", json.dumps(report, indent=2))
    # Budget: a single pandas row + 15 column means should take a few ms. 100 ms p95 is
    # ~20x what I measured on this laptop; it only exists to catch a pathological
    # regression (e.g. reloading or refitting the model per request), not to benchmark.
    assert report["p95_ms"] < 100
