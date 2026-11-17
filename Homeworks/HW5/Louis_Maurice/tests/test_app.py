"""Tests for the HW5 prediction service, using the real saved HW4 model."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from model import FEATURES, to_frame

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
REFERENCE = json.loads((ARTIFACTS / "reference_example.json").read_text())


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
    # AAPL on 2026-06-30, a real HW4 signal row (see artifacts/reference_example.json).
    return dict(REFERENCE["features"])


@pytest.fixture
def block_model(client, monkeypatch):
    """Make any call to the loaded pipeline fail the test."""

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == "hw4-ridge-refit-2026-06-30"
    assert np.isfinite(body["prediction"])

    # Same result as calling the saved pipeline directly with the training columns.
    direct = app.state.bundle["pipeline"].predict(to_frame(valid_input_data))[0]
    assert body["prediction"] == pytest.approx(direct, rel=1e-12)

    # Same result HW4 code produced for this row before it was wrapped in an API
    # (export_model.py scored the whole 2026-06-30 universe in one batch). The pipeline
    # is row-independent, so only floating-point noise is allowed.
    assert body["prediction"] == pytest.approx(REFERENCE["expected_prediction"], rel=1e-9)


def test_missing_financials_are_imputed(client, valid_input_data):
    # null is allowed for features that were sometimes missing in training.
    valid_input_data["net_margin"] = None
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert np.isfinite(response.json()["prediction"])


def _drop(name):
    return lambda d: {k: v for k, v in d.items() if k != name}


def _set(name, value):
    return lambda d: {**d, name: value}


INVALID_CASES = {
    "missing_field": _drop("momentum_21"),
    "missing_sector": _drop("sector"),
    "extra_field": _set("market_cap", 1.0),
    "numeric_string": _set("momentum_21", "0.05"),
    "boolean": _set("volatility_63", True),
    "null_required": _set("momentum_63", None),
    "return_below_minus_100pct": _set("momentum_21", -1.5),
    "negative_volatility": _set("volatility_63", -0.1),
    "positive_drawdown": _set("drawdown_252", 0.2),
    "filing_age_too_large": _set("filing_age_days", 5000),
    "unknown_sector": _set("sector", "Crypto"),
    "list_instead_of_object": lambda d: list(d.values()),
    "too_few_features": lambda d: dict(list(d.items())[:12]),
}


@pytest.mark.parametrize("make_payload", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_predict_invalid_input(client, block_model, valid_input_data, make_payload):
    response = client.post("/predict", json=make_payload(valid_input_data))
    assert response.status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_rejected(client, block_model, valid_input_data, value):
    # Standard JSON has no NaN/Infinity, but Python's json module writes and accepts
    # the NaN / Infinity tokens, so the server must reject them explicitly.
    raw = json.dumps({**valid_input_data, "net_margin": value})
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
    assert "simulated" not in response.text  # details stay in server logs


def test_predict_performance(client, valid_input_data):
    """In-process TestClient latency (no network), measured with time.perf_counter."""
    for _ in range(5):  # warm-up
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = float(np.percentile(latencies_ms, 95))
    print(
        f"\nTestClient latency over {len(latencies_ms)} requests on "
        f"{platform.machine()} {platform.system()}: median {median:.2f} ms, p95 {p95:.2f} ms"
    )
    # Generous budget for a laptop: one-row Ridge inference takes a few milliseconds,
    # so 250 ms p95 only catches a real regression (e.g. reloading or refitting per request).
    assert p95 < 250


def test_api_contract_matches_training_features(client):
    # The request schema, the artifact, and the pipeline all use the HW4 column order.
    from app import InputData

    assert list(InputData.model_fields) == FEATURES
    assert app.state.bundle["features"] == FEATURES
    assert list(app.state.bundle["pipeline"].feature_names_in_) == FEATURES
