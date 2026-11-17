"""API tests for the HW5 service, using the real saved HW4 model."""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import FEATURE_NAMES, to_frame

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "artifacts" / "reference_prediction.json").read_text())
VALID_INPUT: dict[str, float] = REFERENCE["features"]

# Same library versions and the same float64 input, so the result should be
# identical up to floating-point summation order.
TOLERANCE = {"rel": 1e-9, "abs": 1e-12}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input() -> dict[str, float]:
    return dict(VALID_INPUT)


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Wrap the real predict so tests can see whether inference ran."""
    model = client.app.state.model
    real_predict = model.predict
    calls = []

    def spy(X):
        calls.append(X)
        return real_predict(X)

    monkeypatch.setattr(model, "predict", spy)
    return calls


# 1. Valid prediction --------------------------------------------------------


def test_valid_prediction_matches_pipeline_and_hw4_reference(client, valid_input):
    response = client.post("/predict", json=valid_input)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float)
    assert body["model_version"] == client.app.state.model_version

    direct = float(client.app.state.model.predict(to_frame(valid_input))[0])
    assert body["prediction"] == pytest.approx(direct, **TOLERANCE)
    # Recorded in HW4.ipynb right after fitting final_relative_model.
    assert body["prediction"] == pytest.approx(
        REFERENCE["expected_prediction"], **TOLERANCE
    )


def test_null_feature_is_imputed_like_training(client, valid_input):
    valid_input["cash_flow_to_market"] = None
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200

    direct = float(client.app.state.model.predict(to_frame(valid_input))[0])
    assert response.json()["prediction"] == pytest.approx(direct, **TOLERANCE)


def test_feature_order_matches_fitted_pipeline(client):
    assert tuple(client.app.state.model.feature_names_in_) == FEATURE_NAMES


# 2. Input validation --------------------------------------------------------


def _without(data: dict, key: str) -> dict:
    return {k: v for k, v in data.items() if k != key}


INVALID_CASES = {
    "missing_field": lambda d: _without(d, "momentum_1m"),
    "too_few_features": lambda d: dict(list(d.items())[:17]),
    "too_many_features": lambda d: {**d, "extra_feature": 0.1},
    "numeric_string": lambda d: {**d, "roe": "0.12"},
    "boolean": lambda d: {**d, "roa": True},
    "text": lambda d: {**d, "net_margin": "abc"},
    "list_value": lambda d: {**d, "roe": [0.1]},
    "below_range": lambda d: {**d, "momentum_3m": -1.5},
    "above_range": lambda d: {**d, "volatility_21d": 51.0},
    "negative_volatility": lambda d: {**d, "volatility_63d": -0.1},
}


@pytest.mark.parametrize("case", INVALID_CASES)
def test_invalid_input_returns_422_without_inference(
    client, predict_spy, valid_input, case
):
    response = client.post("/predict", json=INVALID_CASES[case](valid_input))
    assert response.status_code == 422
    assert predict_spy == []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_return_422(client, predict_spy, valid_input, token):
    body = json.dumps(valid_input).replace(
        f'"roe": {json.dumps(valid_input["roe"])}', f'"roe": {token}'
    )
    assert token in body
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert predict_spy == []


@pytest.mark.parametrize("payload", [[], "not an object", None])
def test_wrong_body_shape_returns_422(client, predict_spy, payload):
    assert client.post("/predict", json=payload).status_code == 422
    assert predict_spy == []


def test_spy_records_valid_requests(client, predict_spy, valid_input):
    """Guards the validation tests: the spy does see real inference calls."""
    assert client.post("/predict", json=valid_input).status_code == 200
    assert len(predict_spy) == 1


# 3. Inference failure -------------------------------------------------------


def test_inference_failure_returns_500(failure_client, monkeypatch, valid_input):
    def boom(X):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(failure_client.app.state.model, "predict", boom)
    response = failure_client.post("/predict", json=valid_input)

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_non_finite_model_output_returns_500(failure_client, monkeypatch, valid_input):
    monkeypatch.setattr(
        failure_client.app.state.model, "predict", lambda X: [float("nan")]
    )
    assert failure_client.post("/predict", json=valid_input).status_code == 500


# 4. Performance -------------------------------------------------------------


def test_latency_report(client, valid_input):
    """In-process TestClient latency (no network, no uvicorn).

    Budget: 100 ms median. One Ridge predict on 18 features takes well under
    1 ms; the rest is DataFrame construction, validation, and TestClient
    overhead. 100 ms leaves a wide margin on a laptop while still catching
    an accidental refit (several seconds on 140k rows) inside the request.
    """
    for _ in range(5):
        assert client.post("/predict", json=valid_input).status_code == 200

    timings_ms = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(timings_ms)
    p95 = statistics.quantiles(timings_ms, n=20)[-1]
    print(
        f"\nLatency over {len(timings_ms)} requests (in-process TestClient, "
        f"{platform.processor() or platform.machine()}, "
        f"Python {platform.python_version()}): "
        f"median={median:.2f} ms, p95={p95:.2f} ms"
    )
    assert median < 100
