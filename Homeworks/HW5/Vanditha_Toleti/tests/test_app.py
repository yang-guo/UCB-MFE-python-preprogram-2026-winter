"""Tests for the HW5 stock-return prediction API."""

import json
import math
import statistics
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURE_COLUMNS, MODEL_VERSION, app

EXPECTED_PREDICTION = 0.036766915694316164


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
    return {
        "momentum_21": -0.09344481605351174,
        "momentum_63": -0.018112004636673174,
        "momentum_252_21": 0.11633811230585445,
        "volatility_63": 0.22113734238133143,
        "drawdown_252": -0.13438078814587717,
        "net_margin": 0.23320795271359485,
        "operating_roa": 0.03480788874047301,
        "debt_to_assets": 0.2635342185903984,
        "cash_to_assets": 0.14056729787066866,
        "fcf_to_assets": 0.035515046751001804,
        "accruals_to_assets": -0.008721615463188497,
        "filing_age_days": 8.0,
    }


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200

    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert math.isfinite(result["prediction"])
    assert result["model_version"] == MODEL_VERSION

    # Compare the API result with a direct call to the loaded HW4 pipeline.
    features = pd.DataFrame(
        [valid_input_data],
        columns=FEATURE_COLUMNS,
    )
    direct_prediction = float(client.app.state.model.predict(features)[0])

    assert result["prediction"] == pytest.approx(direct_prediction, abs=1e-12)

    # Reference prediction recorded using the saved HW4 artifact.
    assert result["prediction"] == pytest.approx(EXPECTED_PREDICTION, abs=1e-12)


def test_predict_invalid_input(client, valid_input_data, monkeypatch):
    model = client.app.state.model
    predict_called = False

    def unexpected_predict(*args, **kwargs):
        nonlocal predict_called
        predict_called = True
        raise AssertionError("Invalid input reached model inference")

    monkeypatch.setattr(model, "predict", unexpected_predict)

    invalid_cases = []

    # Missing required feature.
    missing = valid_input_data.copy()
    missing.pop("momentum_21")
    invalid_cases.append(missing)

    # Unexpected feature.
    extra = valid_input_data.copy()
    extra["unexpected_feature"] = 123
    invalid_cases.append(extra)

    # Numeric string must not be silently converted.
    wrong_type = valid_input_data.copy()
    wrong_type["momentum_21"] = "-0.0934"
    invalid_cases.append(wrong_type)

    # Boolean must not be accepted as a number.
    boolean_value = valid_input_data.copy()
    boolean_value["momentum_21"] = True
    invalid_cases.append(boolean_value)

    # Non-finite values must be rejected.
    nan_value = valid_input_data.copy()
    nan_value["net_margin"] = float("nan")
    invalid_cases.append(nan_value)

    infinity_value = valid_input_data.copy()
    infinity_value["net_margin"] = float("inf")
    invalid_cases.append(infinity_value)

    # Feature-specific constraints.
    negative_volatility = valid_input_data.copy()
    negative_volatility["volatility_63"] = -0.1
    invalid_cases.append(negative_volatility)

    positive_drawdown = valid_input_data.copy()
    positive_drawdown["drawdown_252"] = 0.1
    invalid_cases.append(positive_drawdown)

    negative_age = valid_input_data.copy()
    negative_age["filing_age_days"] = -1
    invalid_cases.append(negative_age)

    # Momentum cannot represent a loss greater than 100%.
    extreme_momentum = valid_input_data.copy()
    extreme_momentum["momentum_21"] = -5
    invalid_cases.append(extreme_momentum)

    # Drawdown cannot be less than -100%.
    extreme_drawdown = valid_input_data.copy()
    extreme_drawdown["drawdown_252"] = -2
    invalid_cases.append(extreme_drawdown)

    # Debt-to-assets cannot be negative.
    negative_debt = valid_input_data.copy()
    negative_debt["debt_to_assets"] = -3
    invalid_cases.append(negative_debt)

    # Cash-to-assets cannot exceed total assets.
    excessive_cash = valid_input_data.copy()
    excessive_cash["cash_to_assets"] = 1.5
    invalid_cases.append(excessive_cash)

    # Financial data older than 550 days is considered stale in HW4.
    stale_filing = valid_input_data.copy()
    stale_filing["filing_age_days"] = 1e9
    invalid_cases.append(stale_filing)

    for payload in invalid_cases:
        response = client.post(
            "/predict",
            content=json.dumps(payload, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, response.text

    assert predict_called is False


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    model = failure_client.app.state.model

    def failing_predict(*args, **kwargs):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(model, "predict", failing_predict)

    response = failure_client.post(
        "/predict",
        json=valid_input_data,
    )

    assert response.status_code == 500
    assert "Simulated model failure" not in response.text


def test_predict_performance(client, valid_input_data):
    # Warm-up request; excluded from latency measurements.
    warmup = client.post("/predict", json=valid_input_data)
    assert warmup.status_code == 200

    latencies = []

    for _ in range(25):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        latencies.append(elapsed * 1000)

    sorted_latencies = sorted(latencies)
    p95_index = math.ceil(0.95 * len(sorted_latencies)) - 1

    median_latency = statistics.median(latencies)
    p95_latency = sorted_latencies[p95_index]

    print("\nPerformance results (in-process TestClient):")
    print(f"Requests measured: {len(latencies)}")
    print(f"Median latency: {median_latency:.3f} ms")
    print(f"P95 latency: {p95_latency:.3f} ms")
