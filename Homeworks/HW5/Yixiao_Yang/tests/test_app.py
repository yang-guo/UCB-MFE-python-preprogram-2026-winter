"""HW5 checks adapted from the supplied four test groups and Lesson 9 demo."""

import importlib
import json
import math
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import InputData, OutputData, app
from benchmark import measure_requests
from model_contract import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
NULLABLE_FEATURES = [
    "momentum_252_21",
    "drawdown_252",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
]


@pytest.fixture
def client():
    # A context manager runs the same lifespan used by the HTTP server.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    return json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))


@pytest.fixture
def reference():
    return json.loads((ROOT / "reference_prediction.json").read_text(encoding="utf-8"))


def direct_prediction(payload):
    # Build the named matrix independently of the API's conversion helper.
    frame = pd.DataFrame(
        [[payload[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
        dtype=float,
    )
    return float(app.state.bundle["pipeline"].predict(frame)[0])


def test_predict_valid_input(client, valid_input_data, reference):
    assert valid_input_data == reference["input"]
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    parsed = OutputData.model_validate(body)
    assert isinstance(parsed.prediction, float)
    assert math.isfinite(parsed.prediction)
    assert parsed.model_version == reference["model_version"]
    assert parsed.model_version == app.state.bundle["model_version"]
    assert parsed.prediction == pytest.approx(
        direct_prediction(valid_input_data), rel=1e-12, abs=1e-12
    )
    # This expected value came from HW4's frozen forecasts, before the API.
    assert parsed.prediction == pytest.approx(
        reference["prediction"],
        rel=reference["relative_tolerance"],
        abs=reference["absolute_tolerance"],
    )


@pytest.mark.parametrize("missing_field", FEATURE_COLUMNS)
def test_predict_missing_feature(client, valid_input_data, monkeypatch, missing_field):
    invalid = dict(valid_input_data)
    del invalid[missing_field]
    predict = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    assert client.post("/predict", json=invalid).status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize("field", FEATURE_COLUMNS)
@pytest.mark.parametrize("invalid_value", ["0.05", True, False])
def test_predict_strict_numeric_types(
    client, valid_input_data, monkeypatch, field, invalid_value
):
    invalid = {**valid_input_data, field: invalid_value}
    predict = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    assert client.post("/predict", json=invalid).status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected_feature": 1},
        {"momentum_21": -1.0},
        {"momentum_63": -1.1},
        {"momentum_252_21": -1.0},
        {"volatility_63": -0.01},
        {"drawdown_252": 0.01},
        {"drawdown_252": -1.0},
        {"filing_age_days": -1},
        {"filing_age_days": 1.5},
        {"filing_age_days": 5.0},
        {"filing_age_days": 2**53},
        {"filing_age_days": 10**400},
        {"momentum_21": None},
        {"momentum_63": None},
        {"volatility_63": None},
        {"net_margin": []},
        {"operating_roa": {}},
    ],
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, changes):
    predict = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    response = client.post("/predict", json={**valid_input_data, **changes})
    assert response.status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize("payload", [{}, [], [0.0] * 11, [0.0] * 12, [0.0] * 13, None])
def test_named_contract_rejects_wrong_shape(client, monkeypatch, payload):
    # The service accepts exactly twelve named fields, not an anonymous list.
    predict = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    response = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize("field", ["momentum_21", "net_margin", "filing_age_days"])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_nonfinite_json_is_422(client, valid_input_data, monkeypatch, field, token):
    # Bypass httpx's JSON encoder so the server, not the client, rejects this.
    invalid = {**valid_input_data, field: "NONFINITE_TOKEN"}
    raw = json.dumps(invalid).replace('"NONFINITE_TOKEN"', token)
    predict = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    response = client.post(
        "/predict", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    predict.assert_not_called()


def test_malformed_json_is_422(client, monkeypatch):
    predict = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    response = client.post(
        "/predict", content='{"momentum_21":', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize("field", NULLABLE_FEATURES)
def test_explicit_missing_values_use_saved_imputer(client, valid_input_data, field):
    payload = {**valid_input_data, field: None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(
        direct_prediction(payload), rel=1e-12, abs=1e-12
    )


def test_valid_boundaries_and_unbounded_ratios(client, valid_input_data):
    payload = {
        **valid_input_data,
        "volatility_63": 0,
        "drawdown_252": 0,
        "net_margin": -2.0,
        "debt_to_assets": 3.0,
        "filing_age_days": 700,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(direct_prediction(payload))


def test_json_key_order_does_not_change_feature_order(client, valid_input_data):
    reversed_payload = dict(reversed(list(valid_input_data.items())))
    original = client.post("/predict", json=valid_input_data)
    reversed_response = client.post("/predict", json=reversed_payload)
    assert original.status_code == reversed_response.status_code == 200
    assert original.json() == reversed_response.json()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    predict = Mock(side_effect=RuntimeError("private filesystem path and model failure"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "private filesystem path" not in response.text
    assert "RuntimeError" not in response.text
    predict.assert_called_once()


@pytest.mark.parametrize("prediction", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_result_is_500(
    failure_client, valid_input_data, monkeypatch, prediction
):
    predict = Mock(return_value=np.array([prediction]))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "prediction" not in response.json()
    predict.assert_called_once()


def test_artifact_loaded_once_during_lifespan(valid_input_data, monkeypatch):
    app_module = importlib.import_module("app")
    real_load = app_module.joblib.load
    load = Mock(wraps=real_load)
    monkeypatch.setattr(app_module.joblib, "load", load)
    with TestClient(app) as test_client:
        assert load.call_count == 1
        for _ in range(3):
            assert test_client.post("/predict", json=valid_input_data).status_code == 200
        assert load.call_count == 1


def test_requests_do_not_refit_preprocessing(client, valid_input_data, monkeypatch):
    pipeline = app.state.bundle["pipeline"]
    fit_calls = []
    for estimator in [pipeline, *pipeline.named_steps.values()]:
        for method_name in ("fit", "fit_transform"):
            if hasattr(estimator, method_name):
                call = Mock(side_effect=AssertionError("A request tried to fit the model"))
                monkeypatch.setattr(estimator, method_name, call)
                fit_calls.append(call)
    for _ in range(2):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    for call in fit_calls:
        call.assert_not_called()


def test_saved_preprocessing_is_independent_of_batch(client, valid_input_data):
    pipeline = app.state.bundle["pipeline"]
    first = pd.DataFrame([valid_input_data], columns=FEATURE_COLUMNS, dtype=float)
    second_payload = {**valid_input_data, "momentum_21": 5.0, "net_margin": None}
    batch = pd.DataFrame(
        [valid_input_data, second_payload], columns=FEATURE_COLUMNS, dtype=float
    )
    assert pipeline.predict(first)[0] == pytest.approx(pipeline.predict(batch)[0], abs=1e-12)


def test_health_reports_loaded_model(client, reference):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": reference["model_version"]}


def test_contract_lists_all_required_fields():
    assert list(InputData.model_fields) == FEATURE_COLUMNS
    assert all(field.is_required() for field in InputData.model_fields.values())


def test_predict_performance(client, valid_input_data):
    report = measure_requests(client, valid_input_data, warmups=5, requests=50)
    assert report["successful_requests"] == 50
    assert report["median_ms"] > 0
    assert report["p95_ms"] >= report["median_ms"]
    print("\nIN_PROCESS_TESTCLIENT_LATENCY " + json.dumps(report, sort_keys=True))
