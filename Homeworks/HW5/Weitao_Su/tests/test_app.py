"""Check the real HW4 model, validation, failures, startup, and request latency."""

import json
import math
import platform
import time
from importlib.metadata import version
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from model_utils import FEATURES, ROOT


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
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture(scope="session")
def reference():
    return json.loads((ROOT / "artifacts/reference_prediction.json").read_text())


# 1. Real-model predictions and the independently recorded HW4 reference.
@pytest.mark.parametrize("reverse_keys", [False, True])
def test_predict_valid_input(client, valid_input_data, reference, reverse_keys):
    payload = (
        dict(reversed(list(valid_input_data.items()))) if reverse_keys else valid_input_data
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert type(result["prediction"]) in (float, int)
    assert math.isfinite(result["prediction"])
    assert result["model_version"] == reference["model_version"]

    # Load the saved pipeline separately and build its DataFrame without API helpers.
    bundle = joblib.load(ROOT / "artifacts/model.joblib")
    row = pd.DataFrame([[valid_input_data[name] for name in FEATURES]], columns=FEATURES)
    direct = float(bundle["pipeline"].predict(row)[0])
    assert result["prediction"] == pytest.approx(direct, rel=1e-8, abs=1e-10)
    assert valid_input_data == reference["input"]
    assert result["prediction"] == pytest.approx(reference["prediction"], rel=1e-8, abs=1e-10)
    for package, expected in bundle["training_versions"].items():
        # DuckDB is only an export dependency; it is deliberately absent in Docker.
        if package != "duckdb":
            assert version(package) == expected


def assert_rejected(client, monkeypatch, payload, raw=False):
    predict = Mock(side_effect=AssertionError("Invalid input reached the model"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", predict)
    if raw:
        response = client.post(
            "/predict", content=payload, headers={"Content-Type": "application/json"}
        )
    else:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]
    predict.assert_not_called()


# 2. Every required field, numeric type, finite value, range, and feature count.
@pytest.mark.parametrize("field", FEATURES)
def test_predict_missing_field(client, valid_input_data, monkeypatch, field):
    del valid_input_data[field]
    assert_rejected(client, monkeypatch, valid_input_data)


@pytest.mark.parametrize("field", FEATURES)
@pytest.mark.parametrize("value", ["0.1", True, False, None, [0.1], {"value": 0.1}])
def test_predict_invalid_type(client, valid_input_data, monkeypatch, field, value):
    valid_input_data[field] = value
    assert_rejected(client, monkeypatch, valid_input_data)


@pytest.mark.parametrize("field", FEATURES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_predict_nonfinite_input(client, valid_input_data, monkeypatch, field, value):
    valid_input_data[field] = value
    # httpx's json= correctly refuses NaN/Infinity, so send the raw malformed numbers.
    assert_rejected(client, monkeypatch, json.dumps(valid_input_data), raw=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("momentum_21", -1),
        ("momentum_63", -1.01),
        ("momentum_252_21", -1),
        ("volatility_63", -0.001),
        ("volatility_63", 0.8001),
        ("drawdown_252", -1),
        ("drawdown_252", 0.001),
    ],
)
def test_predict_out_of_range(client, valid_input_data, monkeypatch, field, value):
    valid_input_data[field] = value
    assert_rejected(client, monkeypatch, valid_input_data)


@pytest.mark.parametrize(
    "payload",
    [{}, [], [0.1] * 6, [0.1] * 7, [0.1] * 8, {"features": [0.1] * 7}],
)
def test_predict_wrong_shape(client, monkeypatch, payload):
    assert_rejected(client, monkeypatch, payload)


def test_predict_extra_field(client, valid_input_data, monkeypatch):
    valid_input_data["unknown_feature"] = 0.2
    assert_rejected(client, monkeypatch, valid_input_data)


@pytest.mark.parametrize("payload", ['{"momentum_21":', "null"])
def test_predict_invalid_json_body(client, monkeypatch, payload):
    assert_rejected(client, monkeypatch, payload, raw=True)


@pytest.mark.parametrize(
    "updates", [{"volatility_63": 0, "drawdown_252": 0}, {"volatility_63": 0.8}]
)
def test_predict_valid_boundaries(client, valid_input_data, updates):
    valid_input_data.update(updates)
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


# 3. Failures remain HTTP 500, and details stay in the server logs.
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def fail(row):
        raise RuntimeError("Private model error for this test")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "Private model error" not in response.text
    assert "Private model error" in caplog.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_predict_nonfinite_output(failure_client, valid_input_data, monkeypatch, value):
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", lambda row: [value])
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}


def test_model_loads_once(valid_input_data, monkeypatch):
    load = Mock(wraps=app_module.load_model)
    monkeypatch.setattr(app_module, "load_model", load)
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/predict", json=valid_input_data).status_code == 200
        load.assert_called_once_with()
    assert not hasattr(app.state, "bundle")


# 4. In-process TestClient timings; startup and warm-up are excluded.
def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    elapsed_ms = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
        assert math.isfinite(response.json()["prediction"])
    print(
        f"\nTestClient latency: n={len(elapsed_ms)}, warmup=5, "
        f"median={np.median(elapsed_ms):.3f} ms, "
        f"p95={np.percentile(elapsed_ms, 95):.3f} ms; "
        f"Python {platform.python_version()}, {platform.system()} {platform.machine()}"
    )
