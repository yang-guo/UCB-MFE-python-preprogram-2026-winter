"""HW5 tests; fixture/failure patterns adapted from Lecture 9/demo/tests."""

import json
import math
import os
import platform
import statistics
import time
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_service
from app import InputData, app
from model_service import ROOT

FEATURES = list(InputData.model_fields)


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def valid_input_data():
    return json.loads((ROOT / "example_request.json").read_text())


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert isinstance(result["prediction"], (int, float)) and math.isfinite(
        result["prediction"]
    )
    reference = json.loads((ROOT / "artifacts/reference_prediction.json").read_text())
    assert result["model_version"] == reference["model_version"]
    direct = app.state.bundle["pipeline"].predict(
        pd.DataFrame([valid_input_data], columns=app.state.bundle["features"])
    )[0]
    for expected in (direct, reference["prediction"]):
        assert result["prediction"] == pytest.approx(
            expected, abs=reference["absolute_tolerance"], rel=reference["relative_tolerance"]
        )


def assert_rejected(client, payload, monkeypatch):
    spy = Mock(return_value=np.array([0.0]))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    body = payload if isinstance(payload, str) else json.dumps(payload)
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422, response.text
    spy.assert_not_called()


@pytest.mark.parametrize("field", FEATURES)
def test_missing_field(client, valid_input_data, monkeypatch, field):
    del valid_input_data[field]
    assert_rejected(client, valid_input_data, monkeypatch)


@pytest.mark.parametrize("field", FEATURES)
@pytest.mark.parametrize(
    "value", ["0.1", True, False, None, [], {}, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_value(client, valid_input_data, monkeypatch, field, value):
    valid_input_data[field] = value
    assert_rejected(client, valid_input_data, monkeypatch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("momentum_21", -1),
        ("momentum_63", -1),
        ("momentum_252_21", -1),
        ("volatility_63", -0.01),
        ("drawdown_252", -1),
        ("drawdown_252", 0.01),
        ("net_margin", 0),
        ("net_margin", 1.01),
        ("operating_roa", 0),
        ("operating_roa", 1.01),
        ("debt_to_assets", 0),
        ("debt_to_assets", 1.01),
        ("cash_to_assets", 0),
        ("cash_to_assets", 1.01),
        ("filing_age_days", -1),
        ("filing_age_days", 23.5),
    ],
)
def test_invalid_range(client, valid_input_data, monkeypatch, field, value):
    valid_input_data[field] = value
    assert_rejected(client, valid_input_data, monkeypatch)


def test_extra_field(client, valid_input_data, monkeypatch):
    valid_input_data["unexpected"] = 1
    assert_rejected(client, valid_input_data, monkeypatch)


@pytest.mark.parametrize("payload", [[], [0.1] * 9, [0.1] * 10, [0.1] * 11, {}, "{broken"])
def test_invalid_structure(client, monkeypatch, payload):
    assert_rejected(client, payload, monkeypatch)


def test_numeric_overflow(client, valid_input_data, monkeypatch):
    body = json.dumps(valid_input_data).replace(str(valid_input_data["momentum_21"]), "1e309")
    assert_rejected(client, body, monkeypatch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("momentum_21", 0),
        ("momentum_63", -0.999),
        ("momentum_252_21", 2),
        ("volatility_63", 0),
        ("drawdown_252", 0),
        ("net_margin", 1),
        ("operating_roa", 1),
        ("debt_to_assets", 1),
        ("cash_to_assets", 1),
        ("filing_age_days", 0),
        ("filing_age_days", 23.0),
    ],
)
def test_valid_boundaries(client, valid_input_data, field, value):
    valid_input_data[field] = value
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def test_key_order_does_not_change_prediction(client, valid_input_data):
    first = client.post("/predict", json=valid_input_data).json()
    second = client.post("/predict", json=dict(reversed(list(valid_input_data.items())))).json()
    assert first == second


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("internal failure detail")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "internal failure detail" not in response.text
    assert "internal failure detail" in caplog.text


@pytest.mark.parametrize("result", [[float("nan")], [float("inf")], [], [1, 2]])
def test_bad_model_output_is_failure(failure_client, valid_input_data, monkeypatch, result):
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", lambda *_: result)
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_startup_loads_once_and_requests_do_not_fit(valid_input_data, monkeypatch):
    loader = Mock(wraps=model_service.joblib.load)
    monkeypatch.setattr(model_service.joblib, "load", loader)
    with TestClient(app) as client:
        fit = Mock(side_effect=RuntimeError("A request must not train"))
        monkeypatch.setattr(app.state.bundle["pipeline"], "fit", fit)
        for _ in range(3):
            assert client.post("/predict", json=valid_input_data).status_code == 200
        fit.assert_not_called()
        assert loader.call_count == 1
    assert not hasattr(app.state, "bundle")


def test_startup_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ready"


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    milliseconds = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed = (time.perf_counter() - start) * 1000
        assert response.status_code == 200
        milliseconds.append(elapsed)
    report = {
        "method": "In-process FastAPI TestClient; sequential requests; perf_counter",
        "warmup_requests": 5,
        "measured_successful_requests": len(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p95_method": "NumPy percentile, default linear interpolation",
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "latency_budget": "No fixed budget: local measurement, not a load test",
    }
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
