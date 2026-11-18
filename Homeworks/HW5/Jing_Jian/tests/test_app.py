"""Real-model parity, strict validation, failures, lifecycle, and latency."""

import json
import math
import platform
import time
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as service
from app import InputData, OutputData, app
from model_tools import FEATURES, MODEL_VERSION, ROOT, feature_frame


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


@pytest.fixture
def reference():
    return json.loads((ROOT / "artifacts/reference_prediction.json").read_text())


def test_predict_valid_input(client, valid_input_data, reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = OutputData.model_validate(response.json())
    assert set(response.json()) == {"prediction", "model_version"}
    assert math.isfinite(result.prediction)
    assert result.model_version == MODEL_VERSION
    assert valid_input_data == reference["input"]
    direct = app.state.bundle["pipeline"].predict(feature_frame(valid_input_data))[0]
    assert result.prediction == pytest.approx(direct, rel=1e-12, abs=1e-12)
    # This expected result was saved by HW4 before this API was written.
    assert result.prediction == pytest.approx(
        reference["prediction"], rel=reference["rtol"], abs=reference["atol"]
    )


INVALID_FIELDS = [
    (name, value)
    for name in FEATURES
    for value in [None, "0.1", True, [], {}, float("nan"), float("inf"), float("-inf")]
] + [
    ("momentum_21", -1),
    ("momentum_63", -1.01),
    ("momentum_252_21", -1),
    ("volatility_63", -0.01),
    ("volatility_63", 0.600001),
    ("drawdown_252", -1),
    ("drawdown_252", 0.01),
]


@pytest.mark.parametrize("name,value", INVALID_FIELDS)
def test_invalid_field_never_reaches_inference(
    client, valid_input_data, monkeypatch, name, value
):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    payload = {**valid_input_data, name: value}
    # raw JSON is intentional: httpx's json= correctly refuses NaN/Infinity.
    response = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    spy.assert_not_called()


@pytest.mark.parametrize("missing", FEATURES)
def test_missing_field(client, valid_input_data, monkeypatch, missing):
    spy = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    payload = valid_input_data.copy()
    del payload[missing]
    assert client.post("/predict", json=payload).status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("payload", [{}, [], [0.1] * 4, [0.1] * 5, [0.1] * 6, None])
def test_invalid_shape(client, monkeypatch, payload):
    spy = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    assert client.post("/predict", json=payload).status_code == 422
    spy.assert_not_called()


def test_extra_field(client, valid_input_data, monkeypatch):
    spy = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    assert (
        client.post("/predict", json={**valid_input_data, "symbol": "AAL"}).status_code == 422
    )
    spy.assert_not_called()


def test_malformed_json(client, monkeypatch):
    spy = Mock()
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", spy)
    assert (
        client.post(
            "/predict", content="{", headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )
    spy.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("private inference failure detail")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "private inference failure detail" not in response.text
    assert "private inference failure detail" in caplog.text


@pytest.mark.parametrize("output", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_prediction_is_failure(failure_client, valid_input_data, monkeypatch, output):
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", lambda frame: [output])
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_named_column_order_and_no_request_fitting(client, valid_input_data, monkeypatch):
    pipeline = app.state.bundle["pipeline"]
    original_predict = pipeline.predict
    spy = Mock(wraps=original_predict)
    monkeypatch.setattr(pipeline, "predict", spy)
    monkeypatch.setattr(
        pipeline, "fit", Mock(side_effect=AssertionError("Refitting is forbidden"))
    )
    reverse_order = dict(reversed(list(valid_input_data.items())))
    assert client.post("/predict", json=reverse_order).status_code == 200
    assert spy.call_args.args[0].columns.tolist() == FEATURES
    pipeline.fit.assert_not_called()


def test_boundary_inputs(client):
    payload = dict.fromkeys(FEATURES, 0)
    payload["volatility_63"] = 0.6
    assert client.post("/predict", json=payload).status_code == 200
    payload["volatility_63"] = 0
    assert client.post("/predict", json=payload).status_code == 200
    InputData.model_validate(payload)


def test_lifespan_loads_once_and_cleans_up(monkeypatch, valid_input_data):
    load = Mock(wraps=service.load_bundle)
    monkeypatch.setattr(service, "load_bundle", load)
    with TestClient(app) as test_client:
        for _ in range(3):
            assert test_client.post("/predict", json=valid_input_data).status_code == 200
        load.assert_called_once()
    assert not hasattr(app.state, "bundle")


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    elapsed = []
    for _ in range(50):
        started = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        elapsed.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200
        assert math.isfinite(response.json()["prediction"])
    report = {
        "method": "in-process FastAPI/Starlette TestClient; sequential POST requests",
        "clock": "time.perf_counter",
        "warmup_requests": 5,
        "measured_requests": len(elapsed),
        "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "hardware": platform.platform() + " " + platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "budget": "No universal latency assertion; report observed successful-request latency",
    }
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
