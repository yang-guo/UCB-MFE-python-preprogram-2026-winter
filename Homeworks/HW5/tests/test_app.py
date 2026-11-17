"""Test the real pipeline, validation, error handling and request latency."""

import json
import math
import platform
import statistics
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import OutputData, app
from model_service import FEATURES, feature_frame

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = {"rel": 1e-9, "abs": 1e-12}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def reference():
    return json.loads((ROOT / "reference_prediction.json").read_text(encoding="utf-8"))


@pytest.fixture
def valid_input_data():
    return json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("missing_features", [False, True])
def test_predict_valid_input(client, valid_input_data, reference, missing_features):
    assert valid_input_data == reference["request"]
    if missing_features:
        valid_input_data.update(op_margin=None, ret126=None, ret252skip21=None)
    # Reverse client-side key order to exercise the reconstruction of training order.
    payload = dict(reversed(list(valid_input_data.items())))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert type(result["prediction"]) is float
    parsed = OutputData.model_validate(result)
    assert math.isfinite(parsed.prediction)
    assert parsed.model_version == reference["model_version"]
    frame = feature_frame(payload)
    assert tuple(frame.columns) == FEATURES
    if missing_features:
        assert frame[["op_margin", "ret126", "ret252skip21"]].isna().all(axis=None)
    expected = app.state.model.predict(frame).item()
    assert parsed.prediction == pytest.approx(expected, **TOLERANCE)
    if not missing_features:
        assert parsed.prediction == pytest.approx(reference["expected_prediction"], **TOLERANCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ret21", "0.1"),
        ("ret63", True),
        ("ret21", None),
        ("ret252skip21", -1),
        ("vol63", -0.001),
        ("vol63", float("nan")),
        ("op_margin", float("inf")),
        ("cashflow_assets", float("-inf")),
        ("debt_assets", []),
        ("surplus", 0),
    ],
)
def test_invalid_fields(client, valid_input_data, monkeypatch, field, value):
    predict = Mock()
    monkeypatch.setattr(app.state.model, "predict", predict)
    valid_input_data[field] = value
    response = client.post(
        "/predict",
        content=json.dumps(valid_input_data),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    predict.assert_not_called()


@pytest.mark.parametrize(
    "case", ["missing_return", "missing_nullable", "short_list", "long_list", "empty"]
)
def test_wrong_shape(client, valid_input_data, monkeypatch, case):
    predict = Mock()
    monkeypatch.setattr(app.state.model, "predict", predict)
    payload = valid_input_data
    if case == "missing_return":
        del payload["ret21"]
    elif case == "missing_nullable":
        del payload["debt_assets"]
    elif case == "short_list":
        payload = list(payload.values())[:-1]
    elif case == "long_list":
        payload = [*payload.values(), 0]
    else:
        payload = {}
    assert client.post("/predict", json=payload).status_code == 422
    predict.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    failing = Mock(side_effect=RuntimeError("forced private failure"))
    monkeypatch.setattr(app.state.model, "predict", failing)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "forced private failure" not in response.text
    assert "forced private failure" in caplog.text
    failing.assert_called_once()


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    samples = []
    for _ in range(40):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        samples.append((perf_counter() - start) * 1000)
        assert response.status_code == 200
    p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
    print(
        f"In-process TestClient, Python {platform.python_version()}, {len(samples)} requests: "
        f"median {statistics.median(samples):.3f} ms, p95 {p95:.3f} ms"
    )
