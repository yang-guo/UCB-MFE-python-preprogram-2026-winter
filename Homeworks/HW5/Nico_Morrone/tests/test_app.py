"""Replace each TODO failure with meaningful checks for your application."""

import json
import math
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import make_feature_frame, predict_frame

SUBMISSION = Path(__file__).resolve().parents[1]


@pytest.fixture
def valid_input_data():
    return json.loads((SUBMISSION / "example_request.json").read_text())


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_predict_valid_input(client, valid_input_data):
    reference = json.loads((SUBMISSION / "hw4_reference.json").read_text())

    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert type(body["prediction"]) is float
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == reference["model_version"]

    bundle = app.state.bundle
    frame = make_feature_frame(valid_input_data, bundle["features"])
    direct_prediction = float(predict_frame(bundle, frame)[0])

    assert body["prediction"] == pytest.approx(direct_prediction, abs=1e-12, rel=0)
    assert body["prediction"] == pytest.approx(reference["prediction"], abs=1e-10, rel=0)


@pytest.mark.parametrize(
    "changes, omitted",
    [
        ({}, ("momentum_63d",)),
        ({}, ("momentum_63d", "volatility_21d")),
        ({"extra_feature": 1.0}, ()),
        ({"beta_126d": "1.2"}, ()),
        ({"beta_126d": True}, ()),
        ({"beta_126d": None}, ()),
        ({"beta_126d": [1.2]}, ()),
        ({"beta_126d": float("nan")}, ()),
        ({"beta_126d": float("inf")}, ()),
        ({"beta_126d": float("-inf")}, ()),
        ({"momentum_63d": -1.1}, ()),
        ({"volatility_21d": -0.1}, ()),
        ({"max_drawdown_63d": -1.1}, ()),
        ({"max_drawdown_63d": 0.1}, ()),
    ],
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, changes, omitted):
    inference = Mock()
    monkeypatch.setattr("app.predict_frame", inference)

    payload = {**valid_input_data, **changes}
    for field in omitted:
        del payload[field]

    response = client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    inference.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    broken_predict = Mock(side_effect=RuntimeError("deliberate test error"))
    monkeypatch.setattr(app.state.bundle["model"], "predict", broken_predict)

    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed"}
    broken_predict.assert_called_once()
    assert "deliberate test error" in caplog.text


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    times_ms = []
    for _ in range(30):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        times_ms.append((perf_counter() - start) * 1_000)
        assert response.status_code == 200

    p95 = quantiles(times_ms, n=100, method="inclusive")[94]
    print(
        f"In-process TestClient, 30 requests: "
        f"median={median(times_ms):.3f} ms, p95={p95:.3f} ms"
    )
