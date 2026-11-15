import json
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import MODEL_PATH, app

HW5_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def reference():
    path = HW5_DIR / "reference_prediction.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_valid_prediction(client, reference):
    payload = reference["input"]
    features = pd.DataFrame([payload], columns=["score_rank"])
    model = joblib.load(MODEL_PATH)
    direct_prediction = float(model.predict(features)[0])

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert type(body["prediction"]) is float
    assert body["model_version"] == f"hw4-{reference['hw4_commit']}"
    assert body["prediction"] == pytest.approx(direct_prediction, abs=1e-12, rel=0)
    assert body["prediction"] == pytest.approx(
        reference["expected_prediction"], abs=1e-12, rel=0
    )


@pytest.mark.parametrize(
    "body",
    [
        "{}",
        '{"score_rank": "0.42"}',
        '{"score_rank": true}',
        '{"score_rank": null}',
        '{"score_rank": 0.42, "extra": 1}',
        '{"score_rank": [0.42, 0.6]}',
        "[0.42, 0.6]",
        '{"score_rank": 0}',
        '{"score_rank": 1.01}',
        '{"score_rank": 1e309}',
        '{"score_rank": NaN}',
    ],
)
def test_invalid_input_does_not_call_model(client, monkeypatch, body):
    predict = Mock()
    monkeypatch.setattr(app.state.model, "predict", predict)

    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    predict.assert_not_called()


def test_inference_failure(client, reference, monkeypatch):
    predict = Mock(side_effect=RuntimeError("simulated model failure"))
    monkeypatch.setattr(app.state.model, "predict", predict)

    response = client.post("/predict", json=reference["input"])

    assert response.status_code == 500
    predict.assert_called_once()


def test_prediction_latency(client, reference):
    payload = reference["input"]
    for _ in range(5):
        assert client.post("/predict", json=payload).status_code == 200

    durations_ms = []
    for _ in range(30):
        start = perf_counter()
        response = client.post("/predict", json=payload)
        durations_ms.append((perf_counter() - start) * 1000)
        assert response.status_code == 200

    p95 = np.percentile(durations_ms, 95)
    print(
        f"\nTestClient in-process, 5 warm-ups, 30 requests "
        f"| median={median(durations_ms):.3f} ms | p95={p95:.3f} ms"
    )
