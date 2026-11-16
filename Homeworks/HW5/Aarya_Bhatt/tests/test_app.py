import math
import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURE_COLS, app

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "hw4_model.joblib"


@pytest.fixture
def artifact():
    return joblib.load(ARTIFACT_PATH)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data(artifact):
    return dict(artifact["reference_input"])


def test_predict_valid_input(client, artifact, valid_input_data):
    res = client.post("/predict", json=valid_input_data)
    assert res.status_code == 200

    body = res.json()
    assert set(body.keys()) == {"prediction", "model_version"}
    assert math.isfinite(body["prediction"])
    assert body["model_version"] == artifact["model_version"]

    sample_df = pd.DataFrame([valid_input_data], columns=FEATURE_COLS)
    direct_pred = float(artifact["model"].predict(sample_df)[0])
    assert math.isclose(body["prediction"], direct_pred, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(
        body["prediction"], artifact["reference_prediction"], rel_tol=1e-6, abs_tol=1e-6
    )


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"ret_5d_z": 0.1},
        {col: 0.1 for col in FEATURE_COLS[:-1]},
        {**{col: 0.1 for col in FEATURE_COLS}, "extra_feature": 0.5},
        {**{col: 0.1 for col in FEATURE_COLS}, "ret_5d_z": "0.1"},
        {**{col: 0.1 for col in FEATURE_COLS}, "ret_5d_z": True},
        {**{col: 0.1 for col in FEATURE_COLS}, "vol_21d_z": 25.0},
    ],
)
def test_predict_invalid_input(client, monkeypatch, bad_payload):
    called = False

    def spy_predict(*args, **kwargs):
        nonlocal called
        called = True
        return [0.0]

    monkeypatch.setattr(client.app.state.model, "predict", spy_predict)
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422
    assert not called


def test_predict_non_finite_input(client, valid_input_data):
    parts = ['"ret_5d_z": NaN'] + [
        f'"{k}": {v}' for k, v in valid_input_data.items() if k != "ret_5d_z"
    ]
    raw_json = "{" + ", ".join(parts) + "}"
    res = client.post(
        "/predict", content=raw_json, headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 422


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated model error")

    monkeypatch.setattr(failure_client.app.state.model, "predict", broken_predict)
    res = failure_client.post("/predict", json=valid_input_data)
    assert res.status_code == 500


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        client.post("/predict", json=valid_input_data)

    latencies_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        res = client.post("/predict", json=valid_input_data)
        t1 = time.perf_counter()
        assert res.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    median_ms = latencies_ms[len(latencies_ms) // 2]
    p95_ms = latencies_ms[int(len(latencies_ms) * 0.95)]
    print(
        f"\nIn-process TestClient latency (n=30): median={median_ms:.2f} ms, p95={p95_ms:.2f} ms"
    )
    assert median_ms < 50.0
