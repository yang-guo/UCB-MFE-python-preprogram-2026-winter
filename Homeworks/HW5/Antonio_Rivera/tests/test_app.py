"""Tests for the HW5 service, using the real exported HW4 model (model/pipeline.joblib)."""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import FEATURE_ORDER, MODEL_PATH, app

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "model" / "reference.json").read_text())
METADATA = json.loads((ROOT / "model" / "metadata.json").read_text())


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
    # The real HW4 row (symbol A, 2026-09-29) saved by export_model.py.
    return dict(REFERENCE["features"])


@pytest.fixture
def predict_calls(client, monkeypatch):
    """Wrap the loaded model's predict to count calls (it still runs the real model)."""
    model = client.app.state.model
    real_predict = model.predict
    calls = []

    def spy(X):
        calls.append(X)
        return real_predict(X)

    monkeypatch.setattr(model, "predict", spy)
    return calls


# 1. Valid prediction ----------------------------------------------------------------


def test_predict_valid_input(client, valid_input_data, predict_calls):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == METADATA["model_version"]
    assert len(predict_calls) == 1
    assert list(predict_calls[0].columns) == FEATURE_ORDER  # training column names/order

    # Same answer as calling the saved pipeline directly (independent load + DataFrame).
    frame = pd.DataFrame([valid_input_data])[FEATURE_ORDER]
    direct = joblib.load(MODEL_PATH).predict(frame)
    assert body["prediction"] == pytest.approx(float(direct[0]), abs=1e-12)

    # Same answer HW4 recorded for this row before any API existed
    # (outputs/scores_final_ridge_F1_baseline.parquet, see model/reference.json).
    # Tolerance 1e-9: identical code and library versions, so only float noise may differ.
    assert body["prediction"] == pytest.approx(REFERENCE["hw4_score"], abs=1e-9)


def test_predict_accepts_null_for_missing_features(client, valid_input_data):
    # null means "not available"; the pipeline imputes it as in HW4 training.
    payload = valid_input_data | {"momentum_252_21": None, "net_margin": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])
    assert response.json()["prediction"] != pytest.approx(REFERENCE["hw4_score"], abs=1e-9)


# 2. Input validation ----------------------------------------------------------------


def _without(data, key):
    return {k: v for k, v in data.items() if k != key}


INVALID_CASES = {
    "missing_numeric": lambda d: _without(d, "momentum_21"),
    "missing_sector": lambda d: _without(d, "sector"),
    "missing_nullable_field": lambda d: _without(d, "net_margin"),  # null ok, absent not
    "extra_field": lambda d: d | {"momentum_5": 0.01},
    "numeric_string": lambda d: d | {"momentum_21": "0.13"},
    "boolean": lambda d: d | {"volatility_63": True},
    "null_not_allowed": lambda d: d | {"momentum_21": None},
    "sector_not_string": lambda d: d | {"sector": 3},
    "unknown_sector": lambda d: d | {"sector": "Crypto"},
    "return_below_total_loss": lambda d: d | {"momentum_63": -1.5},
    "negative_volatility": lambda d: d | {"volatility_63": -0.1},
    "positive_drawdown": lambda d: d | {"drawdown_252": 0.2},
    "ratio_too_large": lambda d: d | {"net_margin": 1e6},
    "filing_age_too_large": lambda d: d | {"filing_age_days": 5000},
    "feature_list_instead_of_fields": lambda d: {"features": list(d.values())},
    "too_few_features": lambda d: dict(list(d.items())[:11]),
    "empty_body": lambda d: {},
}


@pytest.mark.parametrize("case", INVALID_CASES, ids=list(INVALID_CASES))
def test_predict_invalid_input(client, valid_input_data, predict_calls, case):
    response = client.post("/predict", json=INVALID_CASES[case](valid_input_data))
    assert response.status_code == 422
    assert predict_calls == []  # rejected before inference


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["momentum_21", "net_margin"])
def test_predict_rejects_non_finite(client, valid_input_data, predict_calls, field, bad):
    # Python's json writes NaN / Infinity tokens (not standard JSON); send them raw.
    text = json.dumps(valid_input_data | {field: bad})
    response = client.post(
        "/predict", content=text, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert predict_calls == []


def test_predict_rejects_non_json(client, predict_calls):
    response = client.post(
        "/predict", content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert predict_calls == []


# 3. Inference failure ---------------------------------------------------------------


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(X):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(failure_client.app.state.model, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "secret internal detail" not in response.text  # no detail leak
    assert "prediction" not in response.text  # not turned into a "success"
    assert "inference_failed" in caplog.text  # full exception stays in the server log


def test_predict_non_finite_model_output_is_500(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(failure_client.app.state.model, "predict", lambda X: np.array([np.nan]))
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


# 4. Performance ---------------------------------------------------------------------


def test_predict_performance(client, valid_input_data):
    for _ in range(5):  # warm-up: first calls pay one-time import/allocation costs
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(100):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = float(np.percentile(latencies_ms, 95))
    print(
        f"\nLatency, in-process TestClient (no real network), n={len(latencies_ms)}, "
        f"{platform.machine()} {platform.system()} (MacBook Air, Apple Silicon): "
        f"median {median:.2f} ms, p95 {p95:.2f} ms"
    )
    # Budget: a 1-row Ridge pipeline takes a few ms on a laptop. 250 ms p95 leaves wide room
    # for slow CI runners while still catching a gross regression (e.g. refitting per call).
    assert p95 < 250
