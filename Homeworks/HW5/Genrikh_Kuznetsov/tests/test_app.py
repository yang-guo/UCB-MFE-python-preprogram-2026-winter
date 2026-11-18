"""Tests for the HW5 service, using the real saved HW4 model except in the failure test."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import features_frame

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = json.loads((ROOT / "artifacts" / "reference_examples.json").read_text())[
    "examples"
]
MODEL_VERSION = "hw4-ridge-rank-a1e6-fit2026-06-30"


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
    # AAPL on 2026-08-31, as recorded in HW4 (also saved as example_request.json).
    return dict(REFERENCES[0]["input"])


@pytest.fixture
def model_must_not_run(client, monkeypatch):
    """Replace the loaded model's predict so any call fails the test.

    Depends on ``client`` so the lifespan has already loaded the model.
    """

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)


# ---------------------------------------------------------------- 1. valid prediction


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and np.isfinite(body["prediction"])
    assert body["model_version"] == MODEL_VERSION

    # API agrees with calling the saved pipeline directly on the same DataFrame.
    direct = app.state.bundle["pipeline"].predict(features_frame(valid_input_data))[0]
    assert body["prediction"] == pytest.approx(direct, abs=1e-12)


@pytest.mark.parametrize("example", REFERENCES, ids=[e["symbol"] for e in REFERENCES])
def test_predict_matches_hw4_recorded_score(client, example):
    """Scores were written by the HW4 notebook (outputs/test_scores.parquet) before the API.

    Same model, same float64 inputs, so the result should agree to rounding error; 1e-9 is
    far below the gap between neighbouring stocks' scores (~1e-5) that drives the ranking.
    The second example has null features, so this also checks missing-value handling.
    """
    response = client.post("/predict", json=example["input"])
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(example["hw4_score"], abs=1e-9)


def test_example_request_file_is_valid(client):
    payload = json.loads((ROOT / "example_request.json").read_text())
    assert client.post("/predict", json=payload).status_code == 200


def test_whole_numbers_accepted_for_float_fields(client, valid_input_data):
    # JSON 0 is a valid float (not a string or bool), so strict mode should accept it.
    response = client.post("/predict", json=valid_input_data | {"momentum_21": 0})
    assert response.status_code == 200


# ---------------------------------------------------------------- 2. input validation


def _without(data, key):
    return {k: v for k, v in data.items() if k != key}


INVALID_CASES = {
    "missing required field": lambda d: _without(d, "momentum_21"),
    "missing nullable field (must send null)": lambda d: _without(d, "net_margin"),
    "too few features (12)": lambda d: _without(d, "sector"),
    "too many features (extra field)": lambda d: d | {"surprise": 1.0},
    "null in non-nullable field": lambda d: d | {"volatility_63": None},
    "numeric string": lambda d: d | {"momentum_21": "0.02"},
    "boolean for float": lambda d: d | {"momentum_63": True},
    "boolean for int": lambda d: d | {"filing_age_days": False},
    "float for int": lambda d: d | {"filing_age_days": 30.5},
    "list instead of number": lambda d: d | {"momentum_21": [0.02]},
    "return below -100%": lambda d: d | {"momentum_63": -1.5},
    "return above cap": lambda d: d | {"momentum_252_21": 1e3},
    "negative volatility": lambda d: d | {"volatility_63": -0.1},
    "positive drawdown": lambda d: d | {"drawdown_252": 0.2},
    "negative debt ratio": lambda d: d | {"debt_to_assets": -0.5},
    "ratio out of range": lambda d: d | {"net_margin": 1e300},
    "negative filing age": lambda d: d | {"filing_age_days": -1},
    "unknown sector": lambda d: d | {"sector": "Crypto"},
    "sector wrong case": lambda d: d | {"sector": "technology"},
    "flat list instead of object": lambda d: list(d.values()),
    "empty object": lambda d: {},
}


@pytest.mark.usefixtures("model_must_not_run")
@pytest.mark.parametrize("make_payload", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_predict_invalid_input(client, valid_input_data, make_payload):
    response = client.post("/predict", json=make_payload(valid_input_data))
    assert response.status_code == 422


@pytest.mark.usefixtures("model_must_not_run")
@pytest.mark.parametrize("field", ["momentum_21", "net_margin"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_rejected(client, valid_input_data, field, value):
    # json.dumps writes the non-standard tokens NaN / Infinity, which the server's JSON
    # parser accepts, so these really reach validation. Nullable fields must use null.
    raw = json.dumps(valid_input_data | {field: value})
    response = client.post(
        "/predict", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


@pytest.mark.usefixtures("model_must_not_run")
def test_malformed_json_rejected(client):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------- 3. inference failure


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def broken_predict(*args, **kwargs):
        raise RuntimeError("simulated inference failure: secret internal detail")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "secret internal detail" not in response.text  # no internals leak to clients
    assert response.headers["X-Request-ID"]
    # ...but the full error is in the server log, tagged with the same request id.
    assert "simulated inference failure" in caplog.text
    assert response.headers["X-Request-ID"] in caplog.text


def test_non_finite_model_output_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(
        app.state.bundle["pipeline"], "predict", lambda *a, **k: np.array([np.nan])
    )
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


# ---------------------------------------------------------------- 4. performance

N_WARMUP = 5
N_TIMED = 200
# Budget for THIS measurement (in-process TestClient, no network). On a 2019 MacBook Pro
# (Intel i9-9980HK, 32 GB) the median is ~9 ms and p95 ~10 ms, mostly pandas/sklearn
# overhead for a one-row DataFrame. 100 ms is ~10x that: it catches gross regressions
# (e.g. reloading or refitting the model per request) without being flaky on a busy
# laptop or CI runner. Refitting per request would take minutes, not milliseconds.
P95_BUDGET_MS = 100.0


def test_predict_performance(client, valid_input_data):
    for _ in range(N_WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(N_TIMED):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = float(np.percentile(latencies_ms, 95))
    report = (
        f"\nLatency, in-process TestClient (no network), {N_TIMED} sequential requests "
        f"after {N_WARMUP} warm-up: median {median:.2f} ms, p95 {p95:.2f} ms, "
        f"max {max(latencies_ms):.2f} ms  [{platform.platform()}, {platform.machine()}, "
        f"Python {platform.python_version()}]"
    )
    print(report)
    assert p95 < P95_BUDGET_MS
