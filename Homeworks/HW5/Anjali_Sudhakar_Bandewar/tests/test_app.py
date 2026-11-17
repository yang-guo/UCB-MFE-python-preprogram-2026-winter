"""Tests for the HW4 model service: valid prediction, validation, inference failure, latency."""

import json
import math
import platform
import statistics
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import hw4_model
from app import app

# Same artifact, same numpy/scikit-learn versions, same platform: results should agree to
# floating-point round-off. 1e-12 absolute on a 0-1 rank score is far below any meaningful
# difference while not depending on bit-for-bit equality.
TOLERANCE = 1e-12


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def reference():
    """Input row and prediction recorded by export_model.py from the HW4-code pipeline.

    Row: AAPL on 2015-01-02 (first session of the 2015 HW4 holdout fold). The prediction was
    made by HW4 train_baselines.hgb_pipeline after its 2015 holdout metrics reproduced the
    values recorded in HW4 results/models/selected_by_year.csv.
    """
    return json.loads(hw4_model.REFERENCE_PATH.read_text())


@pytest.fixture
def valid_input_data(reference):
    return dict(reference["input"])


@pytest.fixture(scope="session")
def direct_pipeline():
    """An independently loaded copy of the artifact, used outside the API."""
    return hw4_model.load_pipeline()


@pytest.fixture
def predict_spy(client, monkeypatch):
    """Wrap the loaded model's predict to count calls while still running real inference."""
    model = client.app.state.model
    calls = []
    real_predict = model.predict

    def spy(X):
        calls.append(X)
        return real_predict(X)

    monkeypatch.setattr(model, "predict", spy)
    return calls


def direct_predict(pipeline, features):
    """Call the pipeline directly with a training-shaped DataFrame (no API code)."""
    frame = pd.DataFrame([features], columns=hw4_model.FEATURES).astype(
        {name: "float64" for name in hw4_model.FEATURES if name != "universe_rank"}
    )
    return float(pipeline.predict(frame)[0])


# ---------------------------------------------------------------- 1. valid prediction


def test_predict_valid_input(client, valid_input_data, reference, direct_pipeline, predict_spy):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == hw4_model.MODEL_VERSION == reference["model_version"]
    assert len(predict_spy) == 1
    # The API passes the exact training columns, in training order.
    assert list(predict_spy[0].columns) == hw4_model.FEATURES

    # API == direct pipeline call == prediction recorded from HW4 code before the API existed.
    assert body["prediction"] == pytest.approx(
        direct_predict(direct_pipeline, valid_input_data), abs=TOLERANCE
    )
    assert body["prediction"] == pytest.approx(reference["prediction"], abs=TOLERANCE)


def test_predict_missing_features_as_null(client, valid_input_data, direct_pipeline):
    """null means missing (NaN), as in HW4 training; the fitted pipeline handles NaN."""
    features = {
        **valid_input_data,
        "momentum_12m": None,
        "profit_margin": None,
        "fcf_margin": None,
    }
    response = client.post("/predict", json=features)

    assert response.status_code == 200
    as_nan = {k: np.nan if v is None else v for k, v in features.items()}
    expected = direct_predict(direct_pipeline, as_nan)
    assert response.json()["prediction"] == pytest.approx(expected, abs=TOLERANCE)


def test_model_loaded_once_per_process(client, valid_input_data, monkeypatch):
    """Requests reuse the lifespan-loaded model; they never reload or refit it."""

    def fail(*args, **kwargs):
        raise AssertionError("model loaded or fitted during a request")

    monkeypatch.setattr(hw4_model, "load_pipeline", fail)
    monkeypatch.setattr(client.app.state.model, "fit", fail)
    for _ in range(3):
        assert client.post("/predict", json=valid_input_data).status_code == 200


# ---------------------------------------------------------------- 2. input validation


def _replace(**changes):
    return lambda data: {**data, **changes}


def _drop(name):
    return lambda data: {k: v for k, v in data.items() if k != name}


INVALID_CASES = {
    "missing_field": _drop("momentum_12m"),
    "missing_rank": _drop("universe_rank"),
    "extra_field": _replace(adj_close=150.0),
    "too_few_features_10": lambda d: dict(list(d.items())[:10]),
    "too_many_features_12": _replace(momentum_24m=0.5),
    "numeric_string": _replace(momentum_1m="0.1"),
    "boolean_float": _replace(profit_margin=True),
    "boolean_int": _replace(universe_rank=True),
    "float_rank": _replace(universe_rank=4.5),
    "string_rank": _replace(universe_rank="4"),
    "null_rank": _replace(universe_rank=None),
    "rank_below_1": _replace(universe_rank=0),
    "rank_above_2000": _replace(universe_rank=2001),
    "momentum_at_minus_1": _replace(momentum_3m=-1.0),
    "negative_age": _replace(fundamental_age_days=-1),
    "age_above_548": _replace(fundamental_age_days=549),
    "list_value": _replace(momentum_6m=[0.1]),
}


@pytest.mark.parametrize("make_body", INVALID_CASES.values(), ids=INVALID_CASES.keys())
def test_predict_invalid_input(client, valid_input_data, predict_spy, make_body):
    response = client.post("/predict", json=make_body(valid_input_data))

    assert response.status_code == 422
    assert predict_spy == []  # rejected before inference


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_predict_non_finite_input(client, valid_input_data, predict_spy, literal):
    # Python's json module (used by FastAPI) parses these non-standard literals into floats.
    body = json.dumps({**valid_input_data, "momentum_1m": 123.456}).replace("123.456", literal)
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert predict_spy == []


@pytest.mark.parametrize(
    "body",
    [None, [], [0.1] * 11, {"features": [0.1] * 11}, "not json"],
    ids=["empty", "empty_list", "bare_feature_list", "wrapped_list", "not_json"],
)
def test_predict_wrong_shape(client, predict_spy, body):
    if body == "not json":
        response = client.post(
            "/predict", content=b"not json", headers={"Content-Type": "application/json"}
        )
    else:
        response = client.post("/predict", json=body)

    assert response.status_code == 422
    assert predict_spy == []


# ---------------------------------------------------------------- 3. inference failure


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated inference failure: secret internal detail")

    monkeypatch.setattr(failure_client.app.state.model, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert "secret internal detail" not in response.text  # details stay in server logs
    assert "prediction" not in response.text  # no default prediction is returned


def test_predict_non_finite_model_output_is_error(
    failure_client, valid_input_data, monkeypatch
):
    monkeypatch.setattr(failure_client.app.state.model, "predict", lambda X: np.array([np.nan]))

    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


# ---------------------------------------------------------------- 4. performance

WARMUP_REQUESTS = 5
MEASURED_REQUESTS = 50
# Measured on an Apple-silicon MacBook (see README): median 5-10 ms, p95 7-30 ms. A 250 ms p95
# budget is loose enough not to be flaky on a slower or busy machine, but still catches
# pathological regressions such as reloading the artifact or refitting per request (seconds).
P95_BUDGET_SECONDS = 0.250


def test_predict_performance(client, valid_input_data):
    for _ in range(WARMUP_REQUESTS):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies = []
    for _ in range(MEASURED_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies.append(time.perf_counter() - start)
        assert response.status_code == 200

    median = statistics.median(latencies)
    p95 = float(np.percentile(latencies, 95))
    print(
        f"\nLATENCY (in-process FastAPI TestClient, not real HTTP; "
        f"{WARMUP_REQUESTS} warm-up + {MEASURED_REQUESTS} measured sequential requests, "
        f"time.perf_counter): median={median * 1000:.2f} ms, p95={p95 * 1000:.2f} ms, "
        f"max={max(latencies) * 1000:.2f} ms | {platform.platform()}, "
        f"{platform.machine()}, Python {platform.python_version()}"
    )
    assert len(latencies) >= 20
    assert p95 < P95_BUDGET_SECONDS
