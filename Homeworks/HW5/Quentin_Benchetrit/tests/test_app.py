"""Tests of the /predict service with the saved HW4 model."""

import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "hw4-signed-ridge-2026-06-30"
LATENCY_BUDGET_MS = 50  # see test_predict_performance


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
    # PARR on 2026-09-24, the last signal date of the HW4 test.
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def reference():
    # Input and forecast recorded by the HW4 notebook, before this API existed.
    return json.loads((ROOT / "artifacts" / "reference_example.json").read_text())


@pytest.fixture
def model_must_not_run(client, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("an invalid request reached the model")

    monkeypatch.setattr(app.state.bundle["model"], "predict", unexpected_call)


def direct_prediction(values):
    """The saved model called without the API, on a frame built independently."""
    bundle = app.state.bundle
    frame = pd.DataFrame([values])[bundle["features"]].astype("float32")
    return float(bundle["model"].predict(frame)[0])


# 1. Valid prediction


def test_predict_valid_input(client, valid_input_data, reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == MODEL_VERSION

    # Same float32 inputs through the same fitted ridge: only rounding can differ.
    assert body["prediction"] == pytest.approx(direct_prediction(valid_input_data), abs=1e-12)

    # HW4 scored this stock inside a batch of about 1,900 rows. The model computes in
    # float32, as in training, so one row and a batch can differ in the 9th decimal
    # (2e-9 here). 1e-8 allows for that and stays far below a basis point (1e-4).
    assert reference["features"] == valid_input_data
    assert body["prediction"] == pytest.approx(reference["hw4_prediction"], abs=1e-8)


def test_null_means_unavailable_and_takes_the_middle_rank(client, valid_input_data):
    with_null = client.post("/predict", json={**valid_input_data, "roa": None})
    with_middle = client.post("/predict", json={**valid_input_data, "roa": 0.5})
    assert with_null.status_code == with_middle.status_code == 200
    assert with_null.json()["prediction"] == pytest.approx(with_middle.json()["prediction"])
    expected = direct_prediction({**valid_input_data, "roa": np.nan})
    assert with_null.json()["prediction"] == pytest.approx(expected, abs=1e-12)


def test_field_order_in_the_request_does_not_matter(client, valid_input_data):
    reordered = dict(reversed(list(valid_input_data.items())))
    first = client.post("/predict", json=valid_input_data).json()["prediction"]
    assert client.post("/predict", json=reordered).json()["prediction"] == first


# 2. Input validation


def drop(name):
    return lambda data: {key: value for key, value in data.items() if key != name}


def change(name, value):
    return lambda data: {**data, name: value}


@pytest.mark.parametrize(
    "make_payload",
    [
        pytest.param(drop("beta"), id="missing field, 25 features"),
        pytest.param(change("ret_5", 0.5), id="27 features, ret_5 is not a model input"),
        pytest.param(change("sector", "Energy"), id="unknown categorical field"),
        pytest.param(change("mom_12_1", "0.93"), id="numeric string"),
        pytest.param(change("roa", True), id="boolean"),
        pytest.param(change("beta", float("nan")), id="NaN"),
        pytest.param(change("beta", float("inf")), id="infinity"),
        pytest.param(change("vol_63", 0.0), id="rank of zero"),
        pytest.param(change("vol_63", 1.2), id="rank above one"),
        pytest.param(change("vol_63", -0.3), id="negative rank"),
        pytest.param(lambda data: list(data.values()), id="list instead of named fields"),
        pytest.param(lambda data: {}, id="empty object"),
    ],
)
def test_predict_invalid_input(client, valid_input_data, model_must_not_run, make_payload):
    # json.dumps writes NaN and Infinity tokens, which the server must reject.
    payload = json.dumps(make_payload(valid_input_data))
    response = client.post(
        "/predict", content=payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_body_that_is_not_json_is_rejected(client, model_must_not_run):
    response = client.post(
        "/predict", content="mom_12_1=0.9", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# 3. Inference failure


def raises(*args, **kwargs):
    raise RuntimeError("simulated inference failure")


def returns_nan(*args, **kwargs):
    return np.array([np.nan])


@pytest.mark.parametrize("broken_predict", [raises, returns_nan], ids=["raises", "returns NaN"])
def test_predict_model_failure(
    failure_client, valid_input_data, monkeypatch, caplog, broken_predict
):
    monkeypatch.setattr(app.state.bundle["model"], "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "inference failed" in caplog.text  # the details stay in the server log
    assert "simulated" not in response.text and "Traceback" not in response.text


# 4. Performance


def test_predict_performance(client, valid_input_data, capsys):
    """In-process TestClient measurement (no network), 50 requests after 5 warm-ups.

    On an Apple M4 laptop the median is about 2 ms. The 50 ms budget on p95 only
    catches a gross regression, such as reloading the model on every request, and
    leaves room for slower machines.
    """
    for _ in range(5):
        client.post("/predict", json=valid_input_data)
    timings_ms = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median, p95 = statistics.median(timings_ms), float(np.percentile(timings_ms, 95))
    with capsys.disabled():
        print(
            f"\nlatency, in-process TestClient, 50 requests: median {median:.2f} ms, "
            f"p95 {p95:.2f} ms"
        )
    assert p95 < LATENCY_BUDGET_MS
