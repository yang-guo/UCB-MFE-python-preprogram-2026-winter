"""Replace each TODO failure with meaningful checks for your application."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import InputData, app, predict_signal_average


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
    path = Path(__file__).parent.parent / "example_request.json"
    return json.loads(path.read_text())


@pytest.fixture
def invalid_input_data():
    path = Path(__file__).parent.parent / "invalid_request.json"
    return json.loads(path.read_text())


@pytest.fixture
def reference():
    path = Path(__file__).parent.parent / "reference_predictions.json"
    return json.loads(path.read_text())[0]


def test_predict_valid_input(client, valid_input_data, reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    pred = response.json()["prediction"]

    assert pred == pytest.approx(
        predict_signal_average(client.app.state.bundle, InputData(**valid_input_data))
    )
    assert pred == pytest.approx(sum(reference["predictions"].values()) / 5, abs=1e-9)


def test_predict_invalid_input(client, invalid_input_data):
    invalid_input = invalid_input_data
    assert (
        client.post("/predict", json=invalid_input).status_code == 422
    )  #  Note sourced from Lecture 9 demo code.


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(failure_client.app.state.bundle["beta"], "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


def test_predict_performance(client, valid_input_data):
    # Warm up, then measure >=20 successful requests with time.perf_counter.
    Number_Tests = 20
    Test_Times = []
    input_data = valid_input_data

    # 3 warmups
    new_input = {k: v * (1 + random.uniform(-0.01, 0.01)) for k, v in input_data.items()}
    response = client.post("/predict", json=new_input)
    new_input = {k: v * (1 + random.uniform(-0.01, 0.01)) for k, v in input_data.items()}
    response = client.post("/predict", json=new_input)
    new_input = {k: v * (1 + random.uniform(-0.01, 0.01)) for k, v in input_data.items()}
    response = client.post("/predict", json=new_input)
    new_input = {k: v * (1 + random.uniform(-0.01, 0.01)) for k, v in input_data.items()}

    for i in range(Number_Tests):
        start_time = time.perf_counter()
        response = client.post("/predict", json=new_input)
        end_time = time.perf_counter()
        assert response.status_code == 200
        new_input = {k: v * (1 + random.uniform(-0.01, 0.01)) for k, v in input_data.items()}
        Test_Times.append(end_time - start_time)

    print(
        f"median {np.median(Test_Times) * 1000:.2f} ms, p95 {np.percentile(Test_Times, 95) * 1000:.2f} ms"
    )
    # Report median and p95; document an environment-specific budget if used.
