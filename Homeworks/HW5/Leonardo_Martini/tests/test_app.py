import json
import math
import platform
import statistics
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from model_utils import predict_one

REFERENCE = json.loads((Path(__file__).parent / "reference.json").read_text())


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
    return dict(REFERENCE["input"])


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"].startswith("hw4-")

    # same result as calling the saved pipeline directly
    direct = predict_one(app.state.model, valid_input_data)
    assert body["prediction"] == pytest.approx(direct, rel=1e-12, abs=1e-12)
    # same result as recorded in HW4 (same library versions, so only float noise allowed)
    assert body["prediction"] == pytest.approx(REFERENCE["prediction"], rel=1e-9, abs=1e-9)


def test_predict_accepts_null_for_optional_features(client, valid_input_data):
    valid_input_data["net_margin"] = None
    valid_input_data["momentum_252_21"] = None
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def _with(base, **changes):
    data = dict(base)
    data.update(changes)
    return data


def _without(base, key):
    data = dict(base)
    del data[key]
    return data


INVALID_CASES = {
    "missing field": lambda d: _without(d, "momentum_21"),
    "missing sector": lambda d: _without(d, "sector"),
    "extra field": lambda d: _with(d, price=100.0),
    "null for required price feature": lambda d: _with(d, momentum_21=None),
    "numeric string": lambda d: _with(d, momentum_21="0.1"),
    "boolean": lambda d: _with(d, volatility_63=True),
    "list instead of number": lambda d: _with(d, momentum_63=[0.1]),
    "momentum below -100%": lambda d: _with(d, momentum_21=-1.5),
    "negative volatility": lambda d: _with(d, volatility_63=-0.1),
    "positive drawdown": lambda d: _with(d, drawdown_252=0.2),
    "ratio too large": lambda d: _with(d, net_margin=5000.0),
    "negative filing age": lambda d: _with(d, filing_age_days=-1),
    "unknown sector": lambda d: _with(d, sector="Crypto"),
    "sector as number": lambda d: _with(d, sector=3),
    "too few features": lambda d: {k: d[k] for k in list(d)[:5]},
    "empty body": lambda d: {},
    "list of features": lambda d: list(d.values()),
}


@pytest.mark.parametrize("case", INVALID_CASES)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, case):
    calls = []
    model = app.state.model
    monkeypatch.setattr(model, "predict", lambda X: calls.append(X))
    response = client.post("/predict", json=INVALID_CASES[case](valid_input_data))
    assert response.status_code == 422
    assert calls == []  # rejected before inference


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite(client, valid_input_data, bad):
    # JSON has no NaN/inf, so send the raw text that Python's json would produce
    text = json.dumps(valid_input_data).replace(
        '"momentum_21": 0.02570327927228', f'"momentum_21": {bad}'
    )
    assert bad in text
    response = client.post(
        "/predict", content=text, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def broken_predict(X):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(app.state.model, "predict", broken_predict)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "simulated failure" not in response.text  # details stay in the server log


def test_predict_performance(client, valid_input_data):
    for _ in range(5):  # warm up
        assert client.post("/predict", json=valid_input_data).status_code == 200

    times = []
    for _ in range(50):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        times.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[-1]
    print(
        f"\nIn-process TestClient, 50 requests on {platform.processor() or platform.machine()}, "
        f"Python {platform.python_version()}: median {median:.1f} ms, p95 {p95:.1f} ms"
    )
    # generous budget for a laptop: one row through the pipeline should take well under 1 s
    assert p95 < 1000
