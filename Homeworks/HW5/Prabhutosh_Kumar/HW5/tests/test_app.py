"""Exercise the real fitted HW4 pipeline; mock only rejection/failure paths."""

import json
import math
import platform
import statistics
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import ARTIFACTS, ModelService, feature_frame

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata():
    return json.loads((ARTIFACTS / "metadata.json").read_text())


@pytest.fixture
def valid_input_data():
    return json.loads((HERE / "example_request.json").read_text())


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_predict_valid_input(client, valid_input_data, metadata):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"prediction", "model_version"}
    assert isinstance(result["prediction"], (int, float))
    assert math.isfinite(result["prediction"])
    assert result["model_version"] == metadata["model_version"]
    # Independent artifact load and pipeline call, outside service prediction logic.
    pipeline = joblib.load(ARTIFACTS / "model.joblib")
    direct = pipeline.predict(feature_frame(valid_input_data["features"]))[0]
    assert result["prediction"] == pytest.approx(direct, rel=1e-9, abs=1e-10)
    # Golden value recorded by HW4 before there was an HTTP wrapper.
    reference = metadata["reference"]
    assert valid_input_data == reference["request"]
    assert result["prediction"] == pytest.approx(
        reference["prediction"], rel=reference["rtol"], abs=reference["atol"]
    )


INVALID_CASES = [
    "missing",
    "extra",
    "short",
    "long",
    "empty",
    "null_list",
    "string_list",
    "string",
    "bool",
    "null",
    "object",
    "nan",
    "inf",
    "minus_inf",
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "volatility_21",
    "filing_age_days",
    "drawdown_low",
    "drawdown_high",
    "reversal_5",
]


@pytest.mark.parametrize("case", INVALID_CASES)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, case):
    body = deepcopy(valid_input_data)
    if case == "missing":
        body = {}
    elif case == "extra":
        body["unexpected"] = 1
    elif case == "short":
        body["features"].pop()
    elif case == "long":
        body["features"].append(1.0)
    elif case == "empty":
        body["features"] = []
    elif case == "null_list":
        body["features"] = None
    elif case == "string_list":
        body["features"] = "1,2,3"
    elif case in {"string", "bool", "null", "object", "nan", "inf", "minus_inf"}:
        body["features"][0] = {
            "string": "0.1",
            "bool": True,
            "null": None,
            "object": {},
            "nan": float("nan"),
            "inf": float("inf"),
            "minus_inf": -float("inf"),
        }[case]
    else:
        index, value = {
            "momentum_21": (0, -1.01),
            "momentum_63": (1, -1.01),
            "momentum_252_21": (2, -1.01),
            "volatility_63": (3, -0.1),
            "drawdown_low": (4, -1.01),
            "drawdown_high": (4, 0.01),
            "filing_age_days": (11, -1),
            "reversal_5": (12, 1.01),
            "volatility_21": (13, -0.1),
        }[case]
        body["features"][index] = value
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.service.model, "predict", spy)
    # Raw JSON permits deliberately testing nonstandard NaN/Infinity tokens too.
    response = client.post(
        "/predict", content=json.dumps(body), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    spy.assert_not_called()


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    spy = Mock(side_effect=RuntimeError("private internal model error"))
    monkeypatch.setattr(app.state.service.model, "predict", spy)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert "private internal model error" not in response.text
    assert "private internal model error" in caplog.text
    spy.assert_called_once()


@pytest.mark.parametrize("output", [[float("nan")], [float("inf")], [], [[0.1]], [0.1, 0.2]])
def test_bad_model_output_is_failure(failure_client, valid_input_data, monkeypatch, output):
    monkeypatch.setattr(app.state.service.model, "predict", lambda _: output)
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    durations = []
    for _ in range(50):
        start = perf_counter()
        response = client.post("/predict", json=valid_input_data)
        durations.append(1000 * (perf_counter() - start))
        assert response.status_code == 200
    report = {
        "method": "in-process Starlette TestClient; 5 warmups, 50 sequential requests",
        "clock": "time.perf_counter; milliseconds; includes validation and serialization",
        "median_ms": statistics.median(durations),
        "p95_ms": float(np.percentile(durations, 95)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    print("\nLATENCY_REPORT " + json.dumps(report, sort_keys=True))
    assert len(durations) == 50
    assert all(math.isfinite(duration) and duration > 0 for duration in durations)
    # No machine-independent latency budget; report observed measurements.


def test_model_loaded_once_per_lifespan(valid_input_data, monkeypatch):
    loader = Mock(wraps=ModelService.load)
    monkeypatch.setattr(ModelService, "load", loader)
    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/predict", json=valid_input_data).status_code == 200
        loader.assert_called_once_with()
    assert not hasattr(app.state, "service")


def test_artifact_checksum_checked(tmp_path, metadata):
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "model.joblib").write_bytes(b"corrupted artifact")
    with pytest.raises(ValueError, match="checksum"):
        ModelService.load(tmp_path)
