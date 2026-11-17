"""Verify the exported HW4 return model, API boundaries, and runtime behavior."""

import json
import platform
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from model_service import FEATURE_NAMES, FrozenFF5

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "reference_prediction.json").read_text())


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_input_data():
    return REFERENCE["input"].copy()


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "prediction",
        "model_version",
        "symbol",
        "prediction_horizon_calendar_days",
    }
    direct = app.state.model.predict(
        valid_input_data["symbol"], [valid_input_data[k] for k in FEATURE_NAMES]
    )
    for expected in [direct, REFERENCE["expected"]]:
        assert body["model_version"] == expected["model_version"]
        assert body["symbol"] == expected["symbol"] == "AAPL"
        assert body["prediction_horizon_calendar_days"] == 90
        np.testing.assert_allclose(
            body["prediction"], expected["prediction"], **REFERENCE["tolerance"]
        )


def test_coefficients_match_saved_hw4_training_data(client):
    fixture = np.genfromtxt(
        ROOT / "artifacts/training_fixture.csv",
        delimiter=",",
        skip_header=1,
        usecols=(1, 2, 3, 4, 5, 6),
    )
    design = np.column_stack([np.ones(len(fixture)), fixture[:, :5]])
    expected, *_ = np.linalg.lstsq(design, fixture[:, 5], rcond=None)
    actual = np.r_[app.state.model.intercepts["AAPL"], app.state.model.coefficients["AAPL"]]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("field", FEATURE_NAMES)
@pytest.mark.parametrize(
    "invalid", [None, True, False, "0.01", [0.01], {"x": 0.01}, -1.00001, 1.00001]
)
def test_predict_invalid_input(client, valid_input_data, monkeypatch, field, invalid):
    spy = Mock(side_effect=AssertionError("Invalid input reached inference"))
    monkeypatch.setattr(app.state.model, "predict", spy)
    assert (
        client.post("/predict", json=dict(valid_input_data, **{field: invalid})).status_code
        == 422
    )
    spy.assert_not_called()


@pytest.mark.parametrize("field", ("symbol", *FEATURE_NAMES))
def test_missing_feature(client, valid_input_data, monkeypatch, field):
    del valid_input_data[field]
    spy = Mock()
    monkeypatch.setattr(app.state.model, "predict", spy)
    assert client.post("/predict", json=valid_input_data).status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [{}, [], [0, 0, 0, 0, 0], {"features": [0, 0, 0, 0]}, {"features": [0, 0, 0, 0, 0, 0]}],
)
def test_wrong_representation(client, monkeypatch, payload):
    spy = Mock()
    monkeypatch.setattr(app.state.model, "predict", spy)
    assert client.post("/predict", json=payload).status_code == 422
    spy.assert_not_called()


def test_extra_field(client, valid_input_data, monkeypatch):
    spy = Mock()
    monkeypatch.setattr(app.state.model, "predict", spy)
    assert (
        client.post("/predict", json=dict(valid_input_data, extra="unexpected")).status_code
        == 422
    )
    spy.assert_not_called()


@pytest.mark.parametrize("field", FEATURE_NAMES)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_input(client, valid_input_data, monkeypatch, field, value):
    spy = Mock()
    monkeypatch.setattr(app.state.model, "predict", spy)
    valid_input_data[field] = "RAW_NONFINITE"
    content = json.dumps(valid_input_data).replace('"RAW_NONFINITE"', value)
    assert (
        client.post(
            "/predict", content=content, headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )
    spy.assert_not_called()


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_inclusive_bounds(client, value):
    response = client.post(
        "/predict", json=dict(symbol="AAPL", **dict.fromkeys(FEATURE_NAMES, value))
    )
    assert response.status_code == 200
    assert np.isfinite(response.json()["prediction"])


def test_feature_order_and_effect(client, valid_input_data):
    baseline = client.post("/predict", json=valid_input_data).json()["prediction"]
    reverse = dict(reversed(list(valid_input_data.items())))
    assert client.post("/predict", json=reverse).json()["prediction"] == baseline
    for i, field in enumerate(FEATURE_NAMES):
        changed = dict(valid_input_data, **{field: valid_input_data[field] + 0.01})
        response = client.post("/predict", json=changed)
        assert response.status_code == 200
        np.testing.assert_allclose(
            response.json()["prediction"] - baseline,
            0.01 * app.state.model.coefficients["AAPL"][i],
            rtol=1e-9,
            atol=1e-12,
        )


def test_predict_model_failure(client, valid_input_data, monkeypatch):
    monkeypatch.setattr(
        app.state.model, "predict", Mock(side_effect=RuntimeError("private detail"))
    )
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert "private detail" not in response.text


def test_invalid_model_output_is_error(client, valid_input_data, monkeypatch):
    result = dict(REFERENCE["expected"], prediction=float("nan"))
    monkeypatch.setattr(app.state.model, "predict", lambda *_: result)
    assert client.post("/predict", json=valid_input_data).status_code == 500


def test_requests_are_independent(client, valid_input_data):
    coefficients = app.state.model.coefficients["AAPL"].copy()
    expected = client.post("/predict", json=valid_input_data).json()
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(
            pool.map(lambda _: client.post("/predict", json=valid_input_data).json(), range(8))
        )
    assert all(reply == expected for reply in replies)
    client.post("/predict", json=dict(valid_input_data, **dict.fromkeys(FEATURE_NAMES, -0.4)))
    assert client.post("/predict", json=valid_input_data).json() == expected
    np.testing.assert_array_equal(app.state.model.coefficients["AAPL"], coefficients)


def test_model_loads_once_per_startup(monkeypatch):
    real_load = FrozenFF5.load
    loader = Mock(side_effect=real_load)
    monkeypatch.setattr(FrozenFF5, "load", loader)
    for _ in range(2):
        with TestClient(app) as client:
            for _ in range(2):
                assert client.post("/predict", json=REFERENCE["input"]).status_code == 200
    assert loader.call_count == 2


def test_corrupt_artifact_rejected(tmp_path):
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    (tmp_path / "artifacts/ff5.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        FrozenFF5.load(tmp_path / "artifacts")


def test_version_mismatch_rejected(tmp_path):
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    path = tmp_path / "artifacts/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["library_versions"]["numpy"] = "0.0.0"
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="version mismatch"):
        FrozenFF5.load(tmp_path / "artifacts")


def test_predict_performance(client, valid_input_data):
    for _ in range(5):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    processor = platform.processor()
    if platform.system() == "Darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    report = {
        "method": "in-process Starlette TestClient including JSON/validation/inference, no network",
        "warmup_requests": 5,
        "requests": len(timings),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "hardware": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "python": platform.python_version(),
        "budget": "Measured and reported without a universal latency budget.",
    }
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence/latency.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize(
    "value", [None, True, 123, "", "aapl", " AAPL", "AAPL ", "X" * 33, "BAD_SYMBOL", ["AAPL"]]
)
def test_invalid_symbol(client, valid_input_data, monkeypatch, value):
    spy = Mock()
    monkeypatch.setattr(app.state.model, "predict", spy)
    assert client.post("/predict", json=dict(valid_input_data, symbol=value)).status_code == 422
    spy.assert_not_called()


def test_unknown_symbol(client, valid_input_data):
    assert "ZZZZZZZ" not in app.state.model.universe
    response = client.post("/predict", json=dict(valid_input_data, symbol="ZZZZZZZ"))
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "SYMBOL_OUTSIDE_UNIVERSE"
    assert response.json()["detail"]["snapshot_date"] == "2015-10-30"


def test_entire_snapshot_routes_and_matches_original_hw4(client, valid_input_data):
    # Every dated member is checked against independently captured HW4 scores.
    membership = json.loads((ROOT / "artifacts/membership.json").read_text())
    scores = REFERENCE["expected_predictions"]
    observed = set()
    unavailable_reasons = set()
    for row in membership["members"]:
        symbol = row["symbol"]
        response = client.post("/predict", json=dict(valid_input_data, symbol=symbol))
        if symbol in scores:
            assert row["eligible"] and row["model_available"]
            assert response.status_code == 200, symbol
            body = response.json()
            assert body["symbol"] == symbol
            assert body["model_version"] == REFERENCE["expected"]["model_version"]
            np.testing.assert_allclose(
                body["prediction"], scores[symbol], **REFERENCE["tolerance"]
            )
            observed.add(symbol)
        else:
            assert not row["model_available"]
            assert response.status_code == 404, symbol
            detail = response.json()["detail"]
            assert detail["code"] == "MODEL_UNAVAILABLE"
            assert detail["reason"] == row["unavailable_reason"]
            unavailable_reasons.add(detail["reason"])
    assert observed == set(scores) == set(app.state.model.coefficients)
    assert len(observed) > 1
    assert unavailable_reasons == {"STRATEGY_INELIGIBLE", "NO_VALID_REGRESSION"}


def test_changed_ticker_uses_its_own_model(client, valid_input_data):
    scores = REFERENCE["expected_predictions"]
    other = next(
        s for s in sorted(scores) if s != "AAPL" and not np.isclose(scores[s], scores["AAPL"])
    )
    result = client.post("/predict", json=dict(valid_input_data, symbol=other))
    assert result.status_code == 200
    np.testing.assert_allclose(
        result.json()["prediction"], scores[other], **REFERENCE["tolerance"]
    )
    assert not np.isclose(result.json()["prediction"], scores["AAPL"])


def test_membership_corruption_rejected(tmp_path):
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    (tmp_path / "artifacts/membership.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        FrozenFF5.load(tmp_path / "artifacts")


def test_models_outside_universe_rejected():
    payload = json.loads((ROOT / "artifacts/ff5.json").read_text())
    membership = json.loads((ROOT / "artifacts/membership.json").read_text())
    payload["models"]["ZZZZZZZ"] = payload["models"]["AAPL"]
    with pytest.raises(ValueError, match="eligibility"):
        FrozenFF5(payload, membership)
