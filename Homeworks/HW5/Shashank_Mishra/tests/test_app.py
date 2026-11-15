"""Tests for the HW5 service, using the real saved HW4 model except in the failure tests.

Reference predictions (artifacts/reference_predictions.json) were recorded by
export_hw4_model.py: it refit HW4's lgb_excess model with HW4's own code
(run_hw4.FITTED, hw4.model.training_rows) and called HW4's RankLGBM.predict on
real stocks from the 2026-04-09 signal date, before any API code existed. That
ran in the HW4 environment (Python 3.13); these tests run in the Python 3.12
serving environment, so they also check the artifact survives that move.
"""

import json
import logging
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_service
from app import app

ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
REFERENCE = json.loads((ROOT / "artifacts" / "reference_predictions.json").read_text())

# LightGBM sums the same 300 leaf values in float64 on every platform; predictions are
# ~1e-2, so 1e-12 absolute leaves room only for float rounding, not for a different model.
TOLERANCE = 1e-12


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
    # The committed example request: stock "A" on 2026-04-09 (first reference case).
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def model_must_not_run(monkeypatch):
    """Make any call to the loaded model fail the test (used for invalid input)."""

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    def install():
        monkeypatch.setattr(app.state.model.estimator, "predict", unexpected_call)

    return install


# ---------------------------------------------------------------- 1. valid prediction
def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == METADATA["model_version"] == "hw4-lgb_excess-fit-2026-04-09"
    assert response.headers["X-Request-ID"]

    # API == direct call of the loaded estimator on the training representation.
    bundle = app.state.model
    direct = bundle.estimator.predict(model_service.to_frame(bundle, valid_input_data))[0]
    assert body["prediction"] == pytest.approx(direct, abs=TOLERANCE)

    # API == the value HW4's own code produced for this stock before the API existed.
    assert REFERENCE[0]["features"] == valid_input_data
    assert body["prediction"] == pytest.approx(REFERENCE[0]["hw4_prediction"], abs=TOLERANCE)


@pytest.mark.parametrize("case", REFERENCE, ids=[c["symbol"] for c in REFERENCE])
def test_api_reproduces_hw4_reference_predictions(client, case):
    # Includes stocks with missing (null) fundamentals, which LightGBM saw as NaN in HW4.
    response = client.post("/predict", json=case["features"])
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(case["hw4_prediction"], abs=TOLERANCE)


def test_reference_cases_cover_missing_values():
    assert any(None in c["features"].values() for c in REFERENCE)


def test_prediction_does_not_depend_on_other_rows(client):
    # No learned state is updated per request: a row scores the same alone or in a batch.
    bundle = app.state.model
    frames = [model_service.to_frame(bundle, c["features"]) for c in REFERENCE]
    batch = bundle.estimator.predict(pd.concat(frames, ignore_index=True))
    alone = [bundle.estimator.predict(f)[0] for f in frames]
    assert np.allclose(batch, alone, atol=TOLERANCE, rtol=0)


def test_health_reports_loaded_version(client):
    assert client.get("/health").json() == {
        "status": "ready",
        "model_version": METADATA["model_version"],
    }


def test_feature_order_matches_artifact(client):
    bundle = app.state.model
    assert bundle.features == model_service.FEATURES == tuple(METADATA["features_in_order"])
    assert list(bundle.estimator.feature_name_) == [f"r_{f}" for f in model_service.FEATURES]


def test_model_loaded_once_per_process(monkeypatch, valid_input_data):
    # Reloading is only ~2 ms, so the latency test would not notice it; count loads instead.
    real_load, calls = model_service.joblib.load, []

    def counting_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(model_service.joblib, "load", counting_load)
    with TestClient(app) as test_client:
        for _ in range(5):
            assert test_client.post("/predict", json=valid_input_data).status_code == 200
    assert calls == [model_service.ARTIFACT_PATH]


def test_startup_refuses_mismatched_library_version(monkeypatch):
    real_load = model_service.joblib.load

    def load_with_other_version(path):
        raw = real_load(path)
        raw["versions"] = {**raw["versions"], "lightgbm": "0.0.1"}
        return raw

    monkeypatch.setattr(model_service.joblib, "load", load_with_other_version)
    with pytest.raises(RuntimeError, match="lightgbm"):
        model_service.load_bundle()


# ---------------------------------------------------------------- 2. input validation
def _example():
    return dict(REFERENCE[0]["features"])


def _without(key):
    payload = _example()
    del payload[key]
    return payload


def _with(**changes):
    return {**_example(), **changes}


INVALID_PAYLOADS = {
    "missing_field": _without("mom_12_1"),
    "eleven_features": _without("beta_252"),
    "thirteen_features_extra_field": _with(log_mcap=0.5),
    "empty_object": {},
    "list_instead_of_object": list(_example().values()),
    "numeric_string": _with(mom_12_1="0.5"),
    "boolean": _with(mom_12_1=True),
    "null_for_never_missing_rev_1m": _with(rev_1m=None),
    "zero_is_not_a_percentile": _with(rev_1m=0),
    "above_one": _with(earnings_yield=1.5),
    "negative": _with(beta_252=-0.2),
    "raw_return_instead_of_percentile": _with(mom_12_1=35.0),
    "nested_object": _with(accruals={"value": 0.5}),
}


@pytest.mark.parametrize("payload", INVALID_PAYLOADS.values(), ids=INVALID_PAYLOADS.keys())
def test_predict_invalid_input(client, model_must_not_run, payload):
    model_must_not_run()
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_input_rejected(client, model_must_not_run, value):
    # Standard JSON has no NaN, but Python's parser accepts these tokens, so send raw text.
    model_must_not_run()
    text = json.dumps(_example()).replace(str(_example()["rev_1m"]), value)
    assert value in text
    response = client.post(
        "/predict", content=text, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "rev_1m"]


def test_malformed_json_rejected(client, model_must_not_run):
    model_must_not_run()
    response = client.post(
        "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_explicit_null_is_valid_for_nullable_feature(client):
    response = client.post("/predict", json=_with(earnings_yield=None))
    assert response.status_code == 200


def test_integer_one_is_a_valid_percentile(client):
    # The top-ranked stock has percentile exactly 1; JSON may write it as 1.
    assert client.post("/predict", json=_with(mom_12_1=1)).status_code == 200


# ---------------------------------------------------------------- 3. inference failure
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure: secret internal detail")

    monkeypatch.setattr(app.state.model.estimator, "predict", fail)
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        response = failure_client.post("/predict", json=valid_input_data)

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}  # my app's 500 contract
    assert "secret internal detail" not in response.text  # details stay in the log
    request_id = response.headers["X-Request-ID"]
    assert f"inference_failed request_id={request_id}" in caplog.text
    assert "secret internal detail" in caplog.text


def test_non_finite_model_output_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(
        app.state.model.estimator, "predict", lambda *a, **k: np.array([float("nan")])
    )
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


# ---------------------------------------------------------------- 4. performance
WARMUP, MEASURED = 10, 200
# Budget for in-process TestClient on a laptop. Measured p95 is ~3-4 ms (README); 50 ms
# leaves >10x headroom for a slower machine or CI runner while still catching gross
# regressions such as refitting per request (seconds). Per-request reloading is only
# ~2 ms, so test_model_loaded_once_per_process guards that instead.
P95_BUDGET_MS = 50.0


def test_predict_performance(client, valid_input_data, capsys):
    for _ in range(WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    latencies_ms = []
    for _ in range(MEASURED):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    median = statistics.median(latencies_ms)
    p95 = float(np.percentile(latencies_ms, 95))
    with capsys.disabled():
        print(
            f"\n[latency] in-process TestClient, {MEASURED} sequential requests after "
            f"{WARMUP} warm-up: median {median:.2f} ms, p95 {p95:.2f} ms, "
            f"max {max(latencies_ms):.2f} ms | {platform.processor() or platform.machine()}, "
            f"{platform.system()} {platform.release()}, Python {platform.python_version()}"
        )
    assert p95 < P95_BUDGET_MS
