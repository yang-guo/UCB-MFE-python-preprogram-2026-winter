"""Tests for the HW4 model service.  Run: ``python -m pytest -q`` (add ``-s`` to see latency)."""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import InputData, app, to_frame

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = json.loads((ROOT / "tests" / "reference_examples.json").read_text())
METADATA = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())

# HW4 wrote its scores to CSV with pandas' default float formatting (~8 significant
# digits; the re-created model matched them to 3.7e-9).  1e-6 absolute is far below
# any meaningful difference in a rank score that spans roughly [-0.1, 0.1].
HW4_TOLERANCE = 1e-6


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
    # Same payload as example_request.json: HW4's top-ranked stock on 2026-09-25.
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def model_must_not_run(client, monkeypatch):
    """Make any call into the model fail the test (proves rejection happens first)."""

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)
    return client


# --------------------------------------------------------------------------- 1. valid input
@pytest.mark.parametrize("reference", REFERENCES, ids=[r["symbol"] for r in REFERENCES])
def test_predict_valid_input(client, reference):
    """API == direct pipeline call == score HW4 recorded for the same stock and date.

    Reference results come from HW4's final run (outputs/test_predictions.csv.gz), saved
    by export_model.py before the model was wrapped in this API.  They include the model's
    top pick, a stock with missing (null) features, and a typical stock.
    """
    response = client.post("/predict", json=reference["request"])
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "model_version"}
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == METADATA["model_version"]
    assert response.headers["X-Request-ID"]

    direct = app.state.bundle["pipeline"].predict(to_frame(InputData(**reference["request"])))
    assert body["prediction"] == pytest.approx(float(direct[0]), abs=1e-12)
    assert body["prediction"] == pytest.approx(
        reference["hw4_recorded_score"], abs=HW4_TOLERANCE
    )


def test_valid_example_is_the_documented_request(valid_input_data):
    assert valid_input_data == REFERENCES[0]["request"]


def test_integer_json_numbers_are_accepted(client, valid_input_data):
    # 0 is a valid JSON number for a rank; strict mode accepts int -> float but not str/bool.
    payload = valid_input_data | {"mom_5": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    expected = client.post("/predict", json=valid_input_data | {"mom_5": 0.0}).json()
    assert response.json()["prediction"] == expected["prediction"]


def test_model_loaded_once_per_process(client, valid_input_data):
    bundle = app.state.bundle
    for _ in range(3):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    assert app.state.bundle is bundle
    assert client.get("/health").json() == {
        "status": "ready",
        "model_version": METADATA["model_version"],
    }


# --------------------------------------------------------------------------- 2. invalid input
def _without(payload, key):
    return {k: v for k, v in payload.items() if k != key}


INVALID_CASES = {
    "missing numeric field": lambda p: _without(p, "mom_5"),
    "missing sector": lambda p: _without(p, "sector"),
    "extra field": lambda p: p | {"surprise": 0.1},
    "23rd numeric feature": lambda p: p | {"mom_10": 0.0},
    "numeric string": lambda p: p | {"mom_5": "0.1"},
    "boolean": lambda p: p | {"mom_5": True},
    "above rank range": lambda p: p | {"vol_63": 0.51},
    "below rank range": lambda p: p | {"vol_63": -0.51},
    "raw (unranked) value": lambda p: p | {"log_mcap": 23.4},
    "unknown sector": lambda p: p | {"sector": "Unknown"},
    "sector as number": lambda p: p | {"sector": 3},
    "null sector": lambda p: p | {"sector": None},
    "list instead of number": lambda p: p | {"mom_5": [0.1]},
    "fixed-width list, 21 values": lambda p: {"features": [0.0] * 21, "sector": "Energy"},
    "fixed-width list, 22 values": lambda p: {"features": [0.0] * 22, "sector": "Energy"},
    "empty object": lambda p: {},
    "top-level list": lambda p: [p],
}


@pytest.mark.parametrize("case", INVALID_CASES, ids=list(INVALID_CASES))
def test_predict_invalid_input(model_must_not_run, valid_input_data, case):
    payload = INVALID_CASES[case](dict(valid_input_data))
    response = model_must_not_run.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_rejected(model_must_not_run, valid_input_data, token):
    # Python's json module writes NaN/Infinity tokens; the service must say 422, not 500.
    body = json.dumps(valid_input_data | {"beta_252": float(token)})
    assert token in body
    response = model_must_not_run.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "beta_252"]


def test_malformed_json_rejected(model_must_not_run):
    response = model_must_not_run.post(
        "/predict", content=b'{"mom_5": ', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- 3. failures
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure: secret internal detail")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    with caplog.at_level("ERROR", logger="uvicorn.error"):
        response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert "secret internal detail" not in response.text  # details stay in the log
    request_id = response.headers["X-Request-ID"]
    assert any(request_id in record.getMessage() for record in caplog.records)


def test_non_finite_model_output_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", lambda X: [float("nan")])
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


# --------------------------------------------------------------------------- 4. latency
# In-process FastAPI TestClient (no network socket), so this measures validation +
# DataFrame construction + sklearn pipeline + response serialization.  Real localhost
# HTTP latency through Docker is reported separately in the README.
WARMUP, MEASURED = 10, 200
# Budget: p95 < 100 ms.  Measured in-process p95 is ~3 ms on the Apple M3 used, so this
# leaves ~30x headroom for slower CI machines while still catching gross regressions
# (refitting per request, database calls).  A per-request reload of this tiny artifact
# (~0.1 ms) would not show up here; test_model_loaded_once_per_process covers that.
P95_BUDGET_MS = 100.0


def test_predict_performance(client, valid_input_data):
    for _ in range(WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    timings_ms = []
    for _ in range(MEASURED):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        timings_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    timings_ms.sort()
    median = statistics.median(timings_ms)
    p95 = timings_ms[math.ceil(0.95 * len(timings_ms)) - 1]
    print(
        f"\n[latency] in-process TestClient, {MEASURED} requests after {WARMUP} warm-up, "
        f"{platform.machine()} {platform.system()} Python {platform.python_version()}: "
        f"median {median:.2f} ms, p95 {p95:.2f} ms, max {timings_ms[-1]:.2f} ms"
    )
    assert p95 < P95_BUDGET_MS
