"""Tests for the HW5 service that serves my HW4 scoring model.

Reference results come from HW4 itself: export_hw4_model.py rebuilt the HW4
score matrix with the unchanged HW4 functions (hw4.combine_weighted) and saved
every candidate's score on 2026-09-30 to artifacts/hw4_reference_scores.csv
*before* any API code existed. HW4 computes in float32, so those comparisons use
abs=1e-5; API-vs-direct-pipeline comparisons use the same float64 code path and
must agree to 1e-12.
"""

import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import InputData, app, to_model_frame
from hw4_model import FEATURES, expected_excess_return

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "hw4_reference_scores.csv"
HW4_TOL = 1e-5  # HW4 scored in float32
API_TOL = 1e-12  # same float64 pipeline, only JSON in between


# ---------------------------------------------------------------- fixtures --
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
    """Inputs and scores recorded from the HW4 code (one row per candidate)."""
    return pd.read_csv(REFERENCE, index_col="symbol")


def payload_from(row) -> dict:
    return {f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURES}


@pytest.fixture
def valid_input_data():
    # Same content as example_request.json: HW4's top-ranked stock on 2026-09-30.
    return json.loads((ROOT / "example_request.json").read_text())


@pytest.fixture
def model_must_not_run(monkeypatch):
    """Make any call into the model fail the test (used for invalid input)."""

    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)


# ------------------------------------------------------ 1. valid prediction --
def test_predict_valid_input(client, valid_input_data, reference):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "prediction",
        "is_candidate",
        "expected_excess_return_90d",
        "model_version",
    }
    assert isinstance(body["prediction"], float) and math.isfinite(body["prediction"])
    assert body["model_version"] == app.state.bundle["model_version"]
    assert body["is_candidate"] is True
    assert response.headers["X-Request-ID"]

    # API == direct pipeline call on the same training-format DataFrame.
    frame = to_model_frame(InputData(**valid_input_data))
    direct = app.state.bundle["pipeline"].predict(frame)[0]
    assert body["prediction"] == pytest.approx(direct, abs=API_TOL)

    # API == score recorded from HW4 before the API existed (top-ranked stock).
    top = reference.sort_values("hw4_score", ascending=False).iloc[0]
    # HW4 features are float32; the CSV keeps 9 significant digits (exact in float32).
    as_f32 = {k: None if v is None else np.float32(v) for k, v in valid_input_data.items()}
    assert {
        k: None if v is None else np.float32(v) for k, v in payload_from(top).items()
    } == as_f32
    assert body["prediction"] == pytest.approx(top["hw4_score"], abs=HW4_TOL)


@pytest.mark.parametrize("rank_position", [0, 1, 5, 25, 60, -1])
def test_api_matches_hw4_reference_rows(client, reference, rank_position):
    """Different signal mixes (one, two or more flags) across the ranking."""
    row = reference.sort_values("hw4_score", ascending=False).iloc[rank_position]
    response = client.post("/predict", json=payload_from(row))
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == pytest.approx(row["hw4_score"], abs=HW4_TOL)
    assert body["expected_excess_return_90d"] == pytest.approx(
        row["hw4_expected_excess_return_90d"], abs=HW4_TOL
    )


def test_pipeline_reproduces_every_hw4_candidate(client, reference):
    """Direct call on all candidates at once: same scores and same ranking as HW4."""
    scores = app.state.bundle["pipeline"].predict(reference[FEATURES])
    assert scores == pytest.approx(reference["hw4_score"].to_numpy(), abs=HW4_TOL)
    ranks = pd.Series(scores).rank(ascending=False, method="min").astype(int)
    assert (ranks.to_numpy() == reference["hw4_rank"].to_numpy()).all()


def test_row_alone_equals_row_in_batch(client, reference):
    """Fitted preprocessing is frozen: a row's score must not depend on its batch."""
    pipeline = app.state.bundle["pipeline"]
    batch = reference[FEATURES]
    for i in (0, 10, len(batch) - 1):
        alone = pipeline.predict(batch.iloc[[i]])[0]
        assert alone == pytest.approx(pipeline.predict(batch)[i], abs=API_TOL)


def test_no_signal_is_not_a_candidate(client):
    response = client.post("/predict", json={f: None for f in FEATURES})
    assert response.status_code == 200
    assert response.json()["prediction"] == 0.0
    assert response.json()["is_candidate"] is False
    assert response.json()["expected_excess_return_90d"] is None


def test_expected_return_is_hw4_signal_average(client, reference):
    """Every candidate: pipeline value == HW4-array value (computed in the export
    without the pipeline); MU by hand from model_metadata.json:
    (w_fcf*mu_fcf + w_mom*mu_mom) / (w_fcf + w_mom), both signals flag MU."""
    pipeline = app.state.bundle["pipeline"]
    values = expected_excess_return(pipeline, reference[FEATURES])
    assert values == pytest.approx(
        reference["hw4_expected_excess_return_90d"].to_numpy(), abs=HW4_TOL
    )
    meta = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    w, mu = meta["weights_trailing_126d"], meta["expected_90d_excess_return_trailing_126d"]
    f, m = "fcf_quality_score", "momentum_percentile"
    by_hand = (w[f] * mu[f] + w[m] * mu[m]) / (w[f] + w[m])
    mu_row = reference.loc[["MU"], FEATURES]
    assert expected_excess_return(pipeline, mu_row)[0] == pytest.approx(by_hand, rel=1e-9)


def test_expected_return_single_signal_equals_its_mu_hat(client):
    """Momentum-only stock -> exactly the momentum signal's trailing mean."""
    meta = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
    payload = {f: None for f in FEATURES} | {"momentum_percentile": 0.95}
    body = client.post("/predict", json=payload).json()
    expected = meta["expected_90d_excess_return_trailing_126d"]["momentum_percentile"]
    assert body["expected_excess_return_90d"] == pytest.approx(expected, rel=1e-9)


def test_health_reports_loaded_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ready"
    assert body["model_version"] == app.state.bundle["model_version"]
    assert body["snapshot_date"] == "2026-09-30"


# --------------------------------------------------------- 2. invalid input --
def _with(base: dict, **changes) -> dict:
    out = dict(base)
    out.update(changes)
    return out


VALID = {
    "pairs_divergence_z": None,
    "fcf_quality_score": 0.8,
    "revenue_growth_vs_industry": 0.1,
    "momentum_percentile": 0.95,
}
INVALID_PAYLOADS = {
    "missing field": {k: v for k, v in VALID.items() if k != "momentum_percentile"},
    "empty object": {},
    "extra field": _with(VALID, volume=1.0),
    "five features": _with(VALID, rsi_oversold=0.4),
    "list, wrong count (3)": [0.8, 0.1, 0.95],
    "list, right count (4)": [None, 0.8, 0.1, 0.95],
    "numeric string": _with(VALID, fcf_quality_score="0.8"),
    "boolean": _with(VALID, fcf_quality_score=True),
    "object as number": _with(VALID, momentum_percentile={"value": 0.95}),
    "below flag threshold (pairs)": _with(VALID, pairs_divergence_z=1.2),
    "negative z (sign not allowed)": _with(VALID, pairs_divergence_z=-2.0),
    "percent instead of fraction": _with(VALID, momentum_percentile=97.0),
    "not in momentum top 10%": _with(VALID, momentum_percentile=0.5),
    "fcf score zero": _with(VALID, fcf_quality_score=0.0),
    "fcf score above 1": _with(VALID, fcf_quality_score=1.5),
    "revenue gap below offset": _with(VALID, revenue_growth_vs_industry=0.0),
    "revenue gap absurd": _with(VALID, revenue_growth_vs_industry=50.0),
    "pairs z absurd": _with(VALID, pairs_divergence_z=1e6),
    "null body": None,
}


@pytest.mark.parametrize("payload", INVALID_PAYLOADS.values(), ids=INVALID_PAYLOADS.keys())
def test_predict_invalid_input(client, model_must_not_run, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_rejected(client, model_must_not_run, token):
    # Python's json parser accepts these non-standard tokens, so check them explicitly.
    body = json.dumps(VALID).replace("0.95", token)
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    # Regression: FastAPI's default 422 body echoes the input, and encoding NaN/inf
    # crashed it into a 500. app.py's handler omits the rejected value.
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "momentum_percentile"]


def test_malformed_json_rejected(client, model_must_not_run):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


# ----------------------------------------------------- 3. inference failure --
def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}  # my 500 contract
    assert "simulated" not in response.text  # details stay in the server log
    assert response.headers["X-Request-ID"]


def test_non_finite_model_output_is_an_error(failure_client, valid_input_data, monkeypatch):
    """A NaN from the model must not be returned as a successful prediction."""
    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", lambda X: [float("nan")])
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500


# ----------------------------------------------------------- 4. performance --
WARMUP, N_REQUESTS = 10, 200
# Budget: in-process TestClient (no network) on an Apple M5 measures ~1 ms per
# request. 50 ms p95 leaves ~50x headroom for slower laptops/CI runners while
# still catching a regression such as reloading the model or refitting per request.
P95_BUDGET_MS = 50.0


def test_predict_performance(client, valid_input_data):
    for _ in range(WARMUP):
        assert client.post("/predict", json=valid_input_data).status_code == 200
    latencies_ms = []
    for _ in range(N_REQUESTS):
        start = time.perf_counter()
        response = client.post("/predict", json=valid_input_data)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    median = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=100, method="inclusive")[94]
    print(
        f"\nLatency (in-process TestClient, {N_REQUESTS} requests after {WARMUP} warm-up, "
        f"{platform.machine()} {platform.system()} Python {platform.python_version()}): "
        f"median {median:.2f} ms, p95 {p95:.2f} ms, max {max(latencies_ms):.2f} ms"
    )
    assert p95 < P95_BUDGET_MS
