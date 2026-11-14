"""Tests for the blend-score API: real model, rejected bodies, failures, timing."""

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import InputData, app

TESTS_DIR = Path(__file__).parent
SUBMISSION_DIR = TESTS_DIR.parent
SAVED = json.loads((TESTS_DIR / "frozen_reference.json").read_text())
JSON_HEADERS = {"Content-Type": "application/json"}


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
    return json.loads((SUBMISSION_DIR / "example_request.json").read_text())


@pytest.fixture
def model_calls(client, monkeypatch):
    """Swap in a predict that only counts. Any request reaching it bumps the count."""
    tally = {"n": 0}

    def counting_predict(frame):
        tally["n"] += 1
        return np.array([0.5])

    monkeypatch.setattr(app.state.pipeline, "predict", counting_predict)
    return tally


def send_raw(client, text):
    # raw text, so NaN and Infinity reach the server exactly as typed
    return client.post("/predict", content=text, headers=JSON_HEADERS)


def frame_for(payload):
    return pd.DataFrame([payload], columns=app.state.columns, dtype="float64")


def reference_day():
    # round_trip parsing matters: the default parser can be a hair off, and a
    # hair is enough to split two equal values and move their percentiles
    return pd.read_csv(TESTS_DIR / "sep25_cross_section.csv", float_precision="round_trip")


# 1. valid prediction -----------------------------------------------------------


def test_predict_valid_input(client, valid_input_data):
    reply = client.post("/predict", json=valid_input_data)
    assert reply.status_code == 200
    body = reply.json()
    assert set(body) == {"prediction", "model_version"}
    assert body["model_version"] == SAVED["model_version"]

    direct = float(app.state.pipeline.predict(frame_for(valid_input_data))[0])
    assert body["prediction"] == pytest.approx(direct, rel=1e-12)
    # INCY on Sep 25 2026, scored by HW4's own code before this API existed
    assert valid_input_data == SAVED["inputs"]
    assert body["prediction"] == pytest.approx(SAVED["hw4_score"], abs=1e-9)


def test_whole_reference_day_matches_hw4(client):
    """All 487 tradable stocks of the reference date, scored in one batch, must
    equal the blend scores HW4 computed for them."""
    day = reference_day()
    scores = app.state.pipeline.predict(day[app.state.columns])
    assert len(day) == 487
    assert np.abs(scores - day.hw4_score.to_numpy()).max() < 1e-12


def test_contract_accepts_every_real_row():
    # the bounds must not reject any stock HW4 actually traded on that date
    day = reference_day()
    for record in day[list(InputData.model_fields)].to_dict("records"):
        InputData(**{k: (None if pd.isna(v) else float(v)) for k, v in record.items()})


@pytest.mark.parametrize("ratio", ["operating_roa", "net_margin"])
def test_blank_ratio_counts_as_middle(client, valid_input_data, ratio):
    payload = {**valid_input_data, ratio: None}
    reply = client.post("/predict", json=payload)
    assert reply.status_code == 200
    # worked by hand: the other five percentiles are unchanged, the blank one is 0.5
    pct = app.state.pipeline.named_steps["pct"].transform(frame_for(valid_input_data))[0]
    others = [p for name, p in zip(app.state.columns, pct) if name != ratio]
    assert reply.json()["prediction"] == pytest.approx((sum(others) + 0.5) / 6, abs=1e-12)


def test_whole_numbers_and_shuffled_keys_are_fine(client, valid_input_data):
    payload = dict(reversed(list(valid_input_data.items())))
    payload["drawdown_252"] = 0  # a stock sitting right at its yearly high
    reply = client.post("/predict", json=payload)
    assert reply.status_code == 200
    expected = app.state.pipeline.predict(frame_for(payload))[0]
    assert reply.json()["prediction"] == pytest.approx(expected, abs=1e-12)


# 2. input validation -----------------------------------------------------------


def one_change(field, value):
    """A copy of the example request with one field changed, or removed if value is ...."""

    def make(base):
        out = dict(base)
        if value is ...:
            out.pop(field)
        else:
            out[field] = value
        return json.dumps(out)

    return make


REJECTED = [
    *[pytest.param(one_change(f, ...), id=f"missing {f}") for f in InputData.model_fields],
    pytest.param(lambda base: json.dumps({**base, "beta_252": 1.1}), id="seventh field"),
    pytest.param(one_change("trend_6m", "0.3"), id="number as text"),
    pytest.param(one_change("volatility_63", True), id="boolean"),
    pytest.param(one_change("momentum_252_21", None), id="null price factor"),
    pytest.param(one_change("volatility_63", [0.3]), id="list in a field"),
    pytest.param(lambda base: '{"momentum_252_21": NaN}', id="NaN token"),
    pytest.param(
        lambda base: json.dumps(base).replace(f"{base['net_margin']}", "Infinity"),
        id="Infinity token",
    ),
    pytest.param(
        lambda base: json.dumps(base).replace(f"{base['trend_6m']}", "-Infinity"),
        id="-Infinity token",
    ),
    pytest.param(one_change("momentum_252_21", -1), id="return of -100%"),
    pytest.param(one_change("momentum_252_21", 51), id="12-1 momentum above 50x"),
    pytest.param(one_change("trend_6m", 30), id="6m return above 25x"),
    pytest.param(one_change("volatility_63", -0.01), id="negative volatility"),
    pytest.param(one_change("volatility_63", 12), id="volatility above 10"),
    pytest.param(one_change("drawdown_252", 0.02), id="price above its high"),
    pytest.param(one_change("drawdown_252", -1.5), id="drawdown below -100%"),
    pytest.param(one_change("operating_roa", 2e5), id="ratio past sanity cap"),
    pytest.param(lambda base: json.dumps(list(base.values())), id="bare list of six"),
    pytest.param(lambda base: json.dumps(list(base.values())[:5]), id="bare list of five"),
    pytest.param(lambda base: "{}", id="empty object"),
    pytest.param(lambda base: "", id="empty body"),
]


@pytest.mark.parametrize("build_body", REJECTED)
def test_predict_invalid_input(client, valid_input_data, model_calls, build_body):
    reply = send_raw(client, build_body(valid_input_data))
    assert reply.status_code == 422
    assert reply.json()["detail"]
    assert model_calls["n"] == 0, "a rejected body reached the model"


def test_counter_does_see_valid_requests(client, valid_input_data, model_calls):
    # proves the counter works, so the zeros above mean something
    assert client.post("/predict", json=valid_input_data).status_code == 200
    assert model_calls["n"] == 1


# 3. inference failure ----------------------------------------------------------


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch):
    def exploding_predict(frame):
        raise ValueError("forced for the 500 test")

    monkeypatch.setattr(app.state.pipeline, "predict", exploding_predict)
    reply = failure_client.post("/predict", json=valid_input_data)
    assert reply.status_code == 500
    assert "forced for the 500 test" not in reply.text
    assert "prediction" not in reply.text


def test_infinite_score_is_an_error(failure_client, valid_input_data, monkeypatch):
    monkeypatch.setattr(app.state.pipeline, "predict", lambda frame: np.array([np.inf]))
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


# 4. performance ----------------------------------------------------------------


def test_predict_performance(client, valid_input_data):
    for _ in range(15):
        assert client.post("/predict", json=valid_input_data).status_code == 200

    ms = []
    for _ in range(40):
        t0 = time.perf_counter()
        reply = client.post("/predict", json=valid_input_data)
        ms.append((time.perf_counter() - t0) * 1000)
        assert reply.status_code == 200

    ordered = sorted(ms)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]  # nearest-rank p95
    assert all(t > 0 for t in ms)
    print(
        f"\n[latency] in-process TestClient, 15 warm-up + {len(ms)} timed requests: "
        f"median {statistics.median(ms):.2f} ms, p95 {p95:.2f} ms | "
        f"{platform.platform()}, {platform.processor()}, {os.cpu_count()} logical CPUs, "
        f"Python {platform.python_version()}"
    )
