"""Meaningful HW5 tests, adapted from the starter's four required groups."""

import json
import math
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as application
import serving
from measurements import measure_requests
from schemas import InputData, OutputData
from serving import FEATURES, ROOT, feature_frame

REFERENCE = pd.read_csv(ROOT / "artifacts/fixed_anchor_2012_reference.csv")
EXAMPLE = json.loads((ROOT / "artifacts/example_reference.json").read_text())


@pytest.fixture
def client():
    with TestClient(application.app) as test_client:
        yield test_client


@pytest.fixture
def failure_client():
    with TestClient(application.app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def valid_input_data():
    return EXAMPLE["input"].copy()


def test_predict_valid_input(client, valid_input_data):
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    output = OutputData.model_validate(response.json())
    direct = application.app.state.pipeline.predict(feature_frame(valid_input_data))[0]
    assert output.prediction == pytest.approx(direct, abs=1e-12, rel=0)
    # This expected value was recorded by HW4 before the API was written.
    assert output.prediction == pytest.approx(EXAMPLE["expected_prediction"], abs=1e-12, rel=0)
    assert output.model_version == "hw4-fixed-ridge-2012-84e526d28792-v1"
    assert set(response.json()) == {"prediction", "model_version"}
    assert response.headers["X-Request-ID"]
    assert math.isfinite(output.prediction)


def test_all_saved_hw4_examples(client):
    """Cover 209 varied rows, including genuine missing financial inputs."""
    missing_rows = 0
    for row in REFERENCE.to_dict(orient="records"):
        payload = {f: None if pd.isna(row[f]) else row[f] for f in FEATURES}
        missing_rows += any(value is None for value in payload.values())
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, (row["date"], row["symbol"], response.text)
        assert response.json()["prediction"] == pytest.approx(
            row["expected_prediction"], abs=1e-12, rel=0
        )
    assert len(REFERENCE) == 209
    assert missing_rows > 0


@pytest.mark.parametrize("feature", FEATURES)
def test_missing_field_rejected_before_inference(
    client, valid_input_data, monkeypatch, feature
):
    valid_input_data.pop(feature)
    spy = Mock(side_effect=AssertionError("Invalid request reached inference"))
    monkeypatch.setattr(application.app.state.pipeline, "predict", spy)
    assert client.post("/predict", json=valid_input_data).status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("bad_value", ["0.5", True, False, -0.1, 0, 1.01, [], {}])
def test_predict_invalid_input(client, valid_input_data, monkeypatch, feature, bad_value):
    valid_input_data[feature] = bad_value
    spy = Mock(side_effect=AssertionError("Invalid request reached inference"))
    monkeypatch.setattr(application.app.state.pipeline, "predict", spy)
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_body_is_422(client, valid_input_data, monkeypatch, feature, bad_value):
    valid_input_data[feature] = bad_value
    spy = Mock()
    monkeypatch.setattr(application.app.state.pipeline, "predict", spy)
    # httpx's json= correctly refuses nonstandard JSON, so exercise the server
    # boundary using raw text, as a malformed external client could send.
    response = client.post(
        "/predict",
        content=json.dumps(valid_input_data, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()
    spy.assert_not_called()


@pytest.mark.parametrize("payload", [{}, [], [0.5] * 7, [0.5] * 8, [0.5] * 9, None, 1])
def test_wrong_body_shape_never_reaches_model(client, monkeypatch, payload):
    spy = Mock()
    monkeypatch.setattr(application.app.state.pipeline, "predict", spy)
    response = client.post(
        "/predict", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    spy.assert_not_called()


def test_extra_feature_rejected(client, valid_input_data, monkeypatch):
    spy = Mock()
    monkeypatch.setattr(application.app.state.pipeline, "predict", spy)
    response = client.post("/predict", json={**valid_input_data, "unexpected": 0.5})
    assert response.status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("feature", FEATURES[:6])
def test_market_rank_cannot_be_null(client, valid_input_data, monkeypatch, feature):
    spy = Mock()
    monkeypatch.setattr(application.app.state.pipeline, "predict", spy)
    valid_input_data[feature] = None
    assert client.post("/predict", json=valid_input_data).status_code == 422
    spy.assert_not_called()


def test_field_order_does_not_change_prediction(client, valid_input_data):
    reversed_payload = dict(reversed(list(valid_input_data.items())))
    response = client.post("/predict", json=reversed_payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(
        EXAMPLE["expected_prediction"], abs=1e-12, rel=0
    )


def test_valid_rank_boundaries(client, valid_input_data):
    for value in [1, 1.0, 0.000001]:
        payload = {name: value for name in valid_input_data}
        assert client.post("/predict", json=payload).status_code == 200


def test_predict_model_failure(failure_client, valid_input_data, monkeypatch, caplog):
    def fail(_):
        raise RuntimeError("Internal diagnostic; not for the HTTP response")

    monkeypatch.setattr(application.app.state.pipeline, "predict", fail)
    response = failure_client.post("/predict", json=valid_input_data)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    request_id = response.headers["X-Request-ID"]
    assert request_id in caplog.text
    assert "Internal diagnostic" in caplog.text
    assert "Internal diagnostic" not in response.text


@pytest.mark.parametrize("output", [[float("nan")], [float("inf")], [], [0.1, 0.2]])
def test_bad_model_output_is_500(failure_client, valid_input_data, monkeypatch, output):
    monkeypatch.setattr(application.app.state.pipeline, "predict", lambda _: output)
    assert failure_client.post("/predict", json=valid_input_data).status_code == 500


def test_requests_cannot_refit_or_mutate_model(client, valid_input_data, monkeypatch):
    pipeline = application.app.state.pipeline
    original_state = joblib.hash(pipeline)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Serving attempted a fit")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "fit", forbidden_fit)
        for name in ("impute", "scale", "regression"):
            patch.setattr(pipeline.named_steps[name], "fit", forbidden_fit)
        assert client.post("/predict", json=valid_input_data).status_code == 200
        other = {name: 0.8 for name in FEATURES}
        assert client.post("/predict", json=other).status_code == 200
        again = client.post("/predict", json=valid_input_data)
        assert again.json()["prediction"] == pytest.approx(
            EXAMPLE["expected_prediction"], abs=1e-12, rel=0
        )
    assert joblib.hash(pipeline) == original_state


def test_lifespan_loads_once_and_cleans_up(monkeypatch):
    real_loader = application.load_pipeline
    spy = Mock(wraps=real_loader)
    monkeypatch.setattr(application, "load_pipeline", spy)
    with TestClient(application.app) as client:
        for _ in range(3):
            assert client.get("/health").status_code == 200
            assert client.post("/predict", json=EXAMPLE["input"]).status_code == 200
        assert spy.call_count == 1
    assert not hasattr(application.app.state, "pipeline")


def test_wrong_artifact_stops_startup(tmp_path, monkeypatch):
    wrong = tmp_path / "wrong.joblib"
    wrong.write_bytes(b"not the trusted artifact")
    monkeypatch.setattr(serving, "ARTIFACT", wrong)
    with pytest.raises(RuntimeError, match="checksum"):
        with TestClient(application.app):
            pytest.fail("A bad artifact must not start serving")


def test_wrong_library_version_stops_startup(monkeypatch):
    actual_version = serving.version
    monkeypatch.setattr(
        serving, "version", lambda name: "0.0.0" if name == "numpy" else actual_version(name)
    )
    with pytest.raises(RuntimeError, match="training version"):
        with TestClient(application.app):
            pytest.fail("An incompatible environment must not start serving")


def test_artifact_path_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(application.app) as client:
        assert client.post("/predict", json=EXAMPLE["input"]).status_code == 200


def test_openapi_documents_exact_width_and_required_nullable_fields(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["InputData"]
    assert set(schema["required"]) == set(FEATURES)
    assert set(schema["properties"]) == set(FEATURES)
    assert schema["additionalProperties"] is False
    assert set(InputData.model_fields) == set(FEATURES)


def test_predict_performance(client, valid_input_data):
    result = measure_requests(
        client,
        valid_input_data,
        samples=100,
        warmups=5,
        method="In-process FastAPI/Starlette TestClient; no network socket",
    )
    assert result["samples"] >= 20
    assert 0 < result["median_ms"] <= result["p95_ms"]
    assert np.isfinite(result["raw_ms"]).all()
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence/latency_testclient.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"100 requests: median={result['median_ms']:.3f} ms; p95={result['p95_ms']:.3f} ms")
