import pytest

from app import app


def test_prediction_matches_known_synthetic_result(client, valid_input):
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(-0.005)
    assert response.json()["model_version"] == "classroom-v1"
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "payload",
    [
        {"momentum_20d": 0.02},
        {"momentum_20d": 0.02, "volatility_20d": -0.15},
        {"momentum_20d": "0.02", "volatility_20d": 0.15},
        {"momentum_20d": 0.02, "volatility_20d": 0.15, "surprise": 1},
        {"momentum_20d": 20, "volatility_20d": 0.15},
    ],
)
def test_invalid_request_never_reaches_model(client, payload, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid data reached the model")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", unexpected_call)
    assert client.post("/predict", json=payload).status_code == 422


def test_nonfinite_input_rejected(client):
    from pydantic import ValidationError

    from app import InputData

    with pytest.raises(ValidationError):
        InputData(momentum_20d=float("nan"), volatility_20d=0.15)


def test_inference_failure_is_visible(client, valid_input, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(app.state.bundle["pipeline"], "predict", fail)
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction unavailable"}
    assert response.headers["X-Request-ID"]


def test_saved_preprocessing_is_independent_of_request_batch(client):
    pipeline = app.state.bundle["pipeline"]
    row = [0.02, 0.15]
    assert pipeline.predict([row])[0] == pytest.approx(pipeline.predict([row, [0.8, 1.5]])[0])


def test_health_reports_loaded_version(client):
    assert client.get("/health").json() == {"status": "ready", "model_version": "classroom-v1"}
