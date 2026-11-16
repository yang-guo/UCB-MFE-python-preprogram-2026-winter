"""FastAPI service for the HW4 stock-return model.
The service loads the fitted sklearn pipeline once at process startup and does
not refit on incoming requests.  Input is validated before it can
reach predict.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from hw4_model import (
    FEATURES,
    MODEL_ARTIFACT_FILENAME,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    build_features_frame,
)

logger = logging.getLogger("hw5")

# The pipeline already clips/standardizes features, so this is a deliberately
# broad safety bound rather than a statistical domain bound.  It rejects clearly
# corrupt magnitudes without narrowing the valid HW4 feature distribution.
MAX_ABS_FEATURE_VALUE = 1e12


def _json_safe(value):
    """Replace non-finite floats so error responses are valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Exception):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    # not JSON primitive: Convert them to strings
    return str(value)


def _validate_feature_vector(value):
    """feature vector"""
    if not isinstance(value, list):
        raise ValueError("features must be a list")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"feature {index} must be a number, not a boolean or string")
        if not math.isfinite(item):
            raise ValueError(f"feature {index} must be finite")
        if abs(item) > MAX_ABS_FEATURE_VALUE:
            raise ValueError(
                f"feature {index} is outside the allowed range "
                f"[-{MAX_ABS_FEATURE_VALUE:g}, {MAX_ABS_FEATURE_VALUE:g}]"
            )
    return value


# check the width of FeatureVector
FeatureVector = Annotated[
    list[float],
    BeforeValidator(_validate_feature_vector),
    Field(min_length=len(NUMERIC_FEATURES), max_length=len(NUMERIC_FEATURES)),
]


# Numeric strings and booleans should not silently become numeric features
# input data
class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    features: FeatureVector
    sector: str = Field(min_length=1, max_length=64)


# output data
class OutputData(BaseModel):
    prediction: float
    model_version: str


def _default_artifact_path() -> Path:
    return Path(
        os.environ.get("HW5_MODEL_PATH", Path(__file__).with_name(MODEL_ARTIFACT_FILENAME))
    )


# Load the fitted model and its learned preprocessing once per server process
@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = _default_artifact_path()
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact}. "
            "Set HW5_MODEL_PATH to a trusted fitted HW4 pipeline."
        )
    try:
        app.state.model = joblib.load(artifact)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact {artifact}") from exc
    app.state.model_version = MODEL_VERSION
    app.state.feature_names = FEATURES
    logger.info("Loaded HW4 model %s from %s", MODEL_VERSION, artifact)
    yield
    app.state.model = None
    app.state.model_version = None
    app.state.feature_names = None


app = FastAPI(
    title="HW5 model service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(exc.errors())},
    )


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Return a finite prediction without swallowing inference failures."""
    frame = build_features_frame(data.features, data.sector)
    model = request.app.state.model
    try:
        raw_prediction = model.predict(frame)
        values = np.asarray(raw_prediction).reshape(-1)
        if values.size != 1:
            raise ValueError(f"Expected one prediction, got {values.size} from the pipeline")
        prediction = float(values[0])
        if not math.isfinite(prediction):
            raise ValueError("Pipeline returned a non-finite prediction")
    except Exception:
        logger.exception(
            "Inference failed for a validated request",
        )
        raise
    # 200
    return OutputData(
        prediction=prediction,
        model_version=request.app.state.model_version,
    )
