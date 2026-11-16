"""FastAPI inference service for the HW4 stock-return Ridge pipeline."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifact.joblib"

StrictNumber = StrictFloat | StrictInt


def _sanitize_non_finite(obj: Any) -> Any:
    """Convert non-finite floats in validation error payloads into JSON-safe strings."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


class InputData(BaseModel):
    """Strict input schema matching the HW4 trailing equity features at signal close t."""

    model_config = ConfigDict(extra="forbid")

    mom_20d: StrictNumber = Field(
        ...,
        ge=-1.0,
        le=50.0,
        allow_inf_nan=False,
        description="20-trading-day trailing return at signal close t (decimal).",
    )
    mom_60d: StrictNumber = Field(
        ...,
        ge=-1.0,
        le=50.0,
        allow_inf_nan=False,
        description="60-trading-day trailing return at signal close t (decimal).",
    )
    vol_20d: StrictNumber = Field(
        ...,
        ge=0.0,
        le=5.0,
        allow_inf_nan=False,
        description="20-trading-day trailing return standard deviation at signal close t.",
    )


class OutputData(BaseModel):
    """Prediction response payload for the 90-calendar-day forward return."""

    prediction: float = Field(
        ...,
        allow_inf_nan=False,
        description="Predicted 90-calendar-day forward return starting at t+1 close.",
    )
    model_version: str = Field(
        ...,
        description="Identifier of the served HW4 model artifact.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serialized HW4 pipeline once on server startup."""
    artifact = joblib.load(ARTIFACT_PATH)
    app.state.pipeline = artifact["pipeline"]
    app.state.feature_names = artifact["feature_names"]
    app.state.model_version = artifact["model_version"]
    yield
    app.state.pipeline = None


app = FastAPI(
    title="HW4 Stock Return Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return HTTP 422 even when raw request payloads contain non-finite floats like NaN/Inf."""
    safe_errors = _sanitize_non_finite(exc.errors())
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(safe_errors)})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Run inference using the preloaded HW4 preprocessing and Ridge regression model."""
    pipeline = request.app.state.pipeline
    feature_names = request.app.state.feature_names
    model_version = request.app.state.model_version

    try:
        features_df = pd.DataFrame(
            [[float(data.mom_20d), float(data.mom_60d), float(data.vol_20d)]],
            columns=feature_names,
        )
        if features_df.shape != (1, len(feature_names)):
            raise ValueError(f"Unexpected feature matrix shape: {features_df.shape}")

        prediction_value = float(pipeline.predict(features_df)[0])
        if not math.isfinite(prediction_value):
            raise ValueError("Model returned a non-finite prediction.")
    except Exception as exc:
        logger.exception("Unexpected inference failure")
        raise HTTPException(status_code=500, detail="Internal inference error") from exc

    return OutputData(prediction=prediction_value, model_version=model_version)
