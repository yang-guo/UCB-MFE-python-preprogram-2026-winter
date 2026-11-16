"""Serve the saved HW4 Ridge pipeline without retraining or database access."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import JSONResponse

# Import the custom transformer module before loading the trusted joblib artifact.
from hw4_pipeline import FEATURES

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent

# Bounds prevent impossible or obviously malformed inputs while leaving room for
# financial ratios and price moves outside normal market conditions. See README.
FEATURE_BOUNDS = {
    "return_5": (-1.0, 100.0),
    "momentum_21": (-1.0, 100.0),
    "momentum_252_21": (-1.0, 100.0),
    "volatility_63": (0.0, 100.0),
    "drawdown_252": (-1.0, 0.0),
    "relative_momentum_63": (-100.0, 100.0),
    "beta_252": (-100.0, 100.0),
    "range_21": (0.0, 100.0),
    "volume_ratio_21_252": (0.0, 10_000.0),
    "log_dollar_volume_21": (-30.0, 60.0),
    "operating_roa": (-1_000.0, 1_000.0),
    "gross_margin": (-1_000.0, 1_000.0),
    "revenue_yoy": (-1.0, 1_000.0),
    "asset_growth_yoy": (-1.0, 1_000.0),
    "debt_to_assets": (0.0, 1_000.0),
    "cash_to_assets": (0.0, 1_000.0),
    "fcf_to_assets": (-1_000.0, 1_000.0),
    "accruals_to_assets": (-1_000.0, 1_000.0),
    "current_ratio": (0.0, 10_000.0),
    "filing_age_days": (0.0, 36_500.0),
}
assert list(FEATURE_BOUNDS) == FEATURES

FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class InputData(BaseModel):
    """Twenty finite HW4 feature values, ordered exactly as in README.md."""

    model_config = ConfigDict(extra="forbid")
    features: Annotated[
        list[FiniteNumber],
        Field(
            min_length=len(FEATURES),
            max_length=len(FEATURES),
            description="HW4 feature order: " + ", ".join(FEATURES),
        ),
    ]

    @field_validator("features")
    @classmethod
    def check_feature_bounds(cls, values: list[float]) -> list[float]:
        """Reject values outside each feature's documented inclusive interval."""
        for name, value in zip(FEATURES, values, strict=True):
            lower, upper = FEATURE_BOUNDS[name]
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        return values


class OutputData(BaseModel):
    """Finite predicted 90-day return fraction and immutable model identifier."""

    model_config = ConfigDict(extra="forbid")
    prediction: Annotated[float, Field(allow_inf_nan=False)]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted fitted pipeline once per process during startup."""
    reference = json.loads((HERE / "model_reference.json").read_text())
    if reference["features"] != FEATURES:
        raise ValueError("Saved model feature contract differs from the service")
    app.state.model = joblib.load(HERE / "model.joblib")
    app.state.model_version = reference["model_version"]
    try:
        yield
    finally:
        del app.state.model
        del app.state.model_version


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_response(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Avoid echoing non-finite rejected inputs into a JSON error body."""
    del request
    details = [
        {"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Apply HW4's fitted preprocessing and Ridge model to one feature row."""
    try:
        frame = pd.DataFrame([data.features], columns=FEATURES)
        prediction = float(request.app.state.model.predict(frame)[0])
        if not np.isfinite(prediction):
            raise ValueError("Model produced a non-finite prediction")
        return OutputData(
            prediction=prediction,
            model_version=request.app.state.model_version,
        )
    except Exception:
        LOGGER.exception("Unexpected model inference failure")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
