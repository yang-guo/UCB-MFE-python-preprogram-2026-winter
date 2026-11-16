"""FastAPI service for the HW4 stock-return prediction model."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# Required when loading the serialized custom QuantileClipper transformer.
import course_data as _course_data  # noqa: F401

logger = logging.getLogger(__name__)

MODEL_VERSION = "hw4_random_forest_2025-12-31"

FEATURE_COLUMNS = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
]

FiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]


class InputData(BaseModel):
    """Twelve HW4 model features, in the original training representation.

    Return and financial features use the same decimal units as HW4.
    filing_age_days is measured in calendar days.
    """

    model_config = ConfigDict(extra="forbid")
    momentum_21: Annotated[float, Field(strict=True, ge=-1, allow_inf_nan=False)]
    momentum_63: Annotated[float, Field(strict=True, ge=-1, allow_inf_nan=False)]
    momentum_252_21: Annotated[float, Field(strict=True, ge=-1, allow_inf_nan=False)]

    volatility_63: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    drawdown_252: Annotated[float, Field(strict=True, ge=-1, le=0, allow_inf_nan=False)]

    net_margin: FiniteFloat
    operating_roa: FiniteFloat

    debt_to_assets: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    cash_to_assets: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

    fcf_to_assets: FiniteFloat
    accruals_to_assets: FiniteFloat

    filing_age_days: Annotated[float, Field(strict=True, ge=0, le=550, allow_inf_nan=False)]


class OutputData(BaseModel):
    """Prediction response."""

    prediction: Annotated[float, Field(allow_inf_nan=False)]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted fitted pipeline once per server process."""
    artifact_path = Path(__file__).resolve().parent / "model" / "model_artifact.joblib"

    if not artifact_path.is_file():
        raise RuntimeError(f"Model artifact not found: {artifact_path}")

    app.state.model = joblib.load(artifact_path)
    yield
    app.state.model = None


app = FastAPI(
    title="HW5 Stock Return Prediction API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return clean 422 responses, including for non-finite input values."""

    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        return value

    return JSONResponse(
        status_code=422,
        content={"detail": sanitize(exc.errors())},
    )


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Return a prediction from the saved HW4 pipeline."""
    model = request.app.state.model

    # Preserve the exact feature names and order used during HW4 training.
    features = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = float(model.predict(features)[0])

        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction.")
        return OutputData(
            prediction=prediction,
            model_version=MODEL_VERSION,
        )

    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from exc
