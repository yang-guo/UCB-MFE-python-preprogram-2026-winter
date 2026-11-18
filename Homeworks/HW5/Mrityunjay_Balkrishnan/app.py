"""FastAPI service for the fitted HW4 stock-return model."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictStr,
)

LOGGER = logging.getLogger(__name__)
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "hw4_ridge_pipeline.joblib"
MODEL_VERSION = "hw4-ridge-alpha1000-fit-2026-06-30"
NUMERIC_FEATURES = (
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
)
FEATURES = (*NUMERIC_FEATURES, "sector")


def _strict_json_number(value: object) -> object:
    """Reject booleans and strings before Pydantic's normal numeric parsing."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a JSON number or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("must be finite")
    return value


ReturnFeature = (
    Annotated[
        float,
        BeforeValidator(_strict_json_number),
        Field(ge=-1, le=1_000, allow_inf_nan=False),
    ]
    | None
)
RatioFeature = (
    Annotated[
        float,
        BeforeValidator(_strict_json_number),
        Field(ge=-1_000, le=1_000, allow_inf_nan=False),
    ]
    | None
)


class InputData(BaseModel):
    """The exact ordered HW4 model inputs; nullable values use learned imputation."""

    model_config = ConfigDict(extra="forbid")

    momentum_21: ReturnFeature
    momentum_63: ReturnFeature
    momentum_252_21: ReturnFeature
    volatility_63: (
        Annotated[
            float,
            BeforeValidator(_strict_json_number),
            Field(ge=0, le=1_000, allow_inf_nan=False),
        ]
        | None
    )
    drawdown_252: (
        Annotated[
            float,
            BeforeValidator(_strict_json_number),
            Field(ge=-1, le=0, allow_inf_nan=False),
        ]
        | None
    )
    net_margin: RatioFeature
    operating_roa: RatioFeature
    debt_to_assets: RatioFeature
    cash_to_assets: RatioFeature
    fcf_to_assets: RatioFeature
    accruals_to_assets: RatioFeature
    filing_age_days: (
        Annotated[
            float,
            BeforeValidator(_strict_json_number),
            Field(ge=0, le=3_650, allow_inf_nan=False),
        ]
        | None
    )
    sector: Annotated[StrictStr, Field(min_length=1, max_length=64)]


class OutputData(BaseModel):
    """One predicted next-90-calendar-day adjusted-price return."""

    prediction: Annotated[float, Field(allow_inf_nan=False)]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load one trusted fitted pipeline per server process; never retrain here."""
    if not ARTIFACT_PATH.is_file():
        raise RuntimeError(f"Fitted model artifact is missing: {ARTIFACT_PATH}")
    app.state.model = joblib.load(ARTIFACT_PATH)
    yield
    del app.state.model


app = FastAPI(title="HW4 Stock Return Prediction API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_response(_request: Request, exc: RequestValidationError):
    """Keep rejected values and non-finite numbers out of error serialization."""
    details = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Return the saved pipeline's prediction for one validated feature record."""
    model = request.app.state.model
    frame = pd.DataFrame([{name: getattr(data, name) for name in FEATURES}], columns=FEATURES)
    try:
        value = float(np.asarray(model.predict(frame)).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("model returned a non-finite prediction")
        return OutputData(prediction=value, model_version=MODEL_VERSION)
    except Exception as exc:
        LOGGER.exception("HW4 model inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from exc
