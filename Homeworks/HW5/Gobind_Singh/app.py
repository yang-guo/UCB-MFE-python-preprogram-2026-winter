"""Serve the HW4 Ridge pipeline. The model is loaded once and never refit."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

import course_data  # noqa: F401  (needed so joblib finds QuantileClipper)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ridge_90d_full.joblib"
FEATURE_ORDER = [
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
    "sector",
]
log = logging.getLogger("hw5")


def _finite_number(value):
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError("feature must be a finite number, not a string or boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError("feature must be finite")
    return number


class InputData(BaseModel):
    """One row of the HW4 feature contract, in training column order.

    Momentum, volatility, drawdown, and the ratios are decimals. filing_age_days
    is the number of calendar days since the filing became usable. sector is the
    snapshot sector string stored with the training panel.
    """

    model_config = ConfigDict(extra="forbid")

    momentum_21: float = Field(ge=-1, le=5)
    momentum_63: float = Field(ge=-1, le=10)
    momentum_252_21: float = Field(ge=-1, le=20)
    volatility_63: float = Field(ge=0, le=5)
    drawdown_252: float = Field(ge=-1, le=0)
    net_margin: float = Field(ge=-5, le=5)
    operating_roa: float = Field(ge=-5, le=5)
    debt_to_assets: float = Field(ge=0, le=20)
    cash_to_assets: float = Field(ge=0, le=5)
    fcf_to_assets: float = Field(ge=-5, le=5)
    accruals_to_assets: float = Field(ge=-5, le=5)
    filing_age_days: float = Field(ge=0, le=550)
    sector: str = Field(min_length=1, max_length=64)

    @field_validator(
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
        mode="before",
    )
    @classmethod
    def reject_strings_bools_and_nonfinite(cls, value):
        return _finite_number(value)


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: float
    model_version: str

    @field_validator("prediction")
    @classmethod
    def prediction_is_finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("prediction must be finite")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    saved = joblib.load(MODEL_PATH)
    app.state.pipeline = saved["pipeline"]
    app.state.model_version = saved["model_version"]
    app.state.features = list(saved["features"])
    if app.state.features != FEATURE_ORDER:
        raise RuntimeError("saved feature order does not match the API contract")
    yield
    app.state.pipeline = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    frame = pd.DataFrame([{name: getattr(data, name) for name in request.app.state.features}])
    try:
        raw = request.app.state.pipeline.predict(frame)
        prediction = float(raw[0])
    except Exception:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference failed") from None
    if not math.isfinite(prediction):
        log.error("model returned a non-finite prediction")
        raise HTTPException(status_code=500, detail="inference failed")
    return OutputData(prediction=prediction, model_version=request.app.state.model_version)
