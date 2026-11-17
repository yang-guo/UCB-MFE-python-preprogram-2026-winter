"""Serve the fitted HW4 ridge-rank pipeline.

The process loads ``artifacts/model.joblib`` once. Requests are not used to
fit, and this module does not open the Homework 4 database.
"""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, FiniteFloat

import preprocessing
from contract import FEATURE_BOUNDS, FEATURE_NAMES, MODEL_VERSION
from scoring import score_features

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "model.joblib"
logger = logging.getLogger("uvicorn.error")


def _reject_bool_and_str(value: object) -> object:
    """Keep JSON strings and booleans from becoming feature values."""
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError("booleans and numeric strings are not numeric features")
    return value


Number = Annotated[float, BeforeValidator(_reject_bool_and_str)]


class InputData(BaseModel):
    """One stock on one signal date, in the HW4 feature definition.

    Momentum and drawdown are fractional adjusted-close changes. Volatility is
    the standard deviation of 63 daily adjusted returns times ``sqrt(252)``.
    Financial ratios use the filing available on the next calendar day after
    the filing date. ``filing_age_days`` counts calendar days from that
    availability date to the signal date. The target the model learned is the
    same-date percentile rank of the 90-calendar-day forward return, not the
    return itself.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    momentum_21: Number = Field(
        gt=-1,
        le=100,
        description="21-session adjusted-close return, as a fraction. Greater than -1 and at most 100.",
    )
    momentum_63: Number = Field(
        gt=-1,
        le=100,
        description="63-session adjusted-close return, as a fraction. Greater than -1 and at most 100.",
    )
    momentum_252_21: Number = Field(
        gt=-1,
        le=100,
        description=(
            "Adjusted-close return from 252 sessions ago to 21 sessions ago, "
            "as a fraction. Greater than -1 and at most 100."
        ),
    )
    volatility_63: Number = Field(
        ge=0,
        le=100,
        description="Annualized 63-session volatility, as a fraction. From 0 through 100.",
    )
    drawdown_252: Number = Field(
        ge=-1,
        le=0,
        description="Adjusted close over the 252-session maximum, minus 1. From -1 through 0.",
    )
    net_margin: Number = Field(
        ge=-1000,
        le=1000,
        description="Quarterly net income / revenue. From -1000 through 1000.",
    )
    operating_roa: Number = Field(
        ge=-100,
        le=100,
        description="Quarterly operating income / total assets. From -100 through 100.",
    )
    debt_to_assets: Number = Field(
        ge=-10,
        le=1000,
        description="Total debt / total assets. From -10 through 1000.",
    )
    cash_to_assets: Number = Field(
        ge=-10,
        le=100,
        description="Cash and equivalents / total assets. From -10 through 100.",
    )
    fcf_to_assets: Number = Field(
        ge=-100,
        le=100,
        description="Free cash flow / total assets. From -100 through 100.",
    )
    accruals_to_assets: Number = Field(
        ge=-100,
        le=100,
        description="(Net income - operating cash flow) / total assets. From -100 through 100.",
    )
    filing_age_days: Number = Field(
        ge=0,
        le=10000,
        description="Calendar days since the filing became available. From 0 through 10000.",
    )


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


def _bounds_match_fields() -> None:
    """Fail import if the published bounds and the Pydantic fields disagree."""
    if list(InputData.model_fields) != FEATURE_NAMES:
        raise RuntimeError("InputData fields drifted from the HW4 feature order")
    schema = InputData.model_json_schema()["properties"]
    key_for = {
        "gt": "exclusiveMinimum",
        "ge": "minimum",
        "lt": "exclusiveMaximum",
        "le": "maximum",
    }
    for name, bounds in FEATURE_BOUNDS.items():
        spec = schema[name]
        expected = {key_for[key]: value for key, value in bounds.items()}
        actual = {key: spec[key] for key in expected}
        if actual != expected:
            raise RuntimeError(f"bounds for {name} are {actual}, contract says {expected}")


_bounds_match_fields()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(preprocessing, "QuantileClipper"):
        raise RuntimeError("custom transformer is not importable")
    bundle = joblib.load(ARTIFACT)
    features = list(bundle["features"])
    version = str(bundle["model_version"])
    if features != FEATURE_NAMES or version != MODEL_VERSION:
        raise RuntimeError("saved artifact does not match this service contract")
    app.state.pipeline = bundle["pipeline"]
    app.state.model_version = version
    yield
    del app.state.pipeline
    del app.state.model_version


def _json_safe(value):
    """Make a validation error encodable. Non-finite floats are not JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


app = FastAPI(title="HW4 ridge-rank model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def reject_invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        prediction = score_features(
            request.app.state.pipeline,
            data.model_dump(),
            FEATURE_NAMES,
        )
        return OutputData(prediction=prediction, model_version=request.app.state.model_version)
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from None
