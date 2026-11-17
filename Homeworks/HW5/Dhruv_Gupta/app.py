"""Serve the HW4 static deep histogram gradient-boosting pipeline.

Model version ``hw4-hgb-deep-static-2016-04-01`` is one fit on the first
market session on or after 2016-04-01. Training rows are month-start signals
from 2010-01-01 through the day before that session, and only labels whose
90-day hold had already ended. The prediction is that holding return as a
fraction: next-session adjusted close to the first quoted close on or after
90 calendar days.

Feature order matches the HW4 ``MODEL_COLUMNS`` list. Numeric fields are JSON
numbers. Strings and booleans are rejected, and ``sector`` is a separate
categorical field. The loaded pipeline still applies the training-set
1%/99% ``QuantileClipper`` and the saved one-hot sector encoding; this
service does not refit them.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, FiniteFloat

from scoring import EXAMPLE_PATH, MODEL_VERSION, SECTORS, load_fitted_pipeline, predict_return

Sector = Literal[*SECTORS]

logger = logging.getLogger("uvicorn.error")


def finite_number(value: object) -> float:
    """Accept a JSON number and reject bools, strings, and non-finite values.

    JSON has no integer/float split, so ``1`` and ``1.0`` are both numbers.
    ``True`` is a Python ``int`` subclass and must be rejected before that
    check. Numeric strings are not coerced.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("must be finite")
    return number


Number = Annotated[float, BeforeValidator(finite_number)]


class InputData(BaseModel):
    """One stock on one signal date, using the HW4 feature definition.

    Returns and ratios are fractions, not percents. Ranges follow the feature
    construction, then a serving cap above the HW4 panel's 99.9th percentile
    so a real experiment row is accepted and a broken price or a percent
    entered by mistake is not. The pipeline clips numeric tails at the
    training 1st and 99th percentiles after this check.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [json.loads(EXAMPLE_PATH.read_text())]},
    )

    momentum_5: Number = Field(
        gt=-1,
        le=5,
        description="5-session adjusted-close return, fraction. Prices are positive, so the value is above -1. Cap 5 is above the panel 99.9th percentile of 0.42.",
    )
    momentum_21: Number = Field(
        gt=-1,
        le=10,
        description="21-session adjusted-close return, fraction. Cap 10 is above the panel 99.9th percentile of 0.89.",
    )
    momentum_63: Number = Field(
        gt=-1,
        le=20,
        description="63-session adjusted-close return, fraction. Cap 20 is above the panel 99.9th percentile of 2.06.",
    )
    momentum_126: Number = Field(
        gt=-1,
        le=30,
        description="126-session adjusted-close return, fraction. Cap 30 is above the panel 99.9th percentile of 4.10.",
    )
    momentum_252_21: Number = Field(
        gt=-1,
        le=50,
        description="Return from 252 sessions ago to 21 sessions ago, fraction. Cap 50 is above the panel 99.9th percentile of 9.52.",
    )
    volatility_21: Number = Field(
        ge=0,
        le=10,
        description="Annualized sample volatility of 21 daily adjusted returns, fraction. A standard deviation cannot be negative. Cap 10 is above the panel 99.9th percentile of 3.28.",
    )
    volatility_63: Number = Field(
        ge=0,
        le=10,
        description="Annualized sample volatility of 63 daily adjusted returns, fraction. Cap 10 is above the panel 99.9th percentile of 3.42.",
    )
    drawdown_252: Number = Field(
        gt=-1,
        le=0,
        description="Adjusted close divided by its 252-session maximum, minus 1. The current close is inside the window, so the value is in (-1, 0].",
    )
    volume_ratio_21: Number = Field(
        ge=-1,
        le=20,
        description="Session volume divided by the 21-session average, including today, minus 1. With nonnegative volume that average implies [-1, 20].",
    )
    large_cap_score: Number = Field(
        gt=0,
        le=1,
        description="Quarterly membership score, 1 for the largest name and just above 0 for the smallest. Unitless.",
    )
    net_margin: Number = Field(
        ge=-2000,
        le=100,
        description="Net income / positive revenue, fraction. The cap covers the panel 0.1st percentile (-1068) and rejects the 1e14 filing errors.",
    )
    operating_roa: Number = Field(
        ge=-50,
        le=50,
        description="Operating income / positive assets, fraction. The cap is far above the panel 99.9th percentile of 1.72.",
    )
    debt_to_assets: Number = Field(
        ge=0,
        le=100,
        description="Total debt / positive assets, fraction. Debt in the HW4 feature is nonnegative.",
    )
    cash_to_assets: Number = Field(
        ge=-1,
        le=20,
        description="Cash and equivalents / positive assets, fraction. The panel minimum is about -0.81.",
    )
    fcf_to_assets: Number = Field(
        ge=-50,
        le=50,
        description="Free cash flow / positive assets, fraction.",
    )
    accruals_to_assets: Number = Field(
        ge=-50,
        le=50,
        description="(Net income - operating cash flow) / positive assets, fraction.",
    )
    filing_age_days: Number = Field(
        ge=0,
        le=800,
        description="Calendar days from the day after filing to the signal date. The as-of join makes this nonnegative. The panel maximum is 633.",
    )
    market_momentum_63: Number = Field(
        gt=-1,
        le=1,
        description="^SP500TR 63-session adjusted return, fraction, shared by every name on the signal date. The panel range is about -0.30 to 0.40.",
    )
    market_volatility_63: Number = Field(
        ge=0,
        le=2,
        description="^SP500TR annualized 63-session volatility, fraction. The panel maximum is 0.62.",
    )
    sector_relative_momentum: Number = Field(
        ge=-25,
        le=25,
        description="momentum_63 minus the same-day sector median, fraction. Requires at least 3 sector members with momentum_63. Same unit as momentum_63.",
    )
    sector_profitability_rank: Number = Field(
        gt=0,
        le=1,
        description="Average percentile rank of operating_roa inside the signal date and sector. In (0, 1]. Requires at least 3 sector members with operating_roa.",
    )
    sector: Sector = Field(
        description="GICS-style sector label stored on the HW4 panel. One-hot encoded with the categories learned at the April 2016 fit. Unknown labels are rejected."
    )


class OutputData(BaseModel):
    """90-day holding-return forecast from the frozen HW4 pipeline."""

    model_config = ConfigDict(extra="forbid")
    prediction: FiniteFloat = Field(
        description="Predicted 90-calendar-day adjusted-close holding return, as a fraction. 0.01 means 1 percent."
    )
    model_version: str = Field(
        description="HW4 fit being served. This build is hw4-hgb-deep-static-2016-04-01."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_fitted_pipeline()
    logger.info("loaded model_version=%s", app.state.bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(
    title="HW4 90-day return model",
    version=MODEL_VERSION,
    description=(
        "Predicts the 90-calendar-day adjusted-close holding return from the "
        "static deep HistGradientBoostingRegressor chosen in HW4. "
        f"Served fit: {MODEL_VERSION}."
    ),
    lifespan=lifespan,
)


def _json_safe(value):
    """Replace non-finite floats so a 422 body can be encoded.

    Python's JSON decoder accepts NaN and Infinity. Validation rejects them,
    but the default error payload echoes the raw float and then fails to
    serialize, which would turn the 422 into a 500.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def reject_invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
    )


InputData.model_validate(json.loads(EXAMPLE_PATH.read_text()))


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Score one validated feature row with the pipeline loaded at process start."""
    try:
        prediction = predict_return(request.app.state.bundle["pipeline"], data.model_dump())
    except Exception as exc:
        logger.exception("inference_failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
    return OutputData(
        prediction=prediction,
        model_version=request.app.state.bundle["model_version"],
    )
