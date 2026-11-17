"""HW5: serve the HW4 Ridge 90-day-return model with FastAPI.

Structure adapted from the Lesson 9 demo (Lectures/Lecture 9/demo/app.py).
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model import load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")

Sector = Literal[
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]

# Bounds reject impossible values (a return below -100%, a negative volatility, a positive
# drawdown). Upper bounds are deliberately loose: the training data contains extreme raw
# values, and the saved pipeline clips every feature at its training 1%/99% quantiles, so
# large-but-possible inputs are handled exactly as in HW4. Ratio limits are sanity bounds
# just above the largest magnitude seen in training (about 3.1e14).
RETURN = dict(ge=-1, le=1e6)
RATIO = dict(ge=-1e15, le=1e15)


class InputData(BaseModel):
    """One stock on one month-end signal date; same 13 features and order as HW4.

    Returns, volatility, drawdown, and ratios are fractions (0.05 = 5%), not percents.
    `null` means "not available" (it was missing for some training rows too) and is
    imputed with the training median. strict=True rejects numeric strings and booleans.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: float = Field(description="Return over the last 21 sessions", **RETURN)
    momentum_63: float = Field(description="Return over the last 63 sessions", **RETURN)
    momentum_252_21: float | None = Field(description="12-1 month momentum", **RETURN)
    volatility_63: float = Field(ge=0, le=1e6, description="Annualized 63-day volatility")
    drawdown_252: float | None = Field(ge=-1, le=0, description="Drawdown from 252-day high")
    net_margin: float | None = Field(description="Net income / revenue", **RATIO)
    operating_roa: float | None = Field(description="Operating income / assets", **RATIO)
    debt_to_assets: float | None = Field(ge=0, le=1e15, description="Total debt / assets")
    cash_to_assets: float | None = Field(description="Cash / assets", **RATIO)
    fcf_to_assets: float | None = Field(description="Free cash flow / assets", **RATIO)
    accruals_to_assets: float | None = Field(description="Accruals / assets", **RATIO)
    filing_age_days: float = Field(
        ge=0,
        le=1000,
        description="Calendar days since the latest financial filing became usable",
    )
    sector: Sector = Field(description="Sector label as in the HW4 database")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(description="Predicted 90-day return as a fraction")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fitted pipeline once per process; it is never refit on requests.
    app.state.bundle = load_bundle()
    yield
    del app.state.bundle


app = FastAPI(title="HW5 90-day return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the rejected input. A NaN/Infinity input cannot be
    # written as JSON, which would turn a 422 into a 500, so return location + message only.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    try:
        prediction = predict_one(bundle["pipeline"], data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # Full details stay in the server log; the client gets a generic 500.
        logger.exception("inference_failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
