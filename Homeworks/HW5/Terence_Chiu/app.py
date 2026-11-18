"""HW5: serve the fitted HW4 ridge pipeline with FastAPI."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

logger = logging.getLogger("uvicorn.error")

MODEL_VERSION = "hw4-ridge-alpha1000"

# Sectors seen in HW4 training (the pipeline's one-hot encoder categories).
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


class InputData(BaseModel):
    """One stock on one signal date: the HW4 features, in training order and units.

    Returns and ratios are fractions (0.05 = 5%). All 13 fields are required and
    must be finite; HW4's imputer is not exposed through the API.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    momentum_21: float = Field(gt=-1, description="21-session return")
    momentum_63: float = Field(gt=-1, description="63-session return")
    momentum_252_21: float = Field(gt=-1, description="Return from 252 to 21 sessions ago")
    volatility_63: float = Field(ge=0, description="Annualized volatility of 63 daily returns")
    drawdown_252: float = Field(
        gt=-1, le=0, description="Price vs. its 252-session high, minus 1"
    )
    net_margin: float = Field(description="Net income / revenue")
    operating_roa: float = Field(description="Operating income / total assets")
    debt_to_assets: float = Field(ge=0, description="Total debt / total assets")
    cash_to_assets: float = Field(ge=0, description="Cash and equivalents / total assets")
    fcf_to_assets: float = Field(description="Free cash flow / total assets")
    accruals_to_assets: float = Field(
        description="(Net income - operating cash flow) / total assets"
    )
    filing_age_days: float = Field(
        ge=0, description="Days since the financial report was filed"
    )
    sector: Sector = Field(description="Company sector, as in HW4 training")


# Training column order; the pipeline selects columns by these names.
FEATURES = list(InputData.model_fields)


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted 90-calendar-day return, as a fraction"
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted pipeline once per process, relative to this file.
    app.state.pipeline = joblib.load(
        Path(__file__).resolve().parent / "artifacts" / "model.joblib"
    )
    yield
    del app.state.pipeline


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 echoes the rejected input, and NaN/Infinity cannot be
    # encoded as JSON, which turned a 422 into a 500. Report where/why, not the value.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    # InputData has already validated every field; build one row in training order.
    pipeline = request.app.state.pipeline
    features = pd.DataFrame([data.model_dump()], columns=FEATURES)
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = float(pipeline.predict(features)[0])
        result = OutputData(prediction=prediction, model_version=MODEL_VERSION)
    except Exception as exc:
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc

    response.headers["X-Request-ID"] = request_id

    return result
