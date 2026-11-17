"""HW5: serve my frozen HW4 stock model (Ridge F1, rank target) with FastAPI.

Model: see model/metadata.json (built once by export_model.py from my HW4 code).
Pattern adapted from Lectures/Lecture 9/demo/app.py (lifespan loading, strict Pydantic
input, logged 500 with a request id).
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

import hw5_transformers  # noqa: F401  (QuantileClipper must be importable to unpickle)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "pipeline.joblib"
METADATA_PATH = ROOT / "model" / "metadata.json"
logger = logging.getLogger("uvicorn.error")

# Training column order (cd.NUMERIC_FEATURES, then the categorical column).
NUMERIC_FEATURES = [
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
FEATURE_ORDER = NUMERIC_FEATURES + ["sector"]

# The 11 sectors seen in HW4 training (model/metadata.json "sectors").
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

# Bounds. A simple return cannot be below -1 (total loss). Upper caps and ratio caps are
# sanity guards: they keep >= 99.9% of HW4 training rows (the rest are vendor unit errors
# such as net_margin = -3e14) and are far outside the pipeline's own 1%/99% clip levels,
# so they never change a prediction that the model would make for real data.
Return = Annotated[float, Field(ge=-1, le=100)]
Ratio = Annotated[float, Field(ge=-1e4, le=1e4)]


class InputData(BaseModel):
    """One stock on one signal date (close t). All values use data through close t.

    Fields typed ``X | None`` are REQUIRED but may be JSON null, meaning "not available";
    null becomes NaN and is median-imputed with a missing indicator, as in HW4 training.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: Return = Field(description="21-session price return, fraction (0.05 = +5%)")
    momentum_63: Return = Field(description="63-session price return, fraction")
    momentum_252_21: Return | None = Field(
        description="Return from t-252 to t-21 sessions, fraction; null if < 1y history"
    )
    volatility_63: float = Field(
        ge=0, le=100, description="Annualized 63-session return volatility, fraction"
    )
    drawdown_252: Annotated[float, Field(ge=-1, le=0)] | None = Field(
        description="Close / 252-session max close - 1, fraction in [-1, 0]"
    )
    net_margin: Ratio | None = Field(description="TTM net income / revenue, fraction")
    operating_roa: Ratio | None = Field(description="Operating income / total assets")
    debt_to_assets: Ratio | None = Field(description="Total debt / total assets (>= 0)")
    cash_to_assets: Ratio | None = Field(description="Cash / total assets")
    fcf_to_assets: Ratio | None = Field(description="Free cash flow / total assets")
    accruals_to_assets: Ratio | None = Field(description="(Net income - CFO) / total assets")
    filing_age_days: float = Field(
        ge=0, le=1000, description="Calendar days since the latest financial filing"
    )
    sector: Sector = Field(description="Company sector (one of the 11 training sectors)")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted per-date percentile rank (about 0..1, higher = better) of "
        "the 90-calendar-day return from the next close; used to rank stocks on one date"
    )
    model_version: str


def to_frame(data: InputData) -> pd.DataFrame:
    """One-row DataFrame with the training column names and order (None -> NaN)."""
    row = data.model_dump()
    frame = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    return frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load only the trusted artifact that I exported from my own HW4 code.
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metadata = json.loads(METADATA_PATH.read_text())
    app.state.model_version = app.state.metadata["model_version"]
    yield
    del app.state.model


app = FastAPI(title="HW5 HW4-model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the rejected input.

    The default handler copies each bad value into the response. A NaN/Infinity token
    (accepted by Python's JSON parser, rejected by InputData) cannot be serialized back
    to JSON, so the default handler would crash instead of returning 422.
    """
    errors = [{k: e[k] for k in ("type", "loc", "msg") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ready", "model_version": request.app.state.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = float(request.app.state.model.predict(to_frame(data))[0])
        # OutputData rejects NaN/inf, so a broken model cannot return a "success".
        result = OutputData(
            prediction=prediction, model_version=request.app.state.model_version
        )
    except Exception as exc:
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
