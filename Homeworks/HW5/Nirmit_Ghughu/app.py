"""FastAPI service for the HW4 Ridge stock-ranking model.

Structure adapted from the Lesson 9 demo (``Lectures/Lecture 9/demo/app.py``):
lifespan loading, strict Pydantic contract, request IDs on 500s.  The contract,
the artifact and the model logic (``hw4_model.py``) are specific to my HW4.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from hw4_model import INPUT_COLUMNS

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "hw4_ridge.joblib"
logger = logging.getLogger("uvicorn.error")

# Every numeric feature is a within-date percentile rank: rank(pct=True) - 0.5, computed
# across that signal date's tradable stocks by the HW4 pipeline.  Valid values are therefore
# in [-0.5, 0.5].  ``null`` means the raw feature was unavailable (e.g. no recent filing);
# HW4 treated that as the median (0) plus a missing indicator, so it is allowed explicitly.
# strict=True: numeric strings ("0.1") and booleans are rejected, never coerced.
Rank = Annotated[
    float | None,
    Field(ge=-0.5, le=0.5, strict=True, allow_inf_nan=False),
]
# Sectors the model was fitted with (see artifacts/model_metadata.json).
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
    """One stock on one signal date: 22 ranked features + sector, all required.

    Order below = training column order.  Units before ranking are noted; after
    ranking every value is unitless in [-0.5, 0.5] (or null).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    mom_5: Rank = Field(description="rank of 5-session total return")
    mom_21: Rank = Field(description="rank of 21-session total return")
    mom_63: Rank = Field(description="rank of 63-session total return")
    mom_126: Rank = Field(description="rank of 126-session total return")
    mom_252_21: Rank = Field(description="rank of 12-1 momentum (t-252 to t-21 return)")
    vol_63: Rank = Field(description="rank of 63-session annualized volatility")
    max_ret_21: Rank = Field(description="rank of max daily return over 21 sessions")
    drawdown_252: Rank = Field(description="rank of drawdown from 252-session high")
    beta_252: Rank = Field(description="rank of 252-session beta to ^SP500TR")
    log_dollar_vol_63: Rank = Field(description="rank of log 63-session avg dollar volume")
    gross_profitability: Rank = Field(description="rank of gross profit / total assets")
    operating_roa: Rank = Field(description="rank of operating income / total assets")
    net_margin: Rank = Field(description="rank of net income / revenue")
    debt_to_assets: Rank = Field(description="rank of total debt / total assets")
    cash_to_assets: Rank = Field(description="rank of cash / total assets")
    fcf_to_assets: Rank = Field(description="rank of free cash flow / total assets")
    accruals_to_assets: Rank = Field(description="rank of (net income - CFO) / assets")
    asset_growth: Rank = Field(description="rank of YoY total-asset growth")
    earnings_yield: Rank = Field(description="rank of 4x quarterly net income / cap proxy")
    book_to_market: Rank = Field(description="rank of total equity / cap proxy")
    filing_age_days: Rank = Field(description="rank of days since latest usable filing")
    log_mcap: Rank = Field(description="rank of log capitalization proxy")
    sector: Sector = Field(description="snapshot sector label, one of the 11 fitted sectors")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="predicted within-date percentile rank of the 90-calendar-day return, "
        "minus 0.5 (higher = expected to outperform peers on the same date)"
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted artifact created by export_model.py once per process.
    app.state.bundle = joblib.load(ARTIFACT)
    logger.info("loaded model %s", app.state.bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(title="HW4 stock-ranking model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the raw input.

    FastAPI's default handler echoes each bad value; a NaN/Infinity cannot be
    written as JSON, so the default turned a correct rejection into a 500.
    """
    errors = [{k: e[k] for k in ("type", "loc", "msg") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


def to_frame(data: InputData) -> pd.DataFrame:
    """One-row DataFrame with the training column names and order."""
    row = data.model_dump()
    return pd.DataFrame([[row[c] for c in INPUT_COLUMNS]], columns=INPUT_COLUMNS).astype(
        {c: float for c in INPUT_COLUMNS if c != "sector"}
    )


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = float(bundle["pipeline"].predict(to_frame(data))[0])
        if not np.isfinite(prediction):
            raise ValueError("model returned a non-finite prediction")
        result = OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # Full traceback stays in the server log; the client only sees a generic 500.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
