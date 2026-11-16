"""HW5: serve my HW4 stock-scoring model through FastAPI.

Structure adapted from the HW5 starter and the Lesson 9 demo
(Lectures/Lecture 9/demo/app.py): lifespan loading, request IDs, and the
500 handler. The model itself lives in hw4_model.py; this file only handles
the HTTP contract (validation -> DataFrame in training order -> predict).
"""

import logging
import math
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from hw4_model import FEATURES, expected_excess_return

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "hw4_model.joblib"
logger = logging.getLogger("uvicorn.error")

# Each feature is the signal's strength on the decision date, or null when that
# HW4 signal did not flag the stock. Lower bounds are HW4's own flag thresholds
# (a value below them cannot be a flag); upper bounds are generous sanity limits
# that reject unit mistakes (e.g. percent instead of fraction) and data errors.
PairsZ = Annotated[float, Field(ge=1.5, le=10.0)]
UnitScore = Annotated[float, Field(gt=0.0, le=1.0)]
RevenueGap = Annotated[float, Field(ge=0.001, le=10.0)]
MomentumPct = Annotated[float, Field(ge=0.9, le=1.0)]


class InputData(BaseModel):
    """One stock's four HW4 signal strengths, in training order.

    All four keys are required; use null for "signal not flagged". Strict mode
    rejects numeric strings and booleans; NaN/Infinity and unknown keys are
    rejected too, so the model always receives exactly four finite-or-null values.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    pairs_divergence_z: PairsZ | None = Field(
        description="Pairs signal: |z-score| of the 180-day log price-ratio spread by which "
        "this stock lags a correlated (>=0.7), similar-beta same-industry peer, in standard "
        "deviations. HW4 flags at >= 1.5. Range [1.5, 10] or null.",
        examples=[2.1],
    )
    fcf_quality_score: UnitScore | None = Field(
        description="FCF-growth-without-dilution signal: average same-day percentile (0-1) "
        "of FCF growth, FCF/share growth, share reduction and FCF yield. Range (0, 1] or null.",
        examples=[0.83],
    )
    revenue_growth_vs_industry: RevenueGap | None = Field(
        description="Revenue signal: YoY revenue growth minus the industry median, as a "
        "fraction, plus 0.001 (HW4 offset so a zero gap still counts). 0.05 = 5 points "
        "above peers. Range [0.001, 10] or null.",
        examples=[0.12],
    )
    momentum_percentile: MomentumPct | None = Field(
        description="12-1 momentum signal: percentile (0-1) of the 252-day return skipping "
        "the last 21 days among the day's universe. HW4 flags the top 10%. "
        "Range [0.9, 1] or null.",
        examples=[0.97],
    )


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="HW4 composite score (higher = buy first): sum of weight x percentile "
        "plus one point per agreeing signal; 0.0 = not a buy candidate. Range [0, 5]."
    )
    is_candidate: bool = Field(description="True if a positive-weight signal flags the stock")
    expected_excess_return_90d: FiniteFloat | None = Field(
        description="Informational: HW4's trailing-126-day mean 90-day excess return (vs the "
        "same-day universe average, as a fraction; 0.07 = 7%) of the signals flagging this "
        "stock, weighted by the signal weights. A historical signal-level average, not a "
        "stock-specific forecast. null when the stock is not a candidate."
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted artifact I exported myself (export_hw4_model.py), once per
    # process. joblib can execute code on load: never point this at untrusted files.
    bundle = joblib.load(MODEL_PATH)
    if list(bundle["features"]) != FEATURES:
        raise RuntimeError("Artifact feature order does not match hw4_model.FEATURES")
    app.state.bundle = bundle
    logger.info("model_loaded version=%s", bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(title="HW5 - HW4 stock-score service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with where/why for each error, without echoing the rejected value.

    FastAPI's default handler includes the input; a NaN/Infinity input then cannot
    be encoded as JSON and the 422 turns into a 500 (found by the tests).
    """
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    bundle = request.app.state.bundle
    return {
        "status": "ready",
        "model_version": bundle["model_version"],
        "snapshot_date": bundle["snapshot_date"],
    }


def to_model_frame(data: InputData) -> pd.DataFrame:
    """One-row DataFrame with the training column names and order; null -> NaN."""
    row = [math.nan if (v := getattr(data, f)) is None else v for f in FEATURES]
    return pd.DataFrame([row], columns=FEATURES, dtype="float64")


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        frame = to_model_frame(data)
        prediction = float(bundle["pipeline"].predict(frame)[0])
        if not math.isfinite(prediction):
            raise ValueError(f"non-finite prediction {prediction!r}")
        is_candidate = prediction > 0.0
        expected = float(expected_excess_return(bundle["pipeline"], frame)[0])
        if is_candidate and not math.isfinite(expected):
            raise ValueError(f"non-finite expected return {expected!r} for a candidate")
        result = OutputData(
            prediction=prediction,
            is_candidate=is_candidate,
            expected_excess_return_90d=expected if is_candidate else None,
            model_version=bundle["model_version"],
        )
    except Exception as exc:
        # Full traceback stays in the server log; the client gets a generic message.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
