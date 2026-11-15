"""HW5: serve my fitted HW4 LightGBM stock-return model with FastAPI.

Request handling only; model loading and inference live in model_service.py.
The request-ID header and logged-500 pattern are adapted from the Lesson 9 demo
(Lectures/Lecture 9/demo/app.py).
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import FEATURES, load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")

# Every input is a cross-sectional percentile rank: pandas rank(pct=True) among the
# eligible top-2000 stocks on the signal date. Such a rank is always in (0, 1]: the
# smallest value is 1/n (n <= ~2,000 stocks), the largest is exactly 1. A value
# outside that range cannot come from the HW4 feature pipeline, so it is rejected
# rather than extrapolated.
Percentile = Annotated[float, Field(gt=0, le=1)]

# Fundamentals and 252-day price features were missing for 0.4-12% of the HW4
# training rows (artifacts/model_metadata.json -> feature_stats), and LightGBM was
# trained with those NaNs. Those fields are REQUIRED but may be explicitly null.
# rev_1m was never missing in training, so null is rejected for it.
NullablePercentile = Percentile | None


class InputData(BaseModel):
    """One stock on one signal date: its 12 HW4 feature percentiles.

    All 12 keys are required (no defaults), extra keys are rejected, values must be
    real JSON numbers (strict: no numeric strings, no booleans), NaN/Infinity are
    rejected, and each value must be in (0, 1]. Named fields fix the feature count
    at exactly 12 and the order is taken from the model artifact, not the request.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "idio_vol_252": 0.2416,
                "earnings_yield": 0.5604,
                "debt_to_assets": 0.4713,
                "sales_to_price": 0.2955,
                "gross_profitability": 0.6787,
                "earnings_change": 0.3602,
                "mom_12_1": 0.3711,
                "volume_trend": 0.3525,
                "accruals": 0.7955,
                "asset_growth": 0.5540,
                "rev_1m": 0.4098,
                "beta_252": 0.5851,
            }
        },
    )

    idio_vol_252: NullablePercentile = Field(
        description="Percentile of 252-session residual volatility after beta"
    )
    earnings_yield: NullablePercentile = Field(
        description="Percentile of TTM net income / market cap"
    )
    debt_to_assets: NullablePercentile = Field(
        description="Percentile of total debt / total assets"
    )
    sales_to_price: NullablePercentile = Field(
        description="Percentile of TTM revenue / market cap"
    )
    gross_profitability: NullablePercentile = Field(
        description="Percentile of TTM gross profit / total assets"
    )
    earnings_change: NullablePercentile = Field(
        description="Percentile of (NI_q - NI_q-4) / total assets"
    )
    mom_12_1: NullablePercentile = Field(
        description="Percentile of the 12-month return skipping the last month"
    )
    volume_trend: NullablePercentile = Field(
        description="Percentile of log(21-session avg volume / 252-session avg volume)"
    )
    accruals: NullablePercentile = Field(
        description="Percentile of (TTM net income - TTM operating cash flow) / total assets"
    )
    asset_growth: NullablePercentile = Field(
        description="Percentile of total assets / total assets 4 quarters earlier - 1"
    )
    rev_1m: Percentile = Field(description="Percentile of the last 21-session return")
    beta_252: NullablePercentile = Field(
        description="Percentile of 252-session beta to the S&P 500 TR"
    )


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted 90-day return in excess of the same-date average stock "
        "(winsorized), as a fraction: 0.01 = 1 percentage point"
    )
    model_version: str = Field(description="Which fitted HW4 model produced the prediction")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted HW4 model ONCE per server process; paths are resolved
    # relative to model_service.py, not the caller's working directory.
    app.state.model = load_bundle()
    logger.info("loaded model_version=%s", app.state.model.model_version)
    yield
    del app.state.model


app = FastAPI(title="HW5 - HW4 LightGBM stock-return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the offending input back. For a NaN or
    # Infinity input that echo is not valid JSON, so the default handler itself
    # crashes. Return where and why validation failed, without the input values.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ready", "model_version": request.app.state.model.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    # Reaching this point means FastAPI already validated `data`; invalid input
    # was answered with 422 and never got here.
    bundle = request.app.state.model
    request_id = uuid.uuid4().hex[:12]
    values = {name: getattr(data, name) for name in FEATURES}
    try:
        prediction = predict_one(bundle, values)
    except Exception as exc:
        # Full traceback goes to the server log; the client only gets a generic 500.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return OutputData(prediction=prediction, model_version=bundle.model_version)
