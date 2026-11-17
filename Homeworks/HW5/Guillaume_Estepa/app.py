"""FastAPI service for my HW4 stock-return model (Ridge on 24 ranked features).

This file is the HTTP layer only: the request and response contract, loading the model once
per process, and mapping outcomes to status codes. The model logic is in ``inference.py``.

Adapted from the Lesson 9 demo (``Lectures/Lecture 9/demo/app.py``): the lifespan loader, the
strict and finite model settings, ``/health``, the request id and the 500 handling.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from inference import load_model, predict_one

logger = logging.getLogger("uvicorn.error")

# Every HW4 feature reaches the model as a same-date cross-sectional percentile rank minus
# 0.5, so no value outside [-0.5, 0.5] can be a valid input.
Rank = Annotated[float, Field(ge=-0.5, le=0.5)]


class InputData(BaseModel):
    """One stock on one signal date, described by the 24 HW4 features in training order.

    Each value is the centred percentile rank of the quantity in the field description among
    all eligible stocks at that day's close: just above -0.5 for the lowest stock, +0.5 for
    the highest, no unit.

    All 24 keys are required. A nullable field may be ``null`` when the quantity is undefined
    for that stock: HW4 scored those as the middle rank and the saved pipeline does the same.
    The four non-nullable fields are never missing in HW4, whose eligibility screen needs them.
    """

    # strict: JSON numbers only, so "0.1" and true are rejected instead of converted.
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    # Price and volume features. Returns use adjusted closes; windows are in trading sessions.
    mom_21: Rank = Field(description="Return over the last 21 sessions")
    mom_63: Rank | None = Field(description="Return over the last 63 sessions")
    mom_126: Rank | None = Field(description="Return over the last 126 sessions")
    mom_252_21: Rank = Field(description="Return from 252 to 21 sessions ago")
    vol_63: Rank = Field(description="Standard deviation of daily returns, 63 sessions")
    beta_252: Rank | None = Field(description="Beta to the S&P 500 total return, 252 sessions")
    max_ret_21: Rank | None = Field(description="Largest daily return in the last 21 sessions")
    drawdown_252: Rank | None = Field(description="Price / 252-session high - 1")
    ma_gap_200: Rank | None = Field(description="Price / 200-session average - 1")
    dollar_vol_63: Rank = Field(description="Median daily dollar volume, 63 sessions")
    volume_trend: Rank | None = Field(description="Mean dollar volume, 21 / 252 sessions")
    # Fundamental features (latest quarterly statement, usable from filing date + 1 day).
    # Accruals = net income - operating cash flow. TTM = sum of the last four quarters.
    net_margin: Rank | None = Field(description="Net income / revenue")
    operating_roa: Rank | None = Field(description="Operating income / total assets")
    gross_profitability: Rank | None = Field(description="Gross profit / total assets")
    debt_to_assets: Rank | None = Field(description="Total debt / total assets")
    cash_to_assets: Rank | None = Field(description="Cash and equivalents / total assets")
    fcf_to_assets: Rank | None = Field(description="Free cash flow / total assets")
    accruals_to_assets: Rank | None = Field(description="Accruals / total assets")
    revenue_growth: Rank | None = Field(description="Revenue / revenue four quarters ago - 1")
    asset_growth: Rank | None = Field(description="Total assets / four quarters ago - 1")
    earnings_yield: Rank | None = Field(description="TTM net income / market value")
    fcf_yield: Rank | None = Field(description="TTM free cash flow / market value")
    book_to_market: Rank | None = Field(description="Total equity / market value")
    filing_age_days: Rank | None = Field(description="Days since the statement became usable")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted 90-calendar-day return of a purchase at the next close, as a "
        "centred percentile rank among the same day's stocks (-0.5 worst, +0.5 best)"
    )
    model_version: str = Field(description="Which HW4 fit produced the prediction")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once per server process, before the first request. Nothing is fitted here or later.
    model = load_model()
    if tuple(InputData.model_fields) != model.features:
        raise RuntimeError("InputData does not list the features the saved model was fitted on")
    app.state.model = model
    logger.info("model_loaded version=%s", model.model_version)
    yield
    del app.state.model


app = FastAPI(title="HW4 return-rank model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    # FastAPI's default handler echoes every rejected value back. Python's JSON parser accepts
    # NaN and Infinity in a request body, but they cannot be written into a JSON response, so
    # the default turns such a body into a 500. Report where and why, and leave the value out.
    problems = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]
    return JSONResponse(status_code=422, content={"detail": problems})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ready", "model_version": request.app.state.model.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    # Invalid bodies never get here: FastAPI has already answered 422 from InputData.
    model = request.app.state.model
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = predict_one(model, data.model_dump())
        result = OutputData(prediction=prediction, model_version=model.model_version)
    except Exception as error:
        # The traceback stays in the server log; the client only gets an id to quote.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from error
    response.headers["X-Request-ID"] = request_id
    return result
