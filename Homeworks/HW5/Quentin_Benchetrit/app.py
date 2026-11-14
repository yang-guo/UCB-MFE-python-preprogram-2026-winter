"""Serve the HW4 stock-return model at POST /predict.

The structure (strict input model, model loaded in the lifespan, logged HTTP 500 on
inference failure) follows Lectures/Lecture 9/demo/app.py.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from serving import load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")


def rank(characteristic):
    return Field(gt=0, le=1, description=f"Percentile rank of: {characteristic}")


class InputData(BaseModel):
    """The 26 inputs of the HW4 model, in training order.

    Each value is the stock's percentile rank among the stocks eligible on the signal
    date, as built by hw4/features.py: a float in (0, 1], or null when the
    characteristic is unavailable (the model then uses the middle rank).

    All fields are required. Strings, booleans, NaN, infinities, extra fields and
    values outside (0, 1] are rejected.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    mom_12_1: float | None = rank("12-month return, skipping the latest month")
    mom_6_1: float | None = rank("6-month return, skipping the latest month")
    high_252: float | None = rank("price relative to its 252-session high")
    ret_21: float | None = rank("return over the last 21 sessions")
    vol_63: float | None = rank("volatility of daily returns over 63 sessions")
    vol_252: float | None = rank("volatility of daily returns over 252 sessions")
    idio_vol: float | None = rank("volatility not explained by the S&P 500 TR, 252 sessions")
    beta: float | None = rank("beta to the S&P 500 TR over 252 sessions")
    max_ret_21: float | None = rank("largest daily return over the last 21 sessions")
    roa: float | None = rank("trailing-year net income / total assets")
    roe: float | None = rank("trailing-year net income / equity")
    gross_profitability: float | None = rank("trailing-year gross profit / total assets")
    operating_margin: float | None = rank("trailing-year operating income / revenue")
    asset_turnover: float | None = rank("trailing-year revenue / total assets")
    accruals: float | None = rank(
        "trailing-year net income minus operating cash flow, / assets"
    )
    revenue_acceleration: float | None = rank("change in year-on-year revenue growth")
    income_change: float | None = rank("year-on-year change in quarterly net income / assets")
    margin_change: float | None = rank("year-on-year change in quarterly gross margin")
    asset_growth: float | None = rank("year-on-year growth of total assets")
    share_growth: float | None = rank("year-on-year growth of shares outstanding")
    earnings_yield: float | None = rank("trailing-year net income / market value")
    book_to_price: float | None = rank("equity / market value")
    sales_to_price: float | None = rank("trailing-year revenue / market value")
    cash_flow_yield: float | None = rank("trailing-year operating cash flow / market value")
    fcf_yield: float | None = rank("trailing-year free cash flow / market value")
    ebitda_to_ev: float | None = rank("trailing-year EBITDA / enterprise value")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Forecast 90-calendar-day return minus beta x S&P 500 TR return, as a fraction"
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = load_bundle()
    if bundle["features"] != list(InputData.model_fields):
        raise RuntimeError("InputData does not list the model's features in training order")
    app.state.bundle = bundle
    yield


app = FastAPI(title="HW4 stock-return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def reject_invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 echoes the rejected value. A NaN cannot be written back
    # as JSON, which would turn the 422 into a 500, so the value is left out.
    errors = [{"loc": error["loc"], "msg": error["msg"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    try:
        prediction = predict_one(bundle, data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
