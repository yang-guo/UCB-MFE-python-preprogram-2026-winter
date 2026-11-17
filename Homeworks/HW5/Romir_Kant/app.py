"""HW5: serve the HW4 ridge model over HTTP.

The request-handling pattern (lifespan load, strict Pydantic input, logged 500)
is adapted from the Lesson 9 demo, Lectures/Lecture 9/demo/app.py.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import FEATURES, load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")


def rank(description: str):
    """A feature's percentile rank among that week's stocks, minus 0.5.

    HW4 ranks every feature this way before the model sees it, so a rank
    outside [-0.5, 0.5] cannot occur in training and is rejected here.
    """
    return Field(ge=-0.5, le=0.5, description=f"{description}. Percentile rank minus 0.5")


class InputData(BaseModel):
    """One stock in one week. Every field is required; a feature that was
    missing in HW4 is sent as 0, the middle rank, as HW4 did."""

    # strict: "0.1" and true are not numbers. extra=forbid: no unknown fields.
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    mom_1m: float = rank("Return over the last 21 sessions")
    mom_3m: float = rank("Return over the last 63 sessions")
    mom_6m_skip1m: float = rank("Return from 126 to 21 sessions ago")
    mom_12m_skip1m: float = rank("Return from 252 to 21 sessions ago")
    vol_3m: float = rank("Annualized volatility of daily returns, 63 sessions")
    vol_12m: float = rank("Annualized volatility of daily returns, 252 sessions")
    drawdown_12m: float = rank("Price relative to its 252-session high, minus 1")
    max_ret_1m: float = rank("Largest one-day return in the last 21 sessions")
    beta_12m: float = rank("Beta to the S&P 500 total return, 252 sessions")
    idio_vol_12m: float = rank("Annualized volatility not explained by beta, 252 sessions")
    log_dollar_volume: float = rank("Log of average dollars traded a day, 63 sessions")
    roa: float = rank("Trailing-12-month net income / total assets")
    accruals: float = rank("(Net income - operating cash flow, 12 months) / total assets")
    earnings_yield: float = rank("Trailing-12-month net income / market value")
    sales_yield: float = rank("Trailing-12-month revenue / market value")
    fcf_yield: float = rank("Trailing-12-month free cash flow / market value")
    debt_to_assets: float = rank("Total debt / total assets")
    cash_to_assets: float = rank("Cash and equivalents / total assets")
    asset_growth: float = rank("Total assets vs. the same quarter a year earlier, minus 1")
    sales_growth: float = rank("Quarterly revenue vs. the same quarter a year earlier, minus 1")


assert tuple(InputData.model_fields) == FEATURES, (
    "InputData must list the training features in order"
)


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted percentile rank (minus 0.5) of the stock's 90-day return "
        "among that week's stocks. Higher is better; 0 is the middle."
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once per server process, before the first request.
    app.state.bundle = load_bundle()
    logger.info("loaded model %s", app.state.bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(title="HW5: HW4 stock-ranking model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 echoes the rejected value back. NaN and Infinity
    # can't be written as JSON, so that would turn a bad request into a 500.
    errors = [{"type": e["type"], "loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # FastAPI has already returned 422 if the body broke the InputData contract.
    bundle = request.app.state.bundle
    try:
        prediction = predict_one(bundle, data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # Full traceback goes to the server log; the client gets no details.
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
