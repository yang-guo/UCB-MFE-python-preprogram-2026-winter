"""HW5: serve the HW4 stock-ranking model.

Request handling only; the model itself lives in ``model_service.py``.
The error handling in ``predict`` is adapted from the Lesson 9 demo ``app.py``.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import ModelService

logger = logging.getLogger("uvicorn.error")

# A percentile rank among the signal date's eligible stocks, as HW4 computes with
# pandas ``rank(pct=True)``: the lowest of n stocks gets 1/n, the highest 1.0.
# So 0 is impossible and 1 is allowed.
Rank = Annotated[float, Field(gt=0, le=1)]
# Features that were sometimes missing in HW4 training (e.g. no four-quarter
# history yet). The fitted pipeline has a missing-value indicator for each, so an
# explicit JSON null is a state the model has seen. The field itself is required.
OptionalRank = Annotated[float | None, Field(gt=0, le=1)]

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
    "Unknown",
]


class InputData(BaseModel):
    """One stock on one signal date, in the HW4 training representation.

    Field order is the training column order. ``strict`` means JSON strings and
    booleans are rejected rather than coerced; NaN and infinity are rejected.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    # Price and volume features (percentile rank of the raw value named below)
    mom_1m: OptionalRank = Field(description="21-session return")
    mom_3m: OptionalRank = Field(description="63-session return")
    mom_6m_1m: OptionalRank = Field(description="Return from 126 to 21 sessions ago")
    mom_12_1: OptionalRank = Field(description="Return from 252 to 21 sessions ago")
    vol_63: Rank = Field(description="Std. dev. of daily returns, 63 sessions")
    vol_252: OptionalRank = Field(description="Std. dev. of daily returns, 252 sessions")
    max_ret_21: Rank = Field(description="Largest daily return in 21 sessions")
    dist_high_252: OptionalRank = Field(description="Price / 252-session high - 1")
    beta_252: OptionalRank = Field(description="252-session beta to the S&P 500 TR")
    idio_vol_252: OptionalRank = Field(description="252-session idiosyncratic volatility")
    log_dollar_vol: Rank = Field(description="Log mean dollar volume, 63 sessions")
    log_amihud: OptionalRank = Field(description="Log Amihud illiquidity, 63 sessions")
    # Fundamental features (percentile rank of the raw value named below)
    log_mcap: OptionalRank = Field(description="Log market capitalisation")
    earnings_yield: OptionalRank = Field(description="TTM net income / market cap")
    book_to_market: OptionalRank = Field(description="Total equity / market cap")
    sales_to_price: OptionalRank = Field(description="TTM revenue / market cap")
    cashflow_to_price: OptionalRank = Field(description="TTM operating cash flow / market cap")
    gross_profit_to_assets: OptionalRank = Field(description="TTM gross profit / total assets")
    roe: OptionalRank = Field(description="TTM net income / total equity")
    asset_growth: OptionalRank = Field(description="Year-on-year total asset growth")
    sales_growth: OptionalRank = Field(description="Year-on-year quarterly revenue growth")
    earnings_change_to_price: OptionalRank = Field(
        description="Year-on-year change in quarterly net income / market cap"
    )
    accruals_to_assets: OptionalRank = Field(
        description="(TTM net income - TTM operating cash flow) / total assets"
    )
    debt_to_assets: OptionalRank = Field(description="Total debt / total assets")
    share_issuance: OptionalRank = Field(
        description="Year-on-year growth in shares outstanding"
    )
    filing_age_days: Rank = Field(description="Days since the latest report became public")
    # Categorical
    sector: Sector = Field(description="Sector label; 'Unknown' if the stock has none")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description=(
            "Predicted percentile rank (0 = worst, 1 = best) of the stock's 90-calendar-day "
            "forward return among the signal date's eligible stocks. A linear model's output, "
            "so it is not clipped to [0, 1]."
        )
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once per server process. Only the artifact built by export_model.py
    # is loaded; the path is resolved relative to the code, not the caller.
    app.state.model = ModelService.load()
    logger.info("model_loaded version=%s", app.state.model.model_version)
    yield
    del app.state.model


app = FastAPI(title="HW5: HW4 stock-ranking model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default handler echoes the offending input back. A NaN or Infinity
    # in the body cannot be written as JSON, which would turn this 422 into a 500.
    errors = [{k: e[k] for k in ("type", "loc", "msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ready", "model_version": request.app.state.model.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    model: ModelService = request.app.state.model
    request_id = uuid.uuid4().hex[:12]
    features = data.model_dump(exclude={"sector"})
    try:
        result = OutputData(
            prediction=model.predict(features, data.sector), model_version=model.model_version
        )
    except Exception as exc:
        # Full traceback goes to the server log; the client gets an id to quote.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
