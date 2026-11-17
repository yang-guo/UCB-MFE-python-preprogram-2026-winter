"""FastAPI service for the HW4 market-relative 90-day return model."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from model_service import FEATURE_NAMES, load_model, predict_one

logger = logging.getLogger("hw5")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the fitted pipeline once per server process; never refit."""
    loaded = load_model()
    app.state.model = loaded.pipeline
    app.state.model_version = loaded.version
    logger.info("Loaded model %s", loaded.version)
    yield
    app.state.model = None


app = FastAPI(title="HW5 stock relative-return model", lifespan=lifespan)


def feature(low: float, high: float, description: str):
    """A required, strictly numeric, finite feature that may be null.

    Fields have no default, so omitting a key is a 422. An explicit ``null``
    means "not available"; HW4 trained on such gaps and the fitted median
    imputer fills them, so ``null`` keeps the training representation.
    StrictFloat rejects numeric strings ("0.1") and booleans.
    """
    return Annotated[
        StrictFloat | None,
        Field(ge=low, le=high, allow_inf_nan=False, description=description),
    ]


# Bounds are sanity limits, not model limits. Typical HW4 training values
# (1st-99th percentile) sit well inside them; values outside them in the HW4
# data came from bad prices or tiny denominators and indicate an input error.
FundamentalRatio = feature(-1000.0, 1000.0, "Accounting ratio, decimal")
Momentum = feature(-1.0, 100.0, "Price return over the window, decimal (>= -100%)")
Volatility = feature(0.0, 50.0, "Annualized std of daily returns, decimal")


class InputData(BaseModel):
    """The 18 HW4 factors for one stock on one monthly signal date.

    All values are decimals (0.05 = 5%). Field order equals the training column
    order; the API rebuilds a DataFrame with these exact column names.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {name: 0.0 for name in FEATURE_NAMES},
        },
    )

    book_to_market: FundamentalRatio  # total_equity / market cap
    earnings_yield: FundamentalRatio  # net_income / market cap
    sales_to_market: FundamentalRatio  # revenue / market cap
    cash_flow_to_market: FundamentalRatio  # operating_cash_flow / market cap
    momentum_1m: Momentum  # price / price 21 trading days ago - 1
    momentum_3m: Momentum  # 63 trading days
    momentum_6m: Momentum  # 126 trading days
    momentum_12m: Momentum  # 252 trading days
    momentum_12m_1m: Momentum  # price 21d ago / price 252d ago - 1
    roe: FundamentalRatio  # net_income / total_equity
    roa: FundamentalRatio  # net_income / total_assets
    gross_profitability: FundamentalRatio  # gross_profit / total_assets
    operating_cash_flow_to_assets: FundamentalRatio
    net_margin: FundamentalRatio  # net_income / revenue
    debt_to_assets: Annotated[
        StrictFloat | None,
        Field(ge=0.0, le=1000.0, allow_inf_nan=False, description="Debt / assets"),
    ]
    volatility_21d: Volatility
    volatility_63d: Volatility
    volatility_126d: Volatility


class OutputData(BaseModel):
    prediction: float = Field(
        description="Predicted 90-day return minus same-date universe median, decimal"
    )
    model_version: str


assert tuple(InputData.model_fields) == FEATURE_NAMES


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 without echoing the rejected input.

    FastAPI's default handler echoes each bad value back; for NaN/Infinity that
    echo is not valid JSON and the handler itself crashes with a 500.
    """
    errors = [
        {"type": err["type"], "loc": list(err["loc"]), "msg": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model_version": request.app.state.model_version}


@app.post("/predict", response_model=OutputData)
def predict(payload: InputData, request: Request) -> OutputData:
    try:
        prediction = predict_one(request.app.state.model, payload.model_dump())
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None
    return OutputData(
        prediction=prediction, model_version=request.app.state.model_version
    )
