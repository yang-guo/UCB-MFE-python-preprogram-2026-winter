"""HW5: FastAPI service for the HW4 Ridge model.

Every input is the same-date percentile rank of one HW4 feature among the
stocks scored on that signal date, as computed by HW4 (``FeatureStore.
cross_section`` with the top-1000 screen). Ranks are unitless and lie in
(0, 1]: ``rank(pct=True)`` gives 1/n to 1 for n stocks. The description of
each field gives the underlying raw feature and its unit.
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import load_model, predict_one

logger = logging.getLogger("uvicorn.error")

Rank = Annotated[float, Field(gt=0, le=1)]


def rank_of(raw: str) -> str:
    return f"Same-date percentile rank in (0, 1] of: {raw}"


class InputData(BaseModel):
    """The 28 HW4 rank features, all required, in training order.

    `Rank | None` fields may be null (missing): the HW4 pipeline imputes them with
    its training median and a missing-value indicator. The other seven were never
    missing in the HW4 training rows and must be numbers.
    Strings, booleans, NaN, infinities, missing fields and extra fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    r_rev_5: Rank = Field(description=rank_of("5-session return (fraction)"))
    r_mom_21: Rank = Field(description=rank_of("21-session return (fraction)"))
    r_mom_63: Rank = Field(description=rank_of("63-session return (fraction)"))
    r_mom_126: Rank | None = Field(description=rank_of("126-session return (fraction)"))
    r_mom_252_21: Rank | None = Field(
        description=rank_of("return from 252 to 21 sessions ago (fraction)")
    )
    r_vol_63: Rank = Field(
        description=rank_of("annualized volatility of daily returns, 63 sessions")
    )
    r_vol_252: Rank | None = Field(
        description=rank_of("annualized volatility of daily returns, 252 sessions")
    )
    r_drawdown_252: Rank | None = Field(
        description=rank_of("price / 252-session high - 1 (fraction)")
    )
    r_max_ret_21: Rank = Field(
        description=rank_of("largest daily return over 21 sessions (fraction)")
    )
    r_beta_252: Rank | None = Field(
        description=rank_of("beta to the S&P 500 total return index, 252 sessions")
    )
    r_idio_vol_252: Rank | None = Field(
        description=rank_of("annualized residual volatility vs the index, 252 sessions")
    )
    r_log_dollar_volume_63: Rank | None = Field(
        description=rank_of("log of the median daily dollar volume, 63 sessions")
    )
    r_gross_margin: Rank | None = Field(description=rank_of("TTM gross profit / TTM revenue"))
    r_operating_margin: Rank | None = Field(
        description=rank_of("TTM operating income / TTM revenue")
    )
    r_roa: Rank | None = Field(description=rank_of("TTM net income / total assets"))
    r_roe: Rank | None = Field(description=rank_of("TTM net income / equity"))
    r_accruals: Rank | None = Field(
        description=rank_of("(TTM net income - TTM operating cash flow) / total assets")
    )
    r_fcf_to_assets: Rank | None = Field(
        description=rank_of("TTM free cash flow / total assets")
    )
    r_debt_to_assets: Rank | None = Field(description=rank_of("total debt / total assets"))
    r_cash_to_assets: Rank | None = Field(description=rank_of("cash / total assets"))
    r_sales_growth: Rank | None = Field(
        description=rank_of("year-over-year quarterly revenue growth (fraction)")
    )
    r_asset_growth: Rank | None = Field(
        description=rank_of("year-over-year total asset growth (fraction)")
    )
    r_earnings_yield: Rank | None = Field(
        description=rank_of("TTM net income / market capitalization")
    )
    r_book_to_market: Rank | None = Field(description=rank_of("equity / market capitalization"))
    r_fcf_yield: Rank | None = Field(
        description=rank_of("TTM free cash flow / market capitalization")
    )
    r_filing_age: Rank | None = Field(
        description=rank_of("days since the latest filing became available")
    )
    r_log_size: Rank = Field(description=rank_of("log market capitalization ($)"))
    r_universe_rank_pct: Rank = Field(
        description=rank_of("rank in the quarterly top-2000 list (1 = largest)")
    )


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description=(
            "Predicted same-date percentile rank of the stock's 90-day return "
            "(higher = better expected relative performance). Ridge output, not clipped."
        )
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once per server process; requests never fit or reload the model.
    app.state.model = load_model()
    logger.info("Loaded model %s", app.state.model.model_version)
    yield
    del app.state.model


app = FastAPI(title="HW5 – HW4 stock-return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the input, and a NaN or infinite input
    # cannot be written as JSON, which turned the 422 into a 500. Echo only
    # the location, type and message of each error.
    errors = [{key: e[key] for key in ("type", "loc", "msg")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ready", "model_version": request.app.state.model.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    model = request.app.state.model
    try:
        prediction = predict_one(model, data.model_dump())
    except Exception:
        # Full traceback in the server log, generic message in the response.
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None
    return OutputData(prediction=prediction, model_version=model.model_version)
