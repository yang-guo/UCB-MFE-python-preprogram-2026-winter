"""FastAPI service for my HW4 stock-ranking model (HTTP layer only).

Model logic (artifact loading, feature order, the preprocessing transformer)
lives in ``model_pipeline.py``. The structure of this file (lifespan loading,
``extra="forbid"``/strict validation, request IDs, logged 500s) is adapted from
the Lesson 9 demo, ``Lectures/Lecture 9/demo/app.py``.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_pipeline import INPUT_COLUMNS, SECTORS, load_bundle, predict_score

logger = logging.getLogger("uvicorn.error")

# Every rank field is a same-date percentile rank produced by HW4's
# ``groupby("date").rank(pct=True)``: k / n for the k-th smallest of n eligible
# stocks, so values lie in (0, 1]. 0 itself cannot occur, and anything above 1
# is not a percentile. null means "raw feature unavailable for this stock on
# this date" (e.g. no recent filing), which HW4 trained on: the model imputes
# 0.5 and sets that feature's missing flag. The key itself is still required.
RANK_DOC = " Same-date percentile rank in (0, 1] among eligible stocks; null if unavailable."
SectorName = Literal[tuple(SECTORS)]


def rank(description: str):
    return Field(gt=0.0, le=1.0, description=description + RANK_DOC)


class InputData(BaseModel):
    """One stock on one signal date: the 28 ranked HW4 features plus its sector.

    Field order = training column order. Raw-feature units are given for each
    field; the API itself receives the dimensionless rank of that raw feature.
    """

    # strict: "0.5" (string) and true/false (booleans) are rejected, not coerced.
    # allow_inf_nan=False: NaN/Infinity (which Python's JSON parser accepts) -> 422.
    # extra="forbid": unknown keys, including a 29th feature, -> 422.
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    r_mom_5: float | None = rank("Return over the last 5 sessions (adjusted close, fraction).")
    r_mom_21: float | None = rank("Return over the last 21 sessions (fraction).")
    r_mom_63: float | None = rank("Return over the last 63 sessions (fraction).")
    r_mom_126: float | None = rank("Return over the last 126 sessions (fraction).")
    r_mom_252_21: float | None = rank("12-1 momentum: return from t-252 to t-21 sessions.")
    r_vol_63: float | None = rank("Annualized volatility of daily returns, 63 sessions.")
    r_vol_252: float | None = rank("Annualized volatility of daily returns, 252 sessions.")
    r_max_ret_21: float | None = rank("Largest daily return over the last 21 sessions.")
    r_drawdown_252: float | None = rank("Close / 252-session high - 1 (fraction, <= 0).")
    r_beta_252: float | None = rank("252-session beta of daily returns to the S&P 500.")
    r_log_dollar_volume: float | None = rank("log of 63-session average dollar volume (USD).")
    r_gp_to_assets: float | None = rank("TTM gross profit / total assets.")
    r_roa: float | None = rank("TTM net income / total assets.")
    r_roe: float | None = rank("TTM net income / book equity (positive equity only).")
    r_op_margin: float | None = rank("TTM operating income / TTM revenue.")
    r_accruals: float | None = rank("(TTM net income - TTM operating cash flow) / assets.")
    r_fcf_to_assets: float | None = rank("TTM free cash flow / total assets.")
    r_debt_to_assets: float | None = rank("Total debt / total assets.")
    r_cash_to_assets: float | None = rank("Cash and equivalents / total assets.")
    r_asset_growth: float | None = rank("Year-over-year total-asset growth (fraction).")
    r_sales_growth: float | None = rank("Year-over-year quarterly revenue growth (fraction).")
    r_net_buyback: float | None = rank("TTM (repurchases - issuance) / total assets.")
    r_log_mcap: float | None = rank("log market cap: point-in-time shares x close (USD).")
    r_earnings_yield: float | None = rank("TTM net income / market cap.")
    r_book_to_price: float | None = rank("Book equity / market cap.")
    r_sales_to_price: float | None = rank("TTM revenue / market cap.")
    r_fcf_yield: float | None = rank("TTM free cash flow / market cap.")
    r_filing_age_days: float | None = rank("Calendar days since the latest usable filing.")
    sector: SectorName = Field(
        description="Categorical: one of HW4's 12 sector labels (exact spelling). 'Unknown' is "
        "HW4's label for a missing sector; it never occurred in training, so its dummy has "
        "coefficient 0."
    )


# The API contract and the model must agree on names and order (checked at import).
assert list(InputData.model_fields) == INPUT_COLUMNS, "InputData fields != model columns"


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Forecast of the stock's same-date percentile rank of its 90-calendar-day "
        "forward return, minus 0.5 (dimensionless; training targets lie in -0.5..+0.5). "
        "Higher = better expected relative return. HW4 bought the 10 highest scores weekly."
    )
    model_version: str = Field(description="Identifier of the loaded HW4 artifact.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted artifact once per process (hash- and version-checked).
    # The service never fits, refits or reads the HW4 database.
    app.state.model = load_bundle()
    logger.info("Loaded model %s", app.state.model.model_version)
    yield
    app.state.model = None


app = FastAPI(
    title="HW5: HW4 stock-ranking model service",
    description="Serves my HW4 Ridge ranking model (see README.md for the full contract).",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 without echoing the rejected values back to the client.

    FastAPI's default handler includes each error's ``input``. For a request
    containing NaN or Infinity that input is non-finite, so serializing the
    error response raises and the client would receive a 500 instead of a 422
    (found while testing). Field location, message and error type are enough.
    """
    errors = [{key: err[key] for key in ("type", "loc", "msg")} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ready", "model_version": request.app.state.model.model_version}


@app.post(
    "/predict",
    response_model=OutputData,
    responses={500: {"description": "Inference failed; details are in the server log"}},
)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.model
    request_id = uuid.uuid4().hex[:12]
    try:
        score = predict_score(bundle.pipeline, data.model_dump())
    except Exception as exc:
        # Full traceback goes to the server log; the client only gets a generic message
        # and an ID to quote. No default/fallback prediction is ever returned.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return OutputData(prediction=score, model_version=bundle.model_version)
