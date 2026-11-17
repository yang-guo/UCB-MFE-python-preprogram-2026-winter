"""HW5: a FastAPI service for the frozen HW4 stock-ranking model (Ian Kwon).

This module handles HTTP only: the input/output contract, startup, and error responses.
Model loading and inference live in ``model_service.py``.  The 500-handling pattern
(request ID header, traceback in the server log, generic response body) is adapted
from ``Lectures/Lecture 9/demo/app.py``.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import FEATURES, load_model

logger = logging.getLogger("uvicorn.error")

# Every input is a centred percentile rank: pandas rank(pct=True) - 0.5 across the
# signal date's eligible stocks (HW4 quantlab.features.cross_sectional_rank).  The
# smallest rank is 1/n - 0.5 (n ~ 1,400-1,900 eligible stocks), the largest is 0.5.
Rank = Annotated[float, Field(ge=-0.5, le=0.5)]

# The AAPL row scored by HW4 on 2015-12-31 (tests/data/hw4_reference.json), shown in /docs.
EXAMPLE = {
    "gross_profitability_csr": 0.10099337748344372,
    "operating_roa_csr": 0.4067328918322296,
    "accruals_to_assets_csr": 0.09889502762430935,
    "book_to_market_csr": -0.2560706401766004,
    "vol_63_csr": -0.1940463065049614,
    "mom_252_21_csr": 0.07885714285714285,
    "ret_21_csr": -0.314773980154355,
}


class InputData(BaseModel):
    """One stock on one signal date: the seven HW4 feature ranks, in training order.

    Each value is a same-date cross-sectional percentile rank centred at 0 (unitless,
    in [-0.5, 0.5]; 0.5 = highest raw value among that date's eligible stocks, 0 = the
    median).  All seven fields are required.  ``null`` means the raw feature was
    unavailable on that date (e.g. no 12-month price history or no filed statement);
    HW4 passed it to the model as NaN, which counts as the median rank.  Strings,
    booleans, NaN/Infinity, out-of-range values and unknown fields are rejected.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,  # "0.1" and true are not silently converted to numbers
        allow_inf_nan=False,
        json_schema_extra={"examples": [EXAMPLE]},
    )

    gross_profitability_csr: Rank | None = Field(
        description="Rank of quarterly gross profit / total assets (model sign +)."
    )
    operating_roa_csr: Rank | None = Field(
        description="Rank of quarterly operating income / total assets (sign +)."
    )
    accruals_to_assets_csr: Rank | None = Field(
        description="Rank of (net income - operating cash flow) / total assets (sign -)."
    )
    book_to_market_csr: Rank | None = Field(
        description="Rank of total equity / market-cap proxy (sign +)."
    )
    vol_63_csr: Rank | None = Field(
        description="Rank of annualised volatility of the last 63 daily returns (sign -)."
    )
    mom_252_21_csr: Rank | None = Field(
        description="Rank of the return from 252 to 21 sessions before the signal (sign +)."
    )
    ret_21_csr: Rank | None = Field(
        description="Rank of the return over the last 21 sessions (sign -)."
    )

    def ordered_values(self) -> list[float | None]:
        """Feature values in the model's training column order."""
        return [getattr(self, name) for name in FEATURES]


# Fail at import, not at request time, if the contract and the model order ever diverge.
if tuple(InputData.model_fields) != FEATURES:
    raise RuntimeError("InputData fields must list FEATURES in training order")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description=(
            "HW4 composite score: the signed mean of the seven ranks, in [-0.5, 0.5]. "
            "Higher means ranked more attractive for the next 90 calendar days. It is a "
            "cross-sectional ranking score, not a forecast return."
        )
    )
    model_version: str = Field(description="HW4 experiment that produced the model.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once per server process; requests only call predict, never fit.
    app.state.model = load_model()
    logger.info("model_loaded version=%s", app.state.model.version)
    yield
    del app.state.model


app = FastAPI(
    title="HW4 stock-ranking model (Ian Kwon)",
    description="Serves the frozen HW4 composite that ranks stocks for 90-day holding.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the offending input.  JSON cannot encode NaN or
    # Infinity, so echoing them crashed the response and turned a 422 into a 500 (found
    # by test_non_finite_values_are_rejected).  Report where and why, not the value.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ready", "model_version": request.app.state.model.version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    # Invalid input never gets here: FastAPI returns 422 after InputData validation.
    model = request.app.state.model
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = model.predict_one(data.ordered_values())
    except Exception as exc:
        # Full traceback goes to the server log; the client gets no internal details.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return OutputData(prediction=prediction, model_version=model.version)
