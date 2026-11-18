"""FastAPI service for the final HW4 Ridge model (see README.md for the full contract).

Request handling lives here; loading and scoring live in model_service.py, and the
preprocessing classes used by the saved pipeline live in hw4_pipeline.py.
Structure adapted from the Lesson 9 demo (Lectures/Lecture 9/demo/app.py).
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from hw4_pipeline import FEATURES, SECTORS
from model_service import load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")

# Bounds reject impossible values and obvious garbage. They are deliberately wider than
# typical data: the HW4 training panel has real vendor outliers (for example a 63.4x
# cash-to-assets ratio), and the pipeline itself clips every feature to its learned
# 1st/99th training percentiles, so extreme-but-possible values are handled by the model.
RETURN = {"ge": -1.0, "le": 100.0}  # a price cannot fall more than 100%; +10,000% cap
RATIO = {"ge": -1e6, "le": 1e6}  # accounting ratios; training 0.1%-99.9% is within +-1,000


class InputData(BaseModel):
    """One stock on one signal date; 13 features, all required, in HW4 training order.

    Returns and ratios are decimal fractions (0.05 = 5%). Fields typed ``float | None``
    were sometimes missing in the HW4 panel and must be sent as ``null`` when unknown;
    they cannot be omitted. Strict mode: numeric strings and booleans are rejected.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: float = Field(**RETURN, description="21-session price return, fraction")
    momentum_63: float = Field(**RETURN, description="63-session price return, fraction")
    momentum_252_21: float | None = Field(
        **RETURN, description="Return from 252 to 21 sessions ago, fraction; null if <1y data"
    )
    volatility_63: float = Field(
        ge=0.0, le=50.0, description="Annualized std of 63 daily returns, fraction"
    )
    drawdown_252: float | None = Field(
        ge=-1.0, le=0.0, description="Price / 252-session high - 1, fraction; null if <1y data"
    )
    net_margin: float | None = Field(**RATIO, description="Net income / revenue")
    operating_roa: float | None = Field(**RATIO, description="Operating income / total assets")
    debt_to_assets: float | None = Field(ge=0.0, le=1e6, description="Total debt / assets")
    cash_to_assets: float | None = Field(**RATIO, description="Cash and equivalents / assets")
    fcf_to_assets: float | None = Field(**RATIO, description="Free cash flow / total assets")
    accruals_to_assets: float | None = Field(
        **RATIO, description="(Net income - operating cash flow) / total assets"
    )
    filing_age_days: int = Field(
        ge=0, le=1000, description="Calendar days since the latest filing became usable"
    )
    sector: Literal[SECTORS] = Field(description="One of the 11 sectors seen in training")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Relative score: expected same-day percentile rank of the 90-day "
        "return, minus 0.5 (higher is better). Not a return forecast."
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted pipeline once per process; never refit on requests.
    bundle = load_bundle()
    if list(InputData.model_fields) != FEATURES:
        raise RuntimeError("InputData fields must match the artifact's training features")
    app.state.bundle = bundle
    logger.info("Loaded model %s", bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(title="HW5 - HW4 Ridge stock-ranking model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 without echoing the rejected input.

    FastAPI's default 422 body repeats each bad value. For a NaN/Infinity input that
    value cannot be written as JSON, so the default handler itself crashes and the client
    gets a 500. Returning only where and why validation failed avoids that (and avoids
    reflecting request data back).
    """
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    # FastAPI has already validated `data`; invalid requests return 422 before this runs.
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = predict_one(bundle["pipeline"], data.model_dump())
    except Exception as exc:
        # Full traceback goes to the server log; the client gets no internal details.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return OutputData(prediction=prediction, model_version=bundle["model_version"])
