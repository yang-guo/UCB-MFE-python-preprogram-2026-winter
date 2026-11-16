"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "model.joblib"
logger = logging.getLogger("uvicorn.error")

# sanity checks to restrict ratios and returns/MA distances
RETURN = {"ge": -1, "le": 100}
RATIO = {"ge": -1000, "le": 1000}


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    # Define required features and enforce the documented contract.

    # Tech features
    rsi_21: float = Field(ge=0, le=100, description="21-day RSI, 0-100")
    dma_20: float = Field(**RETURN, description="close / 20-day moving average - 1 (fraction)")
    dma_50: float = Field(**RETURN, description="close / 50-day moving average - 1 (fraction)")
    dma_100: float = Field(**RETURN, description="close / 100-day moving average - 1")
    momentum_252_21: float = Field(**RETURN, description="12-month return skipping last month")
    volatility_63: float = Field(ge=0, le=50, description="63-day annualized volatility")

    # Fundamentals, may be null
    roe: float | None = Field(**RATIO, description="TTM net income / equity")
    debt_to_equity: float | None = Field(**RATIO, description="total debt / equity")
    book_to_px: float | None = Field(**RATIO, description="equity / market cap")
    earnings_yield: float | None = Field(**RATIO, description="TTM net income / market cap")
    fcf_yield: float | None = Field(**RATIO, description="TTM free cash flow / market cap")
    ebitda_ev: float | None = Field(**RATIO, description="TTM EBITDA / enterprise value")


FEATURES = list(InputData.model_fields)


class OutputData(BaseModel):
    # Define a finite prediction and a model_version field.
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted pipeline once and store it on app.state.
    # Resolve artifact paths relative to this file, not the caller's directory.
    bundle = joblib.load(ARTIFACT)
    if bundle["features"] != FEATURES:
        raise RuntimeError(f"Artifact features {bundle['features']} != API {FEATURES}")
    app.state.bundle = bundle
    yield

    # Release any resources that need explicit cleanup.
    del app.state.bundle


app = FastAPI(title="HW5 stock rank model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # return error locations and messages on NaN values
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    # Validate shape, preserve training feature order/representation,
    # predict with the saved pipeline, and return OutputData.
    # Unexpected inference failures must remain errors, not default predictions.
    row = pd.DataFrame([data.model_dump()], columns=FEATURES).astype(float)
    try:
        prediction = float(bundle["pipeline"].predict(row)[0])
        result = OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc

    return result
