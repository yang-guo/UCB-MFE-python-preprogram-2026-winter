"""Serve one frozen HW4 stock-return model."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model import FEATURE_COLUMNS

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts/hw4_random_forest.joblib"
logger = logging.getLogger("uvicorn.error")

Momentum = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1)]
Volatility = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    momentum_21: Momentum
    momentum_63: Momentum
    momentum_126: Momentum
    momentum_252: Momentum
    volatility_21: Volatility
    volatility_63: Volatility
    ma_ratio_50: Annotated[Momentum, Field(le=49)]
    ma_ratio_200: Annotated[Momentum, Field(le=199)]


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(ARTIFACT_PATH)
    app.state.model = bundle["pipeline"]
    app.state.model_version = bundle["model_version"]
    try:
        yield
    finally:
        del app.state.model
        del app.state.model_version


app = FastAPI(title="HW4 Stock Return Model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Echoing a raw NaN or infinity would make the error body invalid JSON.
    detail = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    frame = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS, dtype="float64")
    try:
        value = request.app.state.model.predict(frame).item()
        return OutputData(prediction=value, model_version=request.app.state.model_version)
    except Exception:
        logger.exception("HW4 prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
