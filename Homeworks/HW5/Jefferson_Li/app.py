"""Serve the fitted HW4 stock-return model."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
MODEL_VERSION = "hw4-ridge-v1"
FEATURES = ["momentum_63", "volatility_63"]
logger = logging.getLogger(__name__)


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    momentum_63: FiniteFloat = Field(gt=-1, description="63-session return as a fraction")
    volatility_63: FiniteFloat = Field(ge=0, description="Annualized volatility as a fraction")


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Omit raw inputs so rejected NaN and Infinity can still produce a JSON response.
    detail = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        features = pd.DataFrame([data.model_dump()], columns=FEATURES)
        prediction = float(request.app.state.model.predict(features)[0])
        return OutputData(prediction=prediction, model_version=MODEL_VERSION)
    except Exception as exc:
        logger.exception("HW4 model inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
