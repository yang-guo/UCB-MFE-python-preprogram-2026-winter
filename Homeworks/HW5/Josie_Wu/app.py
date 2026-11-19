"""FastAPI service for the final Ridge pipeline submitted in HW4."""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "model.joblib"
METADATA = ROOT / "artifacts" / "model_metadata.json"
FEATURES = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
]
logger = logging.getLogger("uvicorn.error")


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    momentum_21: float = Field(gt=-1, description="21-session simple return as a fraction")
    momentum_63: float = Field(gt=-1, description="63-session simple return as a fraction")
    momentum_252_21: float = Field(gt=-1, description="Return from 252 to 21 sessions ago")
    volatility_63: float = Field(
        ge=0, description="63-session annualized volatility as a fraction"
    )
    drawdown_252: float = Field(
        ge=-1, le=0, description="Drawdown from the trailing 252-session high"
    )

    @field_validator("*", mode="before")
    @classmethod
    def require_json_numbers(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Feature values must be JSON numbers, not strings or booleans")
        return value


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT.is_file() or not METADATA.is_file():
        raise FileNotFoundError("The HW4 model artifact or metadata is missing")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("features") != FEATURES:
        raise ValueError("Model metadata feature order does not match the API contract")
    bundle = joblib.load(ARTIFACT)
    if bundle.get("features") != FEATURES or not hasattr(bundle.get("pipeline"), "predict"):
        raise ValueError("The model artifact does not match the API contract")
    if bundle.get("model_version") != metadata.get("model_version"):
        raise ValueError("Model artifact and metadata versions do not match")
    app.state.bundle = bundle
    yield
    del app.state.bundle


app = FastAPI(title="HW5 HW4-model prediction service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(_request: Request, _exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    row = pd.DataFrame([[getattr(data, name) for name in FEATURES]], columns=FEATURES)
    request_id = uuid.uuid4().hex[:12]
    try:
        predictions = np.asarray(bundle["pipeline"].predict(row), dtype=float).reshape(-1)
        if predictions.size != 1 or not np.isfinite(predictions[0]):
            raise ValueError("Model returned an invalid prediction")
        result = OutputData(
            prediction=float(predictions[0]), model_version=bundle["model_version"]
        )
    except Exception as exc:
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
