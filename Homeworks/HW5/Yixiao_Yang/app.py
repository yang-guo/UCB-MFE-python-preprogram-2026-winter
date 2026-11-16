"""Serve one restored HW4 quarterly Ridge pipeline without fitting at request time.

Adapted from the course HW5 template and Lesson 9 FastAPI lifespan example.
The feature contract and fitted model come from Yixiao Yang's HW4.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model_contract import FEATURE_COLUMNS, InputData, to_feature_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
logger = logging.getLogger("uvicorn.error")


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    prediction: float = Field(
        description="Gross adjusted-price return fraction over the HW4 90-calendar-day hold."
    )
    model_version: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # This is the trusted artifact produced by export_hw4_model.py, not an upload.
    bundle = joblib.load(MODEL_PATH)
    if bundle["features"] != FEATURE_COLUMNS:
        raise ValueError("Saved feature contract does not match the service")
    if bundle["pipeline"].n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError("Saved model has the wrong input width")
    if not isinstance(bundle["model_version"], str) or not bundle["model_version"]:
        raise ValueError("Saved model must have a nonempty version")
    application.state.bundle = bundle
    try:
        yield
    finally:
        del application.state.bundle


app = FastAPI(
    title="HW5 - Yixiao Yang's HW4 model",
    description="Twelve named features in; one 90-calendar-day stock-return forecast out.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Error inputs can themselves contain NaN/Infinity. Do not echo them into JSON.
    details = [
        {"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex
    try:
        predictions = np.asarray(bundle["pipeline"].predict(to_feature_frame(data)))
        if predictions.shape != (1,):
            raise ValueError("Expected exactly one scalar prediction")
        result = OutputData(
            prediction=float(predictions[0]), model_version=bundle["model_version"]
        )
    except Exception as exc:
        logger.exception("Prediction failed; request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
