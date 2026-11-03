"""Adapted from the January 2026 HW5 FastAPI scaffold; a complete teaching example."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("uvicorn.error")


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    momentum_20d: float = Field(ge=-1, le=1, description="20-day return as a fraction")
    volatility_20d: float = Field(
        ge=0, le=2, description="Annualized volatility from 20 days, as a fraction"
    )


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load only the trusted artifact distributed with this course.
    app.state.bundle = joblib.load(ROOT / "artifacts/model.joblib")
    yield
    del app.state.bundle


app = FastAPI(title="MFE Lesson 9 model service", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    row = [[data.momentum_20d, data.volatility_20d]]
    request_id = uuid.uuid4().hex[:12]
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
    response.headers["X-Request-ID"] = request_id
    return result
