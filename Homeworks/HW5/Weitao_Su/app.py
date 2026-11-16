"""Serve the fixed HW4 Ridge pipeline without training or database access."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_utils import load_model, predict_return

logger = logging.getLogger("uvicorn.error")


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: float = Field(gt=-1, description="21-session adjusted-price return, fraction")
    momentum_63: float = Field(gt=-1, description="63-session adjusted-price return, fraction")
    momentum_252_21: float = Field(
        gt=-1, description="Adjusted-price return from 252 to 21 sessions ago, fraction"
    )
    volatility_63: float = Field(
        ge=0, le=0.8, description="63-session daily return std times sqrt(252), fraction"
    )
    drawdown_252: float = Field(
        gt=-1, le=0, description="Adjusted price / trailing 252-session maximum - 1, fraction"
    )
    relative_momentum_63: float = Field(
        description="Stock momentum_63 minus S&P 500 total-return momentum_63, fraction"
    )
    momentum_acceleration: float = Field(
        description="momentum_21 minus momentum_63 / 3, fraction"
    )


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prediction: FiniteFloat
    model_version: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A missing or incompatible artifact stops startup instead of serving a fallback.
    app.state.bundle = load_model()
    try:
        yield
    finally:
        del app.state.bundle


app = FastAPI(title="Weitao Su - HW4 stock-return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Do not echo invalid numbers: NaN and infinity cannot be encoded in JSON.
    errors = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    try:
        prediction = predict_return(bundle["pipeline"], data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        logger.exception("HW4 model prediction failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
