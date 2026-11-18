"""Serve the frozen HW4 price-only Ridge pipeline; adapted from Lesson 9."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_tools import feature_frame, load_bundle

logger = logging.getLogger(__name__)


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    momentum_21: float = Field(gt=-1, description="21-session adjusted-price return, fraction")
    momentum_63: float = Field(gt=-1, description="63-session adjusted-price return, fraction")
    momentum_252_21: float = Field(
        gt=-1, description="Adjusted-price return from 252 to 21 sessions ago, fraction"
    )
    volatility_63: float = Field(
        ge=0, le=0.6, description="63-session sample daily-return SD times sqrt(252), fraction"
    )
    drawdown_252: float = Field(
        gt=-1, le=0, description="Adjusted close / trailing 252-session maximum - 1, fraction"
    )


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    yield
    del app.state.bundle


app = FastAPI(title="Jing Jian HW4 stock-return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Python's JSON decoder accepts NaN/Infinity. Pydantic rejects them, but
    # echoing those raw input values in the default error JSON can itself fail.
    errors = [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    try:
        prediction = float(bundle["pipeline"].predict(feature_frame(data.model_dump()))[0])
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
