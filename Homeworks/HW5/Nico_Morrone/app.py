"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import make_feature_frame, predict_frame

Number = Annotated[FiniteFloat, Field(strict=True)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    momentum_63d: Annotated[Number, Field(ge=-1)]
    volatility_21d: Annotated[Number, Field(ge=0)]
    beta_126d: Number
    max_drawdown_63d: Annotated[Number, Field(ge=-1, le=0)]
    relative_momentum_63d: Number


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_path = Path(__file__).resolve().with_name("hw4_ridge.joblib")
    app.state.bundle = joblib.load(artifact_path)
    try:
        yield
    finally:
        del app.state.bundle


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, error: RequestValidationError):
    detail = [{key: entry[key] for key in ("loc", "msg", "type")} for entry in error.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle

    try:
        frame = make_feature_frame(data.model_dump(), bundle["features"])
        (prediction,) = predict_frame(bundle, frame)
        return OutputData(
            prediction=float(prediction),
            model_version=bundle["model_version"],
        )
    except Exception as error:
        logging.getLogger(__name__).exception("HW4 inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from error
