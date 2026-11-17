import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

FEATURES = [
    "return_5",
    "return_10",
    "return_21",
    "return_63",
    "return_252_skip_21",
    "volatility_63",
]
MODEL_VERSION = "hw4-ridge-v1"
ROOT = Path(__file__).resolve().parent
Return = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=-1)]
Volatility = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: tuple[Return, Return, Return, Return, Return, Volatility]


class OutputData(BaseModel):
    prediction: Annotated[float, Field(allow_inf_nan=False)]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(ROOT / "artifacts/model.joblib")
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Invalid input"})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        features = pd.DataFrame([data.features], columns=FEATURES)
        prediction = float(request.app.state.model.predict(features)[0])
        return OutputData(prediction=prediction, model_version=MODEL_VERSION)
    except Exception as exc:
        logging.getLogger(__name__).exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc
