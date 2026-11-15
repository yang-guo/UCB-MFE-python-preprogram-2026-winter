import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).with_name("hw4_model.joblib")
MODEL_VERSION = "hw4-ff8f868"
logger = logging.getLogger(__name__)


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_rank: float = Field(
        strict=True,
        gt=0,
        le=1,
        allow_inf_nan=False,
        description="Percentile rank of the HW4 score within the signal-date cohort.",
    )


class OutputData(BaseModel):
    prediction: float = Field(allow_inf_nan=False)
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="HW4 Return Prediction API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Leave rejected values out of the response so NaN and Infinity
    # can be reported as validation errors in valid JSON.
    details = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        features = pd.DataFrame({"score_rank": [data.score_rank]})
        prediction = float(request.app.state.model.predict(features)[0])
        return OutputData(prediction=prediction, model_version=MODEL_VERSION)
    except Exception as exc:
        logger.exception("HW4 model inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from exc
