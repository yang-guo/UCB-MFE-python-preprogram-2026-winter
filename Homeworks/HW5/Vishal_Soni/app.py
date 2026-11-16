"""Serve the fitted HW4 90-day return pipeline."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model_contract import FEATURES, MODEL_VERSION

logger = logging.getLogger(__name__)
ARTIFACT = Path(__file__).resolve().with_name("model.joblib")
ReturnFeature = Annotated[float, Field(strict=True, ge=-1, allow_inf_nan=False)]
Volatility = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class InputData(BaseModel):
    """Decimal returns and annualized daily-return volatility at the signal close."""

    model_config = ConfigDict(extra="forbid")

    mom21: ReturnFeature = Field(description="21-session adjusted-close return, decimal")
    mom63: ReturnFeature = Field(description="63-session adjusted-close return, decimal")
    mom252_21: ReturnFeature = Field(description="Return from 252 to 21 sessions ago, decimal")
    vol63: Volatility = Field(description="63-session annualized return volatility, decimal")


class OutputData(BaseModel):
    prediction: float = Field(allow_inf_nan=False)
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only load the artifact shipped with this application; never accept an upload.
    app.state.model = joblib.load(ARTIFACT)
    yield
    del app.state.model


app = FastAPI(title="HW4 return prediction", version=MODEL_VERSION, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Do not echo NaN/Infinity: these are not valid JSON values.
    errors = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        frame = pd.DataFrame([data.model_dump()], columns=FEATURES)
        prediction = float(request.app.state.model.predict(frame)[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        return OutputData(prediction=prediction, model_version=MODEL_VERSION)
    except Exception:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
