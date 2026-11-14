"""Serve the frozen HW4 stock-return model through a validated prediction API."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model_service import FEATURE_NAMES, HW4Predictor

LOGGER = logging.getLogger(__name__)
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "hw4_frozen.joblib"


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: float = Field(ge=-1)
    momentum_63: float = Field(ge=-1)
    momentum_252_21: float = Field(ge=-1)
    volatility_63: float = Field(ge=0)
    drawdown_252: float = Field(ge=-1, le=0)
    net_margin: float
    operating_roa: float
    debt_to_assets: float
    cash_to_assets: float
    fcf_to_assets: float
    accruals_to_assets: float
    filing_age_days: float = Field(ge=0)


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    prediction: float
    model_version: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = HW4Predictor(ARTIFACT_PATH)
    try:
        yield
    finally:
        del application.state.model


app = FastAPI(title="HW4 Stock Return Predictor", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, error: RequestValidationError):
    # Raw NaN/Infinity values in validation errors cannot be encoded as JSON.
    details = [
        {"type": item["type"], "loc": item["loc"], "msg": item["msg"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        frame = pd.DataFrame([data.model_dump()], columns=FEATURE_NAMES)
        model = request.app.state.model
        prediction = float(model.predict(frame)[0])
        return OutputData(prediction=prediction, model_version=model.model_version)
    except Exception as error:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from error
