"""Strict HTTP contract for the frozen HW4 stock-return pipeline."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator

from model_service import FEATURES, ModelService

logger = logging.getLogger(__name__)
FiniteNumber = Annotated[StrictFloat, Field(allow_inf_nan=False)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: Annotated[
        list[FiniteNumber],
        Field(
            min_length=len(FEATURES), max_length=len(FEATURES), description=", ".join(FEATURES)
        ),
    ]

    @field_validator("features")
    @classmethod
    def check_feature_ranges(cls, values):
        features = dict(zip(FEATURES, values, strict=True))
        for name in ("momentum_21", "momentum_63", "momentum_252_21"):
            if features[name] < -1:
                raise ValueError(f"{name} must be >= -1 (fractional price return)")
        for name in ("volatility_63", "volatility_21", "filing_age_days"):
            if features[name] < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not -1 <= features["drawdown_252"] <= 0:
            raise ValueError("drawdown_252 must be between -1 and 0")
        if features["reversal_5"] > 1:
            raise ValueError("reversal_5 must be <= 1 (negative five-session return)")
        return values


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: FiniteNumber
    model_version: Annotated[str, Field(min_length=1)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService.load()
    try:
        yield
    finally:
        del app.state.service


app = FastAPI(title="Prabhutosh Kumar HW5", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Do not echo NaN/Infinity or validator exceptions into JSON error responses.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()
            ]
        },
    )


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    service = request.app.state.service
    try:
        return OutputData(
            prediction=service.predict(data.features), model_version=service.model_version
        )
    except Exception:
        logger.exception("HW4 inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
