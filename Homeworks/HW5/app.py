"""Serve the selected HW4 Ridge pipeline over HTTP."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import MODEL_VERSION, feature_frame, load_pipeline

logger = logging.getLogger(__name__)
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Return = Annotated[Number, Field(gt=-1)]
Volatility = Annotated[Number, Field(ge=0)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ret21: Return = Field(description="21-session adjusted-close simple return")
    ret63: Return = Field(description="63-session adjusted-close simple return")
    ret126: Return | None = Field(description="126-session adjusted-close simple return")
    ret252skip21: Return | None = Field(
        description="Return from 252 to 21 sessions before signal"
    )
    vol63: Volatility = Field(description="Sample standard deviation of 63 daily returns")
    op_margin: Number | None = Field(description="Quarterly operating income over revenue")
    cashflow_assets: Number | None = Field(
        description="Quarterly operating cash flow over total assets"
    )
    debt_assets: Number | None = Field(description="Quarterly total debt over total assets")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(description="Predicted 90-calendar-day simple return")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_pipeline()
    try:
        yield
    finally:
        del app.state.model


app = FastAPI(title="HW4 Stock Return Model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Omit rejected values so NaN and Infinity cannot break JSON error serialization.
    details = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        row = feature_frame(data.model_dump())
        value = float(request.app.state.model.predict(row).item())
        return OutputData(prediction=value, model_version=MODEL_VERSION)
    except Exception as exc:
        logger.exception("HW4 pipeline inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
