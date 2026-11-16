"""HW4 predictions, served through the Lecture 9 FastAPI pattern."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from model_service import load_bundle, predict_return

logger = logging.getLogger("uvicorn.error")


class InputData(BaseModel):
    """One stock/date; calculated HW4 features, before learned preprocessing."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    momentum_21: float = Field(gt=-1, description="21-session adjusted-price return, decimal")
    momentum_63: float = Field(gt=-1, description="63-session adjusted-price return, decimal")
    momentum_252_21: float = Field(
        gt=-1, description="Return from 252 sessions ago to 21 sessions ago"
    )
    volatility_63: float = Field(
        ge=0, description="63-session daily-return sample std times sqrt(252), decimal"
    )
    drawdown_252: float = Field(gt=-1, le=0, description="Price / 252-session maximum - 1")
    net_margin: float = Field(gt=0, le=1, description="HW4 sector/date rank of net margin")
    operating_roa: float = Field(
        gt=0, le=1, description="HW4 sector/date rank of operating ROA"
    )
    debt_to_assets: float = Field(gt=0, le=1, description="HW4 sector/date rank of debt/assets")
    cash_to_assets: float = Field(gt=0, le=1, description="HW4 sector/date rank of cash/assets")
    filing_age_days: float = Field(
        ge=0, multiple_of=1, description="Whole calendar days since report availability"
    )

    @field_validator("*", mode="before")
    @classmethod
    def require_number(cls, value):
        # Python treats bool as a subclass of int. Reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Use a JSON number; strings, booleans and null are not accepted")
        return value


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    prediction: float = Field(description="Predicted 90-calendar-day return; 0.05 means 5%")
    model_version: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    try:
        yield
    finally:
        del app.state.bundle


app = FastAPI(title="Mauricio Gutierrez - HW5 model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Do not echo rejected NaN/Infinity values: they cannot be encoded as valid JSON.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ]
        },
    )


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        bundle = request.app.state.bundle
        prediction = predict_return(bundle, data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
