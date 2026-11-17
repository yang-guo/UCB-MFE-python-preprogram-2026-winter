"""HW4 prediction API, adapted from the Lecture 9 FastAPI demo."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("uvicorn.error")


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_126: float | None = Field(
        ge=-1, description="126-session return fraction; null if unavailable"
    )
    momentum_252_21: float | None = Field(
        ge=-1, description="Return fraction from 252 to 21 sessions ago"
    )
    volatility_63: float = Field(
        ge=0, description="Annualized volatility from 63 daily returns"
    )
    operating_roa: float | None = Field(description="Quarterly operating income / total assets")
    accruals_to_assets: float | None = Field(
        description="Quarterly (net income - operating cash flow) / assets"
    )
    debt_to_assets: float | None = Field(description="Total debt / total assets")
    sector: str = Field(min_length=1, max_length=100)


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    if app.state.bundle["features"] != list(InputData.model_fields):
        raise ValueError("API fields do not match the saved model features")
    yield
    del app.state.bundle


app = FastAPI(title="HW5 stock-return model", lifespan=lifespan)


def predict_return(bundle, values):
    """Preserve HW4's DataFrame columns and missing-value representation."""
    frame = pd.DataFrame([values], columns=bundle["features"])
    numeric = bundle["features"][:-1]
    frame[numeric] = frame[numeric].astype(float)
    return float(bundle["pipeline"].predict(frame)[0])


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    try:
        return OutputData(
            prediction=predict_return(bundle, data.model_dump()),
            model_version=bundle["model_version"],
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc


@app.exception_handler(
    __import__("fastapi.exceptions", fromlist=["RequestValidationError"]).RequestValidationError
)
async def validation_error_handler(request, exc):
    """Return validation details without non-JSON input values."""
    from fastapi.responses import JSONResponse

    errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})
