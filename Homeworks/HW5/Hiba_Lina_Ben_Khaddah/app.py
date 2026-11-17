"""Serve the fitted HW4 stock-ranking model through FastAPI."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("uvicorn.error")

Sector = Literal[
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]


class InputData(BaseModel):
    """Features calculated at one stock's signal date."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    momentum_21: float = Field(ge=-1, le=20, description="21-session return")
    momentum_63: float = Field(ge=-1, le=20, description="63-session return")
    momentum_252_21: float = Field(ge=-1, le=20, description="252-to-21-session momentum")
    volatility_63: float = Field(ge=0, le=10, description="Annualized 63-session volatility")
    drawdown_252: float = Field(ge=-1, le=0, description="Drawdown from the 252-session high")
    net_margin: float = Field(ge=-10, le=10)
    operating_roa: float = Field(ge=-10, le=10)
    debt_to_assets: float = Field(ge=0, le=10)
    cash_to_assets: float = Field(ge=0, le=10)
    fcf_to_assets: float = Field(ge=-10, le=10)
    accruals_to_assets: float = Field(ge=-10, le=10)
    filing_age_days: float = Field(
        ge=0, le=550, description="Days from filing availability to signal date"
    )
    sector: Sector


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted model once when the server starts."""

    metadata_path = ROOT / "artifacts/model_metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    app.state.model = joblib.load(ROOT / "artifacts/model.joblib")
    app.state.feature_order = metadata["feature_order"]
    app.state.model_version = metadata["model_version"]
    yield
    del app.state.model


app = FastAPI(title="HW5 stock-ranking model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, _error: RequestValidationError):
    """Return a JSON-safe response without echoing invalid non-finite values."""

    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Validate one feature row and return the fitted model's score."""

    values = data.model_dump()
    row = pd.DataFrame(
        [[values[name] for name in request.app.state.feature_order]],
        columns=request.app.state.feature_order,
    )

    try:
        prediction = float(request.app.state.model.predict(row)[0])
        return OutputData(
            prediction=prediction,
            model_version=request.app.state.model_version,
        )
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
