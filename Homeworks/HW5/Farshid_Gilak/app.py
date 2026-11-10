"""FastAPI service for the HW4 90-day stock-return ranking model.

Structure (lifespan loading, strict Pydantic contract, logged 500 on inference failure)
adapted from Lectures/Lecture 9/demo/app.py.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts/model.joblib"
logger = logging.getLogger("uvicorn.error")

Rank = Annotated[float, Field(gt=0, le=1)]
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
    """One stock on one signal date, in the HW4 training representation.

    Each numeric feature is the stock's percentile rank, in (0, 1], among the same signal
    date's eligible top-2000 members (pandas rank(pct=True)). null means the raw feature
    was unavailable; the pipeline imputes it as in training.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    mom_21: Rank = Field(description="1-month (21-session) return")
    mom_63: Rank = Field(description="3-month (63-session) return")
    mom_126: Rank | None = Field(description="6-month (126-session) return")
    mom_252_21: Rank | None = Field(description="12-1 month momentum")
    vol_63: Rank = Field(description="63-session annualized volatility")
    drawdown_252: Rank | None = Field(description="Distance from the 252-session high")
    beta_252: Rank | None = Field(description="252-session beta to ^SP500TR")
    max_ret_21: Rank = Field(description="Largest daily return over 21 sessions")
    gp_to_assets: Rank | None = Field(description="TTM gross profit / total assets")
    roa: Rank | None = Field(description="TTM net income / total assets")
    net_margin: Rank | None = Field(description="TTM net income / TTM revenue")
    accruals: Rank | None = Field(description="(TTM net income - TTM CFO) / total assets")
    debt_to_assets: Rank | None = Field(description="Total debt / total assets")
    cash_to_assets: Rank | None = Field(description="Cash and equivalents / total assets")
    asset_growth: Rank | None = Field(description="Year-over-year total asset growth")
    sales_growth: Rank | None = Field(description="Year-over-year quarterly revenue growth")
    share_growth: Rank | None = Field(description="Year-over-year shares outstanding growth")
    earnings_yield: Rank | None = Field(description="TTM net income / market cap")
    book_to_market: Rank | None = Field(description="Total equity / market cap")
    sales_to_price: Rank | None = Field(description="TTM revenue / market cap")
    fcf_yield: Rank | None = Field(description="TTM free cash flow / market cap")
    log_mcap: Rank | None = Field(description="Log market cap")
    filing_age_days: Rank | None = Field(
        description="Days since the latest filing was available"
    )
    sector: Sector = Field(description="Snapshot sector seen in training")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted percentile rank of the 90-calendar-day return among the same "
        "date's members (0.5 is average; higher is better)"
    )
    model_version: str


def load_model(path: Path) -> tuple[object, str]:
    """Load the trusted HW4 bundle and check it matches the InputData contract."""
    bundle = joblib.load(path)
    pipeline = bundle["pipeline"]
    if list(pipeline.feature_names_in_) != list(InputData.model_fields):
        raise RuntimeError("Artifact features do not match the InputData contract")
    return pipeline, bundle["model_version"]


def to_frame(data: InputData) -> pd.DataFrame:
    """One-row DataFrame with the training column names, order, and dtypes."""
    columns = list(InputData.model_fields)
    numeric = {name: "float64" for name in columns if name != "sector"}
    return pd.DataFrame([data.model_dump()], columns=columns).astype(numeric)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline, app.state.model_version = load_model(ARTIFACT)
    yield
    del app.state.pipeline, app.state.model_version


app = FastAPI(title="HW4 stock-return model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing inputs, which may be non-finite and so not valid JSON."""
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    state = request.app.state
    try:
        prediction = float(state.pipeline.predict(to_frame(data))[0])
        return OutputData(prediction=prediction, model_version=state.model_version)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
