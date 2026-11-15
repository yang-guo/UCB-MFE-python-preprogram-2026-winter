"""HW5 service: serve the HW4 Ridge rank model (fit date 2026-01-02) over HTTP.

POST /predict takes one stock-date's 12 numeric features plus sector and returns the
model's score: a centred cross-sectional percentile-rank score of the 90-day return
(about -0.5..0.5, higher = expected to rank higher). It is NOT a return forecast.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, get_args

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model_service import load_artifact, predict_one

logger = logging.getLogger(__name__)

# Artifacts live next to this file (not in the caller's working directory).
ARTIFACT_DIR = Path(__file__).resolve().parent

# Allowed input range per numeric feature: (lower, upper), both inclusive.
# Wide on purpose: the pipeline clips each feature to its training 1st/99th percentiles,
# so these bounds only reject impossible or garbage values (see feature_ranges.csv).
BOUNDS = {
    "momentum_21": (-1.0, 1000.0),  # a return cannot fall below -100%
    "momentum_63": (-1.0, 1000.0),  # same as momentum_21
    "momentum_252_21": (-1.0, 1000.0),  # same as momentum_21
    "volatility_63": (0.0, 100.0),  # a standard deviation is never negative
    "drawdown_252": (-1.0, 0.0),  # price is between 0 and its own running max
    "net_margin": (-1000.0, 1000.0),  # ratio; generous limits for tiny revenues
    "operating_roa": (-1000.0, 1000.0),  # ratio to total assets
    "debt_to_assets": (0.0, 1000.0),  # debt is never negative
    "cash_to_assets": (-1000.0, 1000.0),  # ratio to total assets
    "fcf_to_assets": (-1000.0, 1000.0),  # free cash flow can be negative
    "accruals_to_assets": (-1000.0, 1000.0),  # accruals can be negative
    "filing_age_days": (0.0, 1095.0),  # 0 to 3 years since the filing became usable
}

# The 11 sectors the model's one-hot encoder learned (checked against metadata at startup).
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

QUARTERLY = "From the latest available quarterly filing (one quarter, not annualised, not TTM)."


def feature_field(name, description):
    """A required strict float with no NaN/inf and the bounds from BOUNDS."""
    lower, upper = BOUNDS[name]
    return Field(
        strict=True,  # reject numeric strings like "0.1" and booleans
        allow_inf_nan=False,  # reject NaN and +/-Infinity
        ge=lower,
        le=upper,
        description=f"{description} Allowed range [{lower:g}, {upper:g}].",
    )


class InputData(BaseModel):
    """One stock on one signal date. Field order = the model's training column order."""

    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422

    momentum_21: float = feature_field(
        "momentum_21",
        "Return of adjusted close over the last 21 trading sessions, as a fraction "
        "(0.05 = +5%).",
    )
    momentum_63: float = feature_field(
        "momentum_63",
        "Return of adjusted close over the last 63 trading sessions, as a fraction.",
    )
    momentum_252_21: float = feature_field(
        "momentum_252_21",
        "adj_close(t-21 sessions) / adj_close(t-252 sessions) - 1, as a fraction.",
    )
    volatility_63: float = feature_field(
        "volatility_63",
        "Standard deviation of the last 63 daily returns x sqrt(252) (annualised fraction).",
    )
    drawdown_252: float = feature_field(
        "drawdown_252",
        "Price / max price over the last 252 trading sessions - 1, as a fraction.",
    )
    net_margin: float = feature_field("net_margin", f"Net income / revenue. {QUARTERLY}")
    operating_roa: float = feature_field(
        "operating_roa", f"Operating income / total assets. {QUARTERLY}"
    )
    debt_to_assets: float = feature_field(
        "debt_to_assets", f"Total debt / total assets. {QUARTERLY}"
    )
    cash_to_assets: float = feature_field(
        "cash_to_assets", f"Cash and equivalents / total assets. {QUARTERLY}"
    )
    fcf_to_assets: float = feature_field(
        "fcf_to_assets", f"Free cash flow / total assets. {QUARTERLY}"
    )
    accruals_to_assets: float = feature_field(
        "accruals_to_assets",
        f"(Net income - operating cash flow) / total assets. {QUARTERLY}",
    )
    filing_age_days: float = feature_field(
        "filing_age_days",
        "Calendar days since (filing date + 1 day), i.e. since the filing became usable.",
    )
    # A Literal accepts only these exact strings (Pydantic does not allow strict= on it).
    sector: Sector = Field(
        description="Company sector; one of the 11 sectors seen in training."
    )


class OutputData(BaseModel):
    prediction: float = Field(
        allow_inf_nan=False,
        description="Centred cross-sectional percentile-rank score of the 90-day return "
        "(about -0.5..0.5). Not a return.",
    )
    model_version: str = Field(description="Identifier of the served HW4 model.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and verify the pipeline once per server process (never per request).
    pipeline, metadata = load_artifact(ARTIFACT_DIR)
    # Fail fast if this API's contract has drifted from the saved model.
    if list(InputData.model_fields) != metadata["feature_order"]:
        raise RuntimeError("InputData field order differs from metadata feature_order")
    if list(get_args(Sector)) != metadata["sector_categories"]:
        raise RuntimeError("Sector values differ from metadata sector_categories")
    app.state.pipeline = pipeline
    app.state.feature_order = metadata["feature_order"]
    app.state.model_version = metadata["model_version"]
    yield
    # Shutdown: drop references so the model can be garbage-collected.
    del app.state.pipeline, app.state.feature_order, app.state.model_version


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 body echoes each bad "input". JSON cannot encode NaN/Infinity,
    # so echoing them would crash the response into a 500. Drop "input", keep the rest.
    errors = [{k: v for k, v in error.items() if k != "input"} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # Invalid input never reaches here: FastAPI/Pydantic already returned 422.
    state = request.app.state
    try:
        # Look up the pipeline on every request (so tests can patch pipeline.predict).
        prediction = predict_one(state.pipeline, state.feature_order, data.model_dump())
    except Exception:
        # Full traceback in the server log; a generic message to the client; no fallback.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
    return OutputData(prediction=prediction, model_version=state.model_version)
