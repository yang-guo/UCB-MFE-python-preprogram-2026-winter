"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

# Import more things that we will need
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

from model_utils import FEATURES_15

# Set up error logging and locate the saved model file in this folder
logger = logging.getLogger(__name__)
ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifact.joblib"

# Define validation rules to ensure every input is a within a realistic range.
Ratio = Annotated[float, Field(strict=True, ge=-50.0, le=50.0, allow_inf_nan=False)]
Ret = Annotated[float, Field(strict=True, ge=-1.0, le=100.0, allow_inf_nan=False)]
NonNegRatio = Annotated[float, Field(strict=True, ge=0.0, le=50.0, allow_inf_nan=False)]
PosVol = Annotated[float, Field(strict=True, gt=0.0, le=5.0, allow_inf_nan=False)]
NegRank = Annotated[float, Field(strict=True, ge=-5000.0, le=-1.0, allow_inf_nan=False)]
McapRev = Annotated[float, Field(strict=True, gt=0.0, le=10000.0, allow_inf_nan=False)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # TODO: Define required features and enforce the documented contract. (features needed)

    model_config = ConfigDict(extra="forbid", strict=True)

    ebitda_to_assets: Ratio = Field(description="EBITDA / total assets (ratio)")
    roa: Ratio = Field(description="Net income / total assets (ratio)")
    fcf_to_assets: Ratio = Field(description="Free cash flow / total assets (ratio)")
    ocf_to_assets: Ratio = Field(description="Operating cash flow / total assets (ratio)")
    op_margin: Ratio = Field(description="Operating income / revenue (ratio)")
    gross_margin: Ratio = Field(description="Gross profit / revenue (ratio)")
    ret_252d: Ret = Field(description="252-trading-day total return (decimal)")
    mom_12m_skip1m: Ret = Field(description="12m-1m momentum return (decimal)")
    earnings_yield: Ratio = Field(description="Net income / market cap (ratio)")
    fcf_yield: Ratio = Field(description="Free cash flow / market cap (ratio)")
    book_to_market: Ratio = Field(description="Total equity / market cap (ratio)")
    neg_universe_rank: NegRank = Field(description="Negative market-cap rank (-5000 to -1)")
    vol_63d: PosVol = Field(description="63-day daily return std dev (decimal > 0)")
    avg_hl_spread_21d: NonNegRatio = Field(description="21-day mean (high-low)/close (decimal)")
    mcap_to_rev_ttm: McapRev = Field(description="Market cap / TTM revenue (positive ratio)")


class OutputData(BaseModel):
    # TODO: Define a finite prediction and a model_version field.
    prediction: float = Field(description="Predicted centered 90d return rank in [-0.5, 0.5]")
    model_version: str = Field(description="Serialized HW4 model version")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: Load the trusted fitted pipeline once and store it on app.state.
    # Resolve artifact paths relative to this file, not the caller's directory.
    artifact = joblib.load(ARTIFACT_PATH)
    app.state.pipeline = artifact["pipeline"]
    app.state.feature_names = list(artifact.get("feature_names", FEATURES_15))
    app.state.model_version = str(artifact["model_version"])
    yield
    # TODO: Release any resources that need explicit cleanup.
    app.state.pipeline = None


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a clean 422 response even when raw input contains non-JSON-compliant NaN/Inf."""
    sanitized_errors = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": sanitized_errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # TODO: Validate shape, preserve training feature order/representation,
    # predict with the saved pipeline, and return OutputData.
    # Unexpected inference failures must remain errors, not default predictions.

    pipeline = getattr(request.app.state, "pipeline", None)
    feature_names = getattr(request.app.state, "feature_names", FEATURES_15)
    model_version = getattr(request.app.state, "model_version", "unknown")

    if pipeline is None:
        logger.error("Model pipeline is not loaded.")
        raise HTTPException(status_code=500, detail="Model unavailable.")

    payload = data.model_dump()
    if list(payload.keys()) != list(feature_names) or len(payload) != len(FEATURES_15):
        raise HTTPException(status_code=422, detail="Invalid feature count or ordering.")

    try:
        features_df = pd.DataFrame(
            [[payload[col] for col in feature_names]], columns=feature_names
        )
        pred_value = float(pipeline.predict(features_df)[0])
        if not math.isfinite(pred_value):
            raise ValueError("Non-finite prediction returned by model.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected inference failure during /predict.")
        raise HTTPException(status_code=500, detail="Internal inference error.") from None

    return OutputData(prediction=pred_value, model_version=model_version)
