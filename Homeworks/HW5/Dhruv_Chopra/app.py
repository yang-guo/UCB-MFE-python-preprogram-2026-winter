"""FastAPI service for the HW4 v1 stock-ranking model (Ridge on cross-sectional ranks).

Structure adapted from the Lesson 9 demo (``Lectures/Lecture 9/demo/app.py``):
lifespan loading, ``FiniteFloat`` output, and a logged, generic 500 on inference
failure. The feature contract and model are my own HW4 work.
"""

import logging
import math
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from ranks_model import FEATURES

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "hw4_ridge_v1.joblib"
logger = logging.getLogger("uvicorn.error")

# Every HW4 feature is a within-date percentile rank among eligible stocks:
# pandas rank(pct=True), so values lie in (0, 1]. null means the raw feature was
# missing (e.g. no recent filing); HW4 trained with a missing indicator for that.
# strict=True rejects numeric strings and booleans; allow_inf_nan=False rejects
# NaN/inf. JSON integers 1 (top rank) are still accepted as 1.0.
Rank = Annotated[
    float | None,
    Field(
        gt=0, le=1, strict=True, allow_inf_nan=False, description="Percentile rank in (0, 1]"
    ),
]


class InputData(BaseModel):
    """One stock's 20 HW4 v1 features, each ranked within its signal date.

    All 20 fields are required (send null for a missing feature, never omit it);
    unknown fields are rejected, so the feature count is always exactly 20.
    """

    model_config = ConfigDict(extra="forbid")

    mom_5: Rank
    mom_21: Rank
    mom_63: Rank
    mom_126: Rank
    mom_252_21: Rank
    vol_63: Rank
    max_ret_21: Rank
    drawdown_252: Rank
    log_dollar_vol_63: Rank
    gross_profitability: Rank
    operating_roa: Rank
    roe: Rank
    net_margin: Rank
    debt_to_assets: Rank
    cash_to_assets: Rank
    fcf_to_assets: Rank
    accruals_to_assets: Rank
    revenue_growth_yoy: Rank
    asset_growth_yoy: Rank
    filing_age_days: Rank

    def to_frame(self) -> pd.DataFrame:
        """One-row DataFrame in the exact HW4 training column order (null -> NaN)."""
        row = [np.nan if getattr(self, f) is None else getattr(self, f) for f in FEATURES]
        return pd.DataFrame([row], columns=FEATURES, dtype=float)


# Fail at import time, not per request, if the schema drifts from the model.
assert list(InputData.model_fields) == FEATURES, "InputData must list FEATURES in order"


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted within-date percentile rank of the stock's 90-day return"
    )
    model_version: str


def load_bundle(path: Path = ARTIFACT) -> dict:
    """Load the trusted artifact written by export_hw4_model.py (never retrains)."""
    bundle = joblib.load(path)
    if bundle["features"] != FEATURES:
        raise RuntimeError("Artifact feature order does not match the service contract")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()
    logger.info("loaded model_version=%s", app.state.bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(title="HW5: HW4 stock-ranking model", lifespan=lifespan)


def _json_safe(value):
    """NaN/inf are not valid JSON; echo them back as strings in error bodies."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Same 422 body as FastAPI's default, but safe when the bad input is NaN/inf.

    The default handler echoes the offending input; json-encoding ``nan`` raises,
    which would turn a rejected request into a server error.
    """
    errors = _json_safe(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = float(bundle["pipeline"].predict(data.to_frame())[0])
        result = OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # Full traceback stays in the server log; the client gets a generic 500.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
