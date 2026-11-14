"""HTTP layer for the HW4 blend score. The model itself lives in blend_model.py.

Pattern for lifespan loading and the logged 500 adapted from
Lectures/Lecture 9/demo/app.py.
"""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

BUNDLE_FILE = Path(__file__).parent.resolve() / "artifacts" / "hw4_blend_v1.joblib"
log = logging.getLogger("blend_service")


class InputData(BaseModel):
    """Raw HW4 factors for one stock on one signal date, in training order.

    Returns and ratios are decimals (0.25 = 25%). The four price factors are
    always present for a stock HW4 could trade. The two accounting ratios may be
    null, which HW4 treated as a middle percentile of 0.5.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_252_21: float = Field(
        gt=-1, le=50, description="Return from 252 to 21 sessions back, decimal"
    )
    trend_6m: float = Field(
        gt=-1, le=25, description="Return over the last 126 sessions, decimal"
    )
    volatility_63: float = Field(
        ge=0, le=10, description="Annualized std of 63 daily returns, decimal"
    )
    drawdown_252: float = Field(ge=-1, le=0, description="Close / 252-session high, minus 1")
    operating_roa: float | None = Field(
        ge=-1e5, le=1e5, description="Operating income / total assets, null if unknown"
    )
    net_margin: float | None = Field(
        ge=-1e5, le=1e5, description="Net income / revenue, null if unknown"
    )


class OutputData(BaseModel):
    prediction: float = Field(
        allow_inf_nan=False,
        description="Blend score in (0, 1]: average factor percentile, higher = buy first. "
        "A ranking score, not a return.",
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(BUNDLE_FILE)  # built by freeze_artifact.py, never from outside
    if bundle["features"] != list(InputData.model_fields):
        raise RuntimeError("artifact columns and InputData fields are out of sync")
    app.state.pipeline = bundle["pipeline"]
    app.state.model_version = bundle["model_version"]
    app.state.columns = bundle["features"]
    yield
    del app.state.pipeline, app.state.model_version, app.state.columns


app = FastAPI(title="HW4 blend score", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def reject_bad_body(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default reply repeats the input. A NaN in there cannot be written
    # as JSON, which would turn a clean 422 into a crash, so send only the reasons.
    reasons = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": reasons})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    state = request.app.state
    row = pd.DataFrame([data.model_dump()], columns=state.columns, dtype="float64")
    try:
        score = float(state.pipeline.predict(row)[0])
        if not math.isfinite(score):
            raise ValueError(f"non-finite score {score}")
    except Exception as exc:
        log.exception("scoring failed for %s", data.model_dump())
        raise HTTPException(status_code=500, detail="Scoring error, see server log") from exc
    return OutputData(prediction=score, model_version=state.model_version)
