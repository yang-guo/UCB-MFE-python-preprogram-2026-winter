"""
HW5: serve the HW4 factor-score model with FastAPI.

POST /predict takes the 5 same-date factor ranks of one stock and returns its
factor score (higher = more attractive). The model is loaded once per server
process in ``lifespan``; requests never fit or refit anything.

Request handling lives here; the model itself lives in ``factor_model.py``.
Structure adapted from the Lecture 9 demo (``Lectures/Lecture 9/demo``).
"""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from factor_model import FACTOR_FEATURES, to_frame

logger = logging.getLogger("uvicorn.error")

# universal working directory.
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "factor_score.joblib"

# A percentile rank between 0 and 1
# strict=True: reject strings like "0.5" and booleans.
# allow_inf_nan=False: reject NaN and Infinity.
Rank = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class InputData(BaseModel):
    """
    One stock on one month-end signal date

    Each value is the stock's percentile rank (0 = lowest, 1 = highest) of the raw
    feature among that month's eligible top 2000 stocks. The order below is the
    training column order. The raw feature is ranked as is. The model itself flips
    volatility (1 - rank), so send the plain rank.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "rank_operating_roa": 0.9777215189873417,
                    "rank_fcf_to_assets": 0.9634888438133874,
                    "rank_drawdown_252": 0.7702632937532267,
                    "rank_momentum_252_21": 0.711925658234383,
                    "rank_volatility_63": 0.32759493670886075,
                }
            ]
        },
    )

    # the 5 HW4 factor ranking in training column order.
    # profitability: higher is better
    rank_operating_roa: Rank = Field(description="Rank of operating income / total assets")
    # cash generation: higher is better
    rank_fcf_to_assets: Rank = Field(description="Rank of free cash flow / total assets")
    # closeness to the 1-year high: higher is better
    rank_drawdown_252: Rank = Field(description="Rank of close / 252-session high - 1")
    # 12-month momentum skipping the last month: higher is better
    rank_momentum_252_21: Rank = Field(
        description="Rank of 12-month return skipping last month"
    )
    # volatility: lower is better, the model flips it to 1-rank
    rank_volatility_63: Rank = Field(description="Rank of 63-session daily return volatility")


class OutputData(BaseModel):
    prediction: float = Field(
        allow_inf_nan=False,
        ge=0.0,
        le=1.0,
        description="Factor score in [0, 1]; higher ranks the stock higher that month",
    )
    model_version: str


def load_artifact(path=ARTIFACT_PATH):
    """Load the HW4 artifact and check it matches this service's contract."""
    artifact = joblib.load(path)
    # raise errors
    if artifact["features"] != FACTOR_FEATURES:
        raise RuntimeError(f"artifact features {artifact['features']} != {FACTOR_FEATURES}")
    if artifact["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(
            f"artifact saved with sklearn {artifact['sklearn_version']}, "
            f"running {sklearn.__version__}"
        )
    return artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted pipeline once and store it on app.state.
    # Resolve artifact paths relative to this file, not the caller's directory.
    artifact = load_artifact()
    app.state.model = artifact["model"]
    app.state.model_version = artifact["model_version"]
    logger.info("Loaded %s from %s", app.state.model_version, ARTIFACT_PATH.name)
    yield


app = FastAPI(title="HW5 factor score", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with where and why validation failed, without echoing the input."""
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict:
    """Shows that the model is loaded for the restart check"""
    return {"status": "ok", "model_version": request.app.state.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # Validate shape, preserve training feature order/representation,
    # predict with the saved pipeline, and return OutputData.
    # Unexpected inference failures must remain errors, not default predictions.
    features = to_frame([data.model_dump()])
    try:
        prediction = float(request.app.state.model.predict(features)[0])
        if not math.isfinite(prediction):
            raise ValueError(f"non-finite prediction {prediction}")
    except Exception:
        # full traceback goes to the server log. The client gets 500.
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None
    return OutputData(prediction=prediction, model_version=request.app.state.model_version)
