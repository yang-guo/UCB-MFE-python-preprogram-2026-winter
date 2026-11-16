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

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent / "hw4_model.joblib"
FEATURE_COLS = [
    "ret_5d_z",
    "ret_21d_z",
    "ret_63d_z",
    "ret_126d_z",
    "mom_12_1_z",
    "vol_21d_z",
    "vol_63d_z",
    "dist_ma200_z",
    "dd_252d_z",
]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ret_5d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    ret_21d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    ret_63d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    ret_126d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    mom_12_1_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    vol_21d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    vol_63d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    dist_ma200_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)
    dd_252d_z: float = Field(..., ge=-10.0, le=10.0, allow_inf_nan=False, strict=True)


class OutputData(BaseModel):
    prediction: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(ARTIFACT_PATH)
    app.state.model = artifact["model"]
    app.state.model_version = artifact.get("model_version", "hw4-ridge-alpha10-v1")
    app.state.feature_names = artifact.get("feature_names", FEATURE_COLS)
    yield
    app.state.model = None


app = FastAPI(title="HW5 Stock Return Predictor", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid input payload"})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        cols = request.app.state.feature_names
        row = {col: getattr(data, col) for col in cols}
        features_df = pd.DataFrame([row], columns=cols)
        if features_df.shape != (1, len(FEATURE_COLS)):
            raise ValueError("Invalid feature shape")

        pred = float(model.predict(features_df)[0])
        if not math.isfinite(pred):
            raise RuntimeError("Non-finite prediction returned")

        return OutputData(
            prediction=pred,
            model_version=request.app.state.model_version,
        )
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Inference error")
