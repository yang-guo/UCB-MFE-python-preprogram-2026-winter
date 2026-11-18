"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
MODEL_VERSION = "hw4-lasso-v1"
SIGNALS = ["book_value", "small_large", "profitability", "investment_aggression", "beta"]
logger = logging.getLogger(__name__)
CLIP = 0.20


class InputData(BaseModel):
    """Five HW4 Fama-French signals for one stock on one date."""

    model_config = ConfigDict(extra="forbid", strict=True)
    book_value: float = Field(ge=-5, le=20, allow_inf_nan=False)
    small_large: float = Field(gt=0, le=1e13, allow_inf_nan=False)
    profitability: float = Field(ge=-5, le=5, allow_inf_nan=False)
    investment_aggression: float = Field(ge=-5, le=5, allow_inf_nan=False)
    beta: float = Field(ge=-5, le=5, allow_inf_nan=False)

    # TODO: Define required features and enforce the documented contract.


class OutputData(BaseModel):
    prediction: float = Field(allow_inf_nan=False)
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH)
    yield
    del app.state.bundle


app = FastAPI(lifespan=lifespan)


def predict_signal_average(models: dict, data: InputData) -> float:
    """Clip each signal's HW4 LassoCV prediction to +/-CLIP, then average the 5."""
    preds = [models[s].predict(pd.DataFrame({"value": [getattr(data, s)]}))[0] for s in SIGNALS]
    return float(np.mean(np.clip(preds, -CLIP, CLIP)))


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # 422: input check (backup to InputData; runs before the model)
    values = [getattr(data, s) for s in SIGNALS]
    if len(values) != len(SIGNALS) or not np.all(np.isfinite(values)):
        raise HTTPException(
            status_code=422, detail="Invalid input: need 5 finite signal values"
        )

    # 500: model failure
    try:
        prediction = predict_signal_average(request.app.state.bundle, data)
        if not np.isfinite(prediction):
            raise ValueError("non-finite prediction")
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")

    # 200: success
    return OutputData(prediction=prediction, model_version=MODEL_VERSION)
