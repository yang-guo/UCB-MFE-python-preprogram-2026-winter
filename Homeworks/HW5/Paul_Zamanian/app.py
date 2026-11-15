"""HW5: serve my HW4 version 2 model (gradient boosting) with FastAPI."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "artifacts", "hw4_v2_boosting_2026.joblib")
MODEL_VERSION = "hw4-v2-boosting-fit-2026-01-02"
logger = logging.getLogger("uvicorn.error")

# HW4 ranks are rank(pct=True) - 0.5, so they lie in (-0.5, 0.5]. A missing raw value is 0.
Rank = Annotated[float, Field(ge=-0.5, le=0.5)]


class InputData(BaseModel):
    """One stock on one signal date: its rank on each HW4 feature among that day's
    eligible stocks, in the order the model was trained with."""

    # strict: "0.1" or true is rejected instead of being turned into a number.
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    ret_1m: Rank  # 1-month return
    ret_12_1: Rank  # 12-1 month momentum
    vol_3m: Rank  # 3-month volatility of daily returns
    dist_high_1y: Rank  # distance to the 1-year high
    size: Rank  # market-cap proxy
    gross_profitability: Rank  # gross profit / total assets
    roa: Rank  # net income / total assets
    debt_to_assets: Rank  # total debt / total assets


class OutputData(BaseModel):
    prediction: FiniteFloat  # predicted 90-day excess return, as a fraction (0.01 = 1%)
    model_version: str


def predict_one(model, features):
    # One-row DataFrame with the training column names, like the HW4 rows() table.
    X = pd.DataFrame([features])
    return float(model.predict(X)[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fitted HW4 model once per server process.
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # The default 422 echoes the input, and JSON can't encode NaN, so a NaN input became a
    # 500. Return the errors without the input values instead.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        prediction = predict_one(request.app.state.model, data.model_dump())
        return OutputData(prediction=prediction, model_version=MODEL_VERSION)
    except Exception:
        logger.exception("inference failed")  # details stay in the server log
        raise HTTPException(status_code=500, detail="Prediction failed")
