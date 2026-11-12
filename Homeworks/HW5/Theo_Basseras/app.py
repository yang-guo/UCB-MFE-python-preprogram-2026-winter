"""Serve the fitted HW4 model through a FastAPI endpoint."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Let's define both variables as floats, but we will validate with
    #  Field that they are finite and do not convert automatically string as required by the milestone 1.
    momentum_63: float = Field(strict=True, gt=-1, le=10, allow_inf_nan=False)
    volatility_63: float = Field(strict=True, ge=0, le=10, allow_inf_nan=False)


class OutputData(BaseModel):
    prediction: float = Field(allow_inf_nan=False)
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # We load the fitted scaler and Ridge model once at startup
    hw5_dir = Path(__file__).resolve().parent

    # Now we load the model
    app.state.model_full = joblib.load(hw5_dir / "model.joblib")

    # We load the reference example and prediction from HW4
    reference = json.loads((hw5_dir / "hw4_reference.json").read_text(encoding="utf-8"))
    app.state.model_version = reference["model_version"]

    print("HW4 model loaded successfully.")
    yield


# Since it is required to return 422 for invalid input, we override
# the default validation error handler to return only the validation messages without the invalid input values.
app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    # We return the validation messages without the invalid input values
    details = []

    for problem in error.errors():
        details.append(
            {
                "loc": problem["loc"],
                "msg": problem["msg"],
                "type": problem["type"],
            }
        )

    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Return a prediction from the fitted model for the given input data."""

    features = ["momentum_63", "volatility_63"]
    data_features = pd.DataFrame([data.model_dump()])[features]

    prediction = request.app.state.model_full.predict(data_features)[0]

    return OutputData(
        prediction=float(prediction), model_version=request.app.state.model_version
    )
