"""FastAPI service for the fitted HW4 stock-return model."""

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent / "hw4_model.joblib"

FEATURES = ["return_20d", "return_60d", "volatility_20d"]


class InputData(BaseModel):
    """Features used by the original HW4 model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    return_20d: float = Field(
        strict=True,
        ge=-1.0,
        le=10.0,
        allow_inf_nan=False,
        description="20-session adjusted-price return, decimal units.",
    )

    return_60d: float = Field(
        strict=True,
        ge=-1.0,
        le=10.0,
        allow_inf_nan=False,
        description="60-session adjusted-price return, decimal units.",
    )

    volatility_20d: float = Field(
        strict=True,
        ge=0.0,
        le=10.0,
        allow_inf_nan=False,
        description="20-session daily return volatility, decimal units.",
    )


class OutputData(BaseModel):
    """Prediction returned by the API."""

    model_config = ConfigDict(extra="forbid")

    prediction: float = Field(allow_inf_nan=False)
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted fitted model once at application startup."""

    artifact = joblib.load(ARTIFACT_PATH)

    if artifact["features"] != FEATURES:
        raise RuntimeError("Saved model feature order does not match API.")

    app.state.artifact = artifact

    yield

    del app.state.artifact


app = FastAPI(
    title="HW5 Stock Return Prediction API",
    description="Serves the fitted HW4 Ridge regression model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Predict the 90-calendar-day stock return."""

    artifact = request.app.state.artifact

    try:
        # Preserve the exact feature order and DataFrame representation
        # used when fitting the HW4 model.
        features = pd.DataFrame(
            [
                [
                    data.return_20d,
                    data.return_60d,
                    data.volatility_20d,
                ]
            ],
            columns=FEATURES,
        )

        # Apply the original HW4 learned clipping thresholds.
        clipped_features = features.clip(
            lower=artifact["feature_lower"],
            upper=artifact["feature_upper"],
            axis=1,
        )

        # The saved pipeline already contains the fitted
        # StandardScaler and Ridge regression model.
        prediction = float(artifact["model"].predict(clipped_features)[0])

        if not math.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction.")

        return OutputData(
            prediction=prediction,
            model_version=artifact["model_version"],
        )

    except Exception as exc:
        # Detailed error goes to server logs, not the HTTP response.
        LOGGER.exception("Unexpected model inference failure")

        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        ) from exc
