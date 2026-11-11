"""Serve the four selected 2026 HW4 models on one feature vector."""

import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, FiniteFloat, StrictFloat, StrictInt, field_validator

from inference import HW4Models

SCHEMA = json.loads((Path(__file__).parent / "feature_schema.json").read_text())
FEATURE_COUNT = len(SCHEMA["model_features"])
logger = logging.getLogger(__name__)


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[StrictFloat | StrictInt | None]

    @field_validator("features")
    @classmethod
    def check_features(cls, values):
        if len(values) != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} features")
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("Features must be finite numbers or null")
        if any(
            name.startswith("SECTOR_") and value not in (0, 1)
            for name, value in zip(SCHEMA["model_features"], values)
        ):
            raise ValueError("Sector indicators must be 0 or 1")
        return values


class OutputData(BaseModel):
    prediction: FiniteFloat
    model_version: str
    model_scores: dict[str, FiniteFloat]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = HW4Models(Path(__file__).parent)
    yield
    del app.state.model


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError):
    errors = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        scores = request.app.state.model.predict(data.features)
        return OutputData(
            prediction=scores["model_01"],
            model_version=SCHEMA["model_version"],
            model_scores=scores,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
