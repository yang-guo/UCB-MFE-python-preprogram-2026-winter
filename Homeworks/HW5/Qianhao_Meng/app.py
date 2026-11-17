"""HW5 service: the HW4 Ridge comparison model (2014-12-31 fit) behind POST /predict.

Adapted from the Lesson 9 demo app.py (strict Pydantic contract, lifespan loading,
logged 500s). Added here: a validation-error handler that never echoes raw input.
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

import model_service

logger = logging.getLogger("uvicorn.error")
RANK = {"ge": 0, "le": 1}


class InputData(BaseModel):
    """One stock on one signal date: the four same-date percentile ranks computed upstream by HW4."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    rank_momentum_252_21: float = Field(
        **RANK,
        description="Rank of 12-1 momentum among eligible stocks that day, higher is stronger",
    )
    rank_drawdown_252: float = Field(
        **RANK, description="Rank of closeness to the 252-session high, higher is closer"
    )
    rank_volatility_63: float = Field(
        **RANK, description="Rank of 63-session volatility, reversed so higher is calmer"
    )
    rank_operating_roa: float = Field(
        **RANK, description="Rank of operating return on assets, higher is more profitable"
    )


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Expected gross adjusted-price return over 90 calendar days, as a fraction"
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted artifact once per process and check it matches the input contract.
    bundle = model_service.load_bundle()
    if set(bundle["features"]) != set(InputData.model_fields):
        raise ValueError("Artifact features do not match the InputData contract")
    app.state.bundle = bundle
    logger.info("loaded model_version=%s", bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(title="HW5 model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Echoing the raw input would fail to serialize NaN or inf and turn this 422 into a 500.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = model_service.predict_one(bundle, data.model_dump())
        result = OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
