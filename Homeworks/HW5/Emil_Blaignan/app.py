"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("uvicorn.error")


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # TODO: Define required features and enforce the documented contract.

    from pydantic import Field

    model_config.update(strict=True, allow_inf_nan=False)

    momentum_21: float = Field(ge=-1, le=5)
    momentum_63: float = Field(ge=-1, le=10)
    momentum_252_21: float = Field(ge=-1, le=100)
    volatility_63: float = Field(ge=0, le=10)
    drawdown_252: float = Field(ge=-1, le=0)
    net_margin: float = Field(ge=-1000, le=1000)
    operating_roa: float = Field(ge=-100, le=100)
    debt_to_assets: float = Field(ge=0, le=1000)
    cash_to_assets: float = Field(ge=-1, le=100)
    fcf_to_assets: float = Field(ge=-10, le=10)
    accruals_to_assets: float = Field(ge=-100, le=100)
    beta_mkt: float = Field(ge=-5, le=5)
    beta_mom: float = Field(ge=-10, le=10)
    resid_mom: float = Field(ge=-10, le=10)
    idisc: float = Field(ge=-1, le=1)
    adv_log: float = Field(ge=10, le=30)

    del Field


class OutputData(BaseModel):
    # TODO: Define a finite prediction and a model_version field.
    model_config = ConfigDict(allow_inf_nan=False)

    prediction: float
    model_version: str
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: Load the trusted fitted pipeline once and store it on app.state.
    # Resolve artifact paths relative to this file, not the caller's directory.
    from pathlib import Path

    import joblib

    model_path = Path(__file__).resolve().with_name("model.joblib")
    artifact = joblib.load(model_path)
    feature_names = artifact["features"]

    if set(feature_names) != set(InputData.model_fields):
        raise RuntimeError("The input fields do not match the saved model")

    app.state.pipeline = artifact["pipeline"]
    app.state.feature_names = feature_names
    app.state.model_version = artifact["model_version"]

    yield
    # TODO: Release any resources that need explicit cleanup.


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # Return 422 without echoing the submitted values. FastAPI's default response includes
    # each bad input, and a NaN/inf value cannot be encoded as JSON, which would turn this
    # client error into a 500.
    errors = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # TODO: Validate shape, preserve training feature order/representation,
    # predict with the saved pipeline, and return OutputData.
    # Unexpected inference failures must remain errors, not default predictions.
    import pandas as pd

    state = request.app.state
    values = data.model_dump()
    row = pd.DataFrame([values], columns=state.feature_names)
    try:
        predictions = state.pipeline.predict(row)
        result = OutputData(
            prediction=float(predictions[0]),
            model_version=state.model_version,
        )
    except Exception as exc:
        # Full traceback goes to the server log; the client gets a generic message only.
        logger.exception("inference_failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc

    return result
    # raise NotImplementedError("Complete the /predict implementation")
