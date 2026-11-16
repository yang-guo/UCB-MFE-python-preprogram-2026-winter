import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model_utils import load_model, predict_one

logger = logging.getLogger("hw5")

SECTORS = Literal[
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]

# strict=True: numeric strings ("0.1") and booleans are rejected, not converted.
# allow_inf_nan=False: NaN and +/-inf are rejected.
Momentum = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1, le=100)]
Ratio = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1000, le=1000)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # price features (always available in HW4)
    momentum_21: Momentum
    momentum_63: Momentum
    volatility_63: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=100)]
    # can be missing in HW4 (short history / no reports): null is allowed and imputed
    momentum_252_21: Momentum | None
    drawdown_252: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1, le=0)] | None
    net_margin: Ratio | None
    operating_roa: Ratio | None
    debt_to_assets: Ratio | None
    cash_to_assets: Ratio | None
    fcf_to_assets: Ratio | None
    accruals_to_assets: Ratio | None
    filing_age_days: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=3650)]
    sector: SECTORS


class OutputData(BaseModel):
    prediction: Annotated[float, Field(allow_inf_nan=False)]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_version = load_model()
    logger.info("Loaded model %s", app.state.model_version)
    yield
    app.state.model = None


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # the default 422 body echoes the input, which crashes on NaN/inf (not valid JSON)
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        prediction = predict_one(request.app.state.model, data.model_dump())
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")
    return OutputData(prediction=prediction, model_version=request.app.state.model_version)
