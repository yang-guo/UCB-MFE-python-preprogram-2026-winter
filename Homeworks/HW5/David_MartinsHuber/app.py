"""FastAPI service for a frozen HW4 FF5-inspired stock-return model."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model_service import FEATURE_NAMES, FrozenFF5, SymbolUnavailable

logger = logging.getLogger(__name__)
FactorReturn = Annotated[float, Field(strict=True, ge=-1.0, le=1.0, allow_inf_nan=False)]


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    symbol: Annotated[
        str,
        Field(
            strict=True,
            min_length=1,
            max_length=32,
            pattern=r"^[A-Z0-9][A-Z0-9.\-^/]{0,31}$",
            description="Exact uppercase HW4 ticker; historical snapshot, not today's universe.",
        ),
    ]
    MKT: FactorReturn = Field(
        description="Expected compounded market-factor return, decimal units."
    )
    SMB: FactorReturn = Field(
        description="Expected compounded small-minus-big factor return, decimal units."
    )
    HML: FactorReturn = Field(
        description="Expected compounded high-minus-low factor return, decimal units."
    )
    RMW: FactorReturn = Field(
        description="Expected compounded robust-minus-weak factor return, decimal units."
    )
    CMA: FactorReturn = Field(
        description="Expected compounded conservative-minus-aggressive factor return, decimal units."
    )


class OutputData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prediction: Annotated[float, Field(allow_inf_nan=False)] = Field(
        description="Predicted adjusted stock return before costs over 90 calendar days, decimal units."
    )
    model_version: Annotated[str, Field(min_length=1)]
    symbol: Annotated[str, Field(min_length=1)]
    prediction_horizon_calendar_days: Literal[90]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = FrozenFF5.load(Path(__file__).resolve().parent / "artifacts")
    try:
        yield
    finally:
        del app.state.model


app = FastAPI(
    title="HW4 FF5 universe return snapshot",
    lifespan=lifespan,
    description="Frozen per-stock FF5-inspired OLS; 90-calendar-day gross adjusted return.",
)


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        # Explicit training order regardless of JSON key order.
        features = [getattr(data, name) for name in FEATURE_NAMES]
        return OutputData(**request.app.state.model.predict(data.symbol, features))
    except SymbolUnavailable as exc:
        detail = {
            "code": exc.code,
            "symbol": exc.symbol,
            "snapshot_date": request.app.state.model.snapshot_date,
            "model_version": request.app.state.model.model_version,
        }
        if exc.reason:
            detail["reason"] = exc.reason
        raise HTTPException(status_code=404, detail=detail) from None
    except Exception:
        logger.exception("FF5 inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Omit invalid values so NaN/Infinity cannot break JSON error serialization.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})
