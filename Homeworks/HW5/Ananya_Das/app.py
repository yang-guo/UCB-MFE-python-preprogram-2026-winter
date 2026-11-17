"""HW5 service: serve my HW4 stock-return model through POST /predict (Ananya Das).

Input contract (see README.md): the eight HW4 features, in HW4's order, each a cross-sectional
percentile rank in [0, 1] or null when HW4 had no value. Every key is required, extra keys
are rejected, and at least one feature must be a number. Invalid requests get HTTP 422 and
never reach the model. Model logic lives in model.py; this file only handles HTTP.

Run locally: python -m uvicorn app:app --host 127.0.0.1 --port 8000
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from model import StockReturnModel

logger = logging.getLogger("uvicorn.error")

# A percentile rank: a finite number in [0, 1]. "| None" allows an explicit null; the fields
# below have no default, so every key must still be present in the request.
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


def rank(description: str):
    """A required feature that is a percentile rank or null."""
    return Annotated[UnitInterval | None, Field(description=description)]


class InputData(BaseModel):
    """One stock on one signal date, as HW4's FeatureBuilder.features_on() ranked it.

    Each value is the stock's percentile rank among that date's eligible stocks
    (pandas rank(pct=True)). null means HW4 had no value; the model fills it with 0.5.
    """

    # strict: no "0.5" strings or true/false as numbers; allow_inf_nan: no NaN/Infinity.
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "mom_21": 0.8199174406604747,
                    "mom_252_21": 0.4978969505783386,
                    "vol_63": 0.25748194014447884,
                    "drawdown_252": 0.8934942287513117,
                    "operating_roa": 0.7301341589267286,
                    "accruals_to_assets": 0.8384932920536635,
                    "debt_to_assets": 0.48348813209494323,
                    "log_cap": 0.8555211558307534,
                }
            ]
        },
    )

    mom_21: rank("Rank of the 21-session return")
    mom_252_21: rank("Rank of the return from 252 to 21 sessions ago")
    vol_63: rank("Rank of annualized volatility over 63 sessions")
    drawdown_252: rank("Rank of price / 252-session high - 1")
    operating_roa: rank("Rank of operating income / total assets (latest filing)")
    accruals_to_assets: rank("Rank of (net income - operating cash flow) / total assets")
    debt_to_assets: rank("Rank of total debt / total assets (latest filing)")
    log_cap: rank("Rank of the log market-cap proxy for the quarter")

    @model_validator(mode="after")
    def at_least_one_value(self) -> "InputData":
        if all(getattr(self, name) is None for name in type(self).model_fields):
            raise ValueError("at least one feature must be a number, not null")
        return self


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted 90-day return minus that date's median stock, as a fraction"
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted model once per server process; requests reuse it."""
    model = StockReturnModel.load()
    if model.features != list(InputData.model_fields):
        raise RuntimeError("request schema does not match the model's feature order")
    app.state.model = model
    logger.info("model_loaded version=%s", model.model_version)
    yield
    del app.state.model


app = FastAPI(title="HW5 stock-return model (HW4 Ridge)", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 listing what failed and where, without echoing the submitted values.

    FastAPI's default handler echoes the input, and a NaN/Infinity input cannot be written
    as JSON, which would turn a validation error into a 500.
    """
    detail = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model_version": request.app.state.model.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    """Score one validated stock with the loaded HW4 pipeline."""
    model: StockReturnModel = request.app.state.model
    request_id = uuid.uuid4().hex[:12]
    try:
        result = model.predict(data.model_dump())
        return OutputData(prediction=result.value, model_version=result.model_version)
    except Exception as exc:
        # Details go to the server log; the client only gets a generic message and an id.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
