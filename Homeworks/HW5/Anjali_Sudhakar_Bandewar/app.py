"""FastAPI service for the fitted HW4 stock-return model (see README for the contract).

One request = one stock on one feature date. The response is the model's predicted
cross-sectional percentile rank (0 = worst, 1 = best) of the stock's 90-calendar-day forward
return, entering at the next session's close. Model logic lives in hw4_model.py.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import hw4_model

# Momentum = adj_close / adj_close k sessions earlier - 1, as a decimal (0.10 = +10%).
# Prices are positive, so momentum must be > -1. null = history too short (missing in HW4).
Momentum = Annotated[float, Field(gt=-1.0, allow_inf_nan=False)] | None
# Fundamental ratios (decimal, unitless) are unbounded in HW4 data (e.g. negative equity
# makes debt/equity negative); the learned QuantileClipper bounds them. null = no eligible
# filing (missing in HW4).
Ratio = Annotated[float, Field(allow_inf_nan=False)] | None


class InputData(BaseModel):
    """Exactly the 11 HW4 features. All are required keys; nullable ones may be null.

    strict=True: numeric strings ("0.1") and booleans are rejected rather than coerced.
    JSON integers are accepted for float fields.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    momentum_1m: Momentum  # 21-session return
    momentum_3m: Momentum  # 63-session return
    momentum_6m: Momentum  # 126-session return
    momentum_12m: Momentum  # 252-session return
    # Dated quarterly top-2,000 universe rank by size (1 = largest); never missing in HW4.
    universe_rank: Annotated[int, Field(ge=1, le=2000)]
    profit_margin: Ratio  # net_income / revenue
    ebitda_margin: Ratio  # ebitda / revenue
    debt_to_equity: Ratio  # total_debt / total_equity
    asset_turnover: Ratio  # revenue / total_assets
    fcf_margin: Ratio  # free_cash_flow / revenue
    # Days from fiscal_date to feature date; HW4 only used filings <= 548 days old.
    fundamental_age_days: Annotated[float, Field(ge=0, le=548, allow_inf_nan=False)] | None


class OutputData(BaseModel):
    prediction: Annotated[float, Field(allow_inf_nan=False)]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted pipeline once per process; never fit on requests.
    app.state.model = hw4_model.load_pipeline()
    app.state.model_version = hw4_model.MODEL_VERSION
    yield
    # Nothing holds external resources (no DB or files stay open); drop the reference.
    app.state.model = None


app = FastAPI(title="HW4 stock-return model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 body echoes the offending input; a NaN/Infinity literal cannot be
    # serialized as JSON, which would turn a validation error into a 500. Return only the
    # field location, error type and message.
    detail = [{k: err[k] for k in ("type", "loc", "msg")} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # Pydantic has already rejected invalid input with 422 before this runs. Inference errors
    # are not caught: they propagate to a plain 500, with the traceback in the server log.
    prediction = hw4_model.predict_one(request.app.state.model, data.model_dump())
    return OutputData(prediction=prediction, model_version=request.app.state.model_version)
