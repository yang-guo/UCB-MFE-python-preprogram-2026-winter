"""HW5 API: serve the HW4 elastic net. 18 ranked characteristics in, predicted 90-day return out.

Adapted from the HW5 template and the Lesson 9 demo (Lectures/Lecture 9/demo/app.py).
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")

# In HW4 each characteristic is ranked across the stocks of the month, from -0.5 (lowest)
# to +0.5 (highest). So a value outside [-0.5, 0.5] can only be a mistake, we refuse it.
Rank = Annotated[float, Field(ge=-0.5, le=0.5)]


class InputData(BaseModel):
    """The 18 HW4 characteristics of one stock, as ranks, in the training order.

    All fields are required. A missing characteristic is sent as 0 (the middle), like in HW4.
    strict=True: a string like "0.1" or a boolean is refused, it is not turned into a number.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    z_momentum_21: Rank = Field(description="Rank of the 1-month return (21 sessions)")
    z_momentum_63: Rank = Field(description="Rank of the 3-month return (63 sessions)")
    z_momentum_252_21: Rank = Field(description="Rank of the 12-month return, last month out")
    z_volatility_63: Rank = Field(description="Rank of the 63-session volatility")
    z_drawdown_252: Rank = Field(description="Rank of the price / 1-year high - 1")
    z_beta_252: Rank = Field(description="Rank of the 252-session beta vs the S&P 500 TR")
    z_log_dollar_volume_63: Rank = Field(description="Rank of the 63-day dollar volume")
    z_net_margin: Rank = Field(description="Rank of net income / revenue")
    z_operating_roa: Rank = Field(description="Rank of operating income / assets")
    z_debt_to_assets: Rank = Field(description="Rank of debt / assets")
    z_cash_to_assets: Rank = Field(description="Rank of cash / assets")
    z_fcf_to_assets: Rank = Field(description="Rank of free cash flow / assets")
    z_accruals_to_assets: Rank = Field(description="Rank of accruals / assets")
    z_asset_growth: Rank = Field(description="Rank of the 1-year growth of assets")
    z_gross_profitability: Rank = Field(description="Rank of gross profit / assets")
    z_book_to_market: Rank = Field(description="Rank of book equity / market value")
    z_earnings_yield: Rank = Field(description="Rank of 12-month net income / market value")
    z_log_size_rank: Rank = Field(description="Rank of the log universe rank (high = smaller)")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted 90-day return minus the average stock of the month, as a fraction"
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # we load the model once when the server starts, never per request
    bundle = load_bundle()
    # the API fields must be the training columns, in the same order
    if list(InputData.model_fields) != bundle["features"]:
        raise RuntimeError("InputData does not match the features of the saved model")
    app.state.bundle = bundle
    yield
    del app.state.bundle


app = FastAPI(title="HW5: HW4 elastic net service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 repeats the bad value. A NaN cannot be written in JSON, so the
    # default answer crashed with a 500. We answer 422 with where and why, without the value.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # here the input is already validated by InputData (422 otherwise)
    bundle = request.app.state.bundle
    try:
        prediction = predict_one(bundle, data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # the details stay in the server logs, the client only sees a 500
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from exc
