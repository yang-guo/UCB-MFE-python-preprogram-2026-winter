"""HW5: an API for my HW4 model.

You send the 13 features of one stock. The API sends back the prediction of my HW4 model.
The structure comes from the Lesson 9 demo. The model code is in hw4_model.py.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from hw4_model import load_bundle, predict_return

# The folder with the model. It is next to this file.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
logger = logging.getLogger("uvicorn.error")

# Rules for the inputs. The README explains them.
# "| None" means the value can be null (unknown).
Return = Annotated[float, Field(ge=-1, le=100)]  # a price cannot lose more than 100%
Ratio = Annotated[float, Field(ge=-1e6, le=1e6)]
Sector = Literal[
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
    "Unknown",  # used in HW4 when the sector is not known
]


class InputData(BaseModel):
    """The 13 HW4 features, in the HW4 order. All are required."""

    # extra="forbid": no extra field
    # strict=True: no text or true/false instead of a number
    # allow_inf_nan=False: no NaN, no Infinity
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: Return
    momentum_63: Return
    momentum_252_21: Return | None
    volatility_63: Annotated[float, Field(ge=0, le=50)]
    drawdown_252: Annotated[float, Field(ge=-1, le=0)] | None
    net_margin: Ratio | None
    operating_roa: Ratio | None
    debt_to_assets: Annotated[float, Field(ge=0, le=1e6)] | None
    cash_to_assets: Ratio | None
    fcf_to_assets: Ratio | None
    accruals_to_assets: Ratio | None
    filing_age_days: Annotated[float, Field(ge=0, le=3650)] | None
    sector: Sector


class OutputData(BaseModel):
    prediction: FiniteFloat  # a real number: no NaN, no Infinity
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model one time, when the server starts.
    app.state.bundle = load_bundle(ARTIFACT_DIR / "hw4_ridge.joblib")
    yield
    # The server stops.
    del app.state.bundle


app = FastAPI(title="HW5 - Mila Marsot - HW4 Ridge model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Bad input: answer 422, with where and why.
    # I do not copy the bad value: FastAPI crashes when the value is NaN.
    errors = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # Here, the input is always good.
    bundle = request.app.state.bundle
    try:
        prediction = predict_return(bundle["pipeline"], data.model_dump())
        return OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # The model failed: full error in the log, short message for the user.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
