"""HW5 service: the HW4 round 1 Ridge return model behind POST /predict.

Adapted from the HW5 template and the Lecture 9 demo app (lifespan loading, strict input
model, and the 500 response for an inference failure).
"""

import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from inference import load_bundle, predict_return

logger = logging.getLogger("uvicorn.error")

# Accounting ratios have very long tails when revenue or assets are near zero (training
# values reach about 950,000). The pipeline clips each feature at its 1% and 99% training
# quantiles, so this limit only refuses values that cannot be a real ratio.
RATIO_LIMIT = 1_000_000


class InputData(BaseModel):
    """One stock on one signal date. All 12 features are required, in the training order.

    Returns and ratios are fractions (0.05 = 5%), not percent. `strict=True` refuses
    numeric strings and booleans. `allow_inf_nan=False` refuses NaN and infinity.
    `extra="forbid"` refuses unknown fields, so a wrong feature count is always a 422.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    momentum_21: float = Field(
        ge=-1, le=5, description="Return over the last 21 sessions. A price cannot fall >100%."
    )
    momentum_63: float = Field(
        ge=-0.8, le=5, description="Return over the last 63 sessions. Limits = HW4 screen rule."
    )
    momentum_252_21: float = Field(
        ge=-1, le=100, description="Return from 252 to 21 sessions ago (12-1 momentum)."
    )
    mom63_vs_sector: float = Field(
        ge=-6, le=6, description="momentum_63 minus the same-date median of the sector."
    )
    mom21_vs_sector: float = Field(
        ge=-6, le=6, description="momentum_21 minus the same-date median of the sector."
    )
    net_margin: float = Field(
        ge=-RATIO_LIMIT, le=RATIO_LIMIT, description="Net income / revenue, latest filing."
    )
    operating_roa: float = Field(
        ge=-RATIO_LIMIT, le=RATIO_LIMIT, description="Operating income / total assets."
    )
    debt_to_assets: float = Field(
        ge=0, le=RATIO_LIMIT, description="Total debt / total assets. Debt is not negative."
    )
    cash_to_assets: float = Field(
        ge=-RATIO_LIMIT, le=RATIO_LIMIT, description="Cash / total assets."
    )
    fcf_to_assets: float = Field(
        ge=-RATIO_LIMIT, le=RATIO_LIMIT, description="Free cash flow / total assets."
    )
    accruals_to_assets: float = Field(
        ge=-RATIO_LIMIT, le=RATIO_LIMIT, description="Accruals / total assets."
    )
    filing_age_days: float = Field(
        ge=0, le=1000, description="Days since the latest filing. Training range: 0 to 615."
    )


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(description="Predicted 90-day return, as a fraction.")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted pipeline one time for each server process. Nothing is fitted here.
    bundle = load_bundle()
    if list(InputData.model_fields) != list(bundle["features"]):
        raise RuntimeError("InputData fields do not agree with the features of the model")
    app.state.bundle = bundle
    yield
    del app.state.bundle


app = FastAPI(title="HW5: HW4 return model service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    """Return 422 with the location and reason of each problem, without the input value.

    The default FastAPI handler copies the input into the response. A NaN or infinite input
    cannot be written as JSON, so the default handler fails and the client gets a 500.
    """
    detail = [
        {"type": item["type"], "loc": list(item["loc"]), "msg": item["msg"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health(request: Request):
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    bundle = request.app.state.bundle
    try:
        prediction = predict_return(bundle, data.model_dump())
        if not math.isfinite(prediction):
            raise ValueError("The model returned a prediction that is not finite")
    except Exception as error:
        # The full error goes to the server log. The client gets no internal details.
        logger.exception("inference_failed")
        raise HTTPException(status_code=500, detail="Prediction unavailable") from error
    return OutputData(prediction=prediction, model_version=bundle["model_version"])
