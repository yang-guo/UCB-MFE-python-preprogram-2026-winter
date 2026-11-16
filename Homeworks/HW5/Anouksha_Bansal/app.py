"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from hw4_model import FEATURES, load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")

# A same-date percentile rank minus 0.5, so the training values lie in (-0.5, 0.5]. Anything outside
# [-0.5, 0.5] cannot be a rank and is rejected; `null` means the raw feature was
# unavailable (e.g. no fundamentals) and is imputed as 0 (the median), as in HW4.
Rank = Annotated[
    float,
    Field(ge=-0.5, le=0.5, description="Same-date percentile rank minus 0.5 (median = 0)"),
]


class InputData(BaseModel):
    """Seven ranked signals for one stock on one weekly signal date (HW4 `CORE` order).

    All fields are required (send `null` explicitly for an unavailable feature).
    `strict=True` stops "0.1" or `true` from being coerced into numbers, and
    `allow_inf_nan=False` rejects NaN/Infinity.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    r_gross_profitability: Rank | None = Field(description="Gross profit / total assets")
    r_operating_roa: Rank | None = Field(description="Operating income / total assets")
    r_fcf_to_assets: Rank | None = Field(description="Free cash flow / total assets")
    r_earnings_yield: Rank | None = Field(description="4 x quarterly net income / market cap")
    r_sales_to_price: Rank | None = Field(description="4 x quarterly revenue / market cap")
    r_sales_growth_yoy: Rank | None = Field(
        description="Year-over-year quarterly revenue growth"
    )
    r_mom_252_21: Rank | None = Field(description="12-1 month price momentum")


assert list(InputData.model_fields) == FEATURES

class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="Predicted 90-day return minus the screened-universe mean, in decimal "
        "units (0.01 = 1 percentage point)"
    )
    model_version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted fitted pipeline once and store it on app.state.
    app.state.bundle = load_bundle()
    logger.info("loaded model_version=%s", app.state.bundle["model_version"])
    yield
    del app.state.bundle


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 for missing required fields, invalid types, extra fields, non-finite values, 
       and the wrong feature count. 
    """
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ready", "model_version": request.app.state.bundle["model_version"]}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        prediction = predict_one(bundle["pipeline"], data.model_dump())
        result = OutputData(prediction=prediction, model_version=bundle["model_version"])
    except Exception as exc:
        # Full traceback goes to the server log; the client gets a generic 500.
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
