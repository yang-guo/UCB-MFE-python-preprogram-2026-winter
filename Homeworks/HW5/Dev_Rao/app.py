"""FastAPI service for my HW4 stock-ranking model (request handling only).

Structure adapted from the Lecture 9 demo app (Lectures/Lecture 9/demo/app.py):
strict pydantic contract, artifact loaded once in the lifespan, request id + logged
exception on inference failure. The model logic lives in model_service.py / composite.py.
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from model_service import load_bundle, predict_one

logger = logging.getLogger("uvicorn.error")

# Every feature is a same-date percentile rank among that date's eligible stocks
# (pandas rank(pct=True)), so values lie in (0, 1]: the lowest of n stocks gets 1/n > 0.
RANK_DOC = "same-date percentile rank among eligible stocks, (0, 1]"


def rank(description: str, nullable: bool = True):
    """A required rank field. `None` is allowed where HW4 had missing values; the
    model then averages the features that are present, exactly as in HW4."""
    extra = " Null if the input was unavailable (HW4 kept these as NaN)." if nullable else ""
    return Field(gt=0, le=1, description=f"{description}; {RANK_DOC}.{extra}")


class InputData(BaseModel):
    """15 ranked HW4 features, in training order. All keys are required."""

    model_config = ConfigDict(
        extra="forbid",  # unknown fields -> 422
        strict=True,  # no "0.5" strings, no true/false as numbers
        allow_inf_nan=False,  # NaN / inf -> 422
    )

    # value (+)
    r_ey: float | None = rank("TTM net income / market cap")
    r_bm: float | None = rank("book equity / market cap")
    r_sp: float | None = rank("TTM revenue / market cap")
    r_fcfy: float | None = rank("TTM free cash flow / market cap")
    # quality (+, accruals -)
    r_gp_a: float | None = rank("TTM gross profit / total assets")
    r_roe: float | None = rank("TTM net income / book equity (equity > 0)")
    r_roa: float | None = rank("TTM net income / total assets")
    r_accruals_a: float | None = rank("(TTM net income - TTM operating cash flow) / assets")
    # investment (-)
    r_asset_growth: float | None = rank("year-over-year total asset growth")
    r_share_iss: float | None = rank("year-over-year change in shares outstanding")
    # momentum (+)
    r_mom_12_1: float | None = rank("return from t-252 to t-21 sessions")
    r_sector_mom_126: float = rank(
        "sector median 126-session return (never missing in 1,276,254 HW4 rows)", nullable=False
    )
    r_ni_chg_a: float | None = rank("(net income - same quarter last year) / assets")
    # low risk (-)
    r_ivol_63: float | None = rank("63-session idiosyncratic volatility vs ^SP500TR")
    r_max_ret_21: float | None = rank("max daily return over 21 sessions")


class OutputData(BaseModel):
    prediction: FiniteFloat = Field(
        description="unitless composite score in [-0.5, 0.5]; higher = higher expected "
        "90-calendar-day forward return rank. Not a return forecast."
    )
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trusted artifact once per process (hash-checked against its metadata).
    app.state.bundle = load_bundle()
    logger.info("loaded model %s", app.state.bundle.model_version)
    yield
    del app.state.bundle


app = FastAPI(title="HW5 - HW4 composite stock-ranking model", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the rejected values. FastAPI's default body includes each
    bad `input`, and a NaN/inf input cannot be JSON-encoded, which would turn a
    validation error into a server crash. Not echoing inputs is also safer."""
    detail = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ready", "model_version": request.app.state.bundle.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    bundle = request.app.state.bundle
    request_id = uuid.uuid4().hex[:12]
    try:
        value = predict_one(bundle, data.model_dump())
    except Exception as exc:
        # full traceback stays in the server log; the client gets a generic 500
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return OutputData(prediction=value, model_version=bundle.model_version)
