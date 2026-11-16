"""Serve the unchanged HW4 pipeline; adapted from the HW5 starter and Lesson 9."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import InputData, OutputData
from serving import load_pipeline, predict_one

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the trusted artifact once per process, before accepting requests."""
    pipeline, metadata = load_pipeline()
    application.state.pipeline = pipeline
    application.state.model_version = metadata["model_version"]
    try:
        yield
    finally:
        del application.state.pipeline
        del application.state.model_version


app = FastAPI(title="Sterling Couts HW4 prediction service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Return 422 even when a nonstandard JSON body contains NaN or Infinity.

    Avoid echoing the invalid value: JSONResponse cannot encode non-finite
    numbers, and client inputs do not belong in public error diagnostics.
    """
    errors = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request):
    """Successful startup has verified and loaded the artifact."""
    return {"status": "ready", "model_version": request.app.state.model_version}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request, response: Response) -> OutputData:
    """Validate first, reuse fitted transforms, and expose a finite return forecast."""
    request_id = uuid4().hex
    try:
        prediction = predict_one(request.app.state.pipeline, data.model_dump())
        result = OutputData(
            prediction=prediction, model_version=request.app.state.model_version
        )
    except Exception as exc:
        logger.exception("inference_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction unavailable",
            headers={"X-Request-ID": request_id},
        ) from exc
    response.headers["X-Request-ID"] = request_id
    return result
