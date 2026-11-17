import hashlib
import io
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

logging.basicConfig(level=logging.INFO)

# pre-processed: extracted feature schema from course_data.py (HW4) into 
# artifacts/feature_contract.json
# with additional fields such as feature names, units, ranges and data types

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
log = logging.getLogger(__name__)

# Small JSON, read at import so the request schema exists when routes are registered.
CONTRACT = json.loads((ARTIFACTS / "feature_contract.json").read_text())
FEATURE_NAMES = [f["name"] for f in CONTRACT["features"]]
NUMERIC_NAMES = [f["name"] for f in CONTRACT["features"] if f["kind"] == "numeric"]


class _InputBase(BaseModel):
    """Exact model inputs, in training order (see /docs for names, units, and ranges).

    Every feature is required. Numeric features must be JSON numbers (strings are rejected),
    finite (no NaN/Inf), and inside the documented [min, max]. Unknown fields are rejected.
    Field order here is documentation only; the server rebuilds the row in training order.
    """
    model_config = ConfigDict(extra="forbid")


def _field(spec):
    if spec["kind"] == "numeric":
        desc = f"{spec['unit']}. Allowed range [{spec['min']:.6g}, {spec['max']:.6g}]."
        return (float, Field(..., strict=True, allow_inf_nan=False,
                             ge=spec["min"], le=spec["max"], description=desc))
    desc = f"Categorical. Allowed values: {', '.join(spec['categories'])}."
    return (Literal[tuple(spec["categories"])], Field(..., description=desc))


InputData = create_model("InputData", __base__=_InputBase,
                         **{f["name"]: _field(f) for f in CONTRACT["features"]})


class OutputData(BaseModel):
    prediction: float = Field(allow_inf_nan=False,
                              description="Predicted 90-day forward return (training-target units).")
    model_version: str = Field(description="Version of the serialized model that produced this value.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = ARTIFACTS / CONTRACT["model_file"]
    data = path.read_bytes()
    # verify the hash before deserializing
    if hashlib.sha256(data).hexdigest() != CONTRACT["model_sha256"]:
        raise RuntimeError("model artifact does not match feature_contract.json")
    app.state.model = joblib.load(io.BytesIO(data))
    app.state.model_version = CONTRACT["model_version"]
    log.info("loaded model %s", app.state.model_version)
    yield
    app.state.model = None                         # release the reference on shutdown


app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"service": "HW4 stock-return model", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model_version}

@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    try:
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise RuntimeError("model not loaded")
        row = data.model_dump()
        # Same representation as training: a DataFrame with the training column names and order.
        X = pd.DataFrame([row], columns=FEATURE_NAMES).astype({n: "float64" for n in NUMERIC_NAMES})
        pred = float(model.predict(X)[0])
        if not math.isfinite(pred):
            raise ValueError(f"non-finite prediction: {pred}")
    except Exception:
        log.exception("inference failed")          # full traceback stays in server logs
        raise HTTPException(status_code=500, detail="Internal inference error") from None
    return OutputData(prediction=pred, model_version=request.app.state.model_version)