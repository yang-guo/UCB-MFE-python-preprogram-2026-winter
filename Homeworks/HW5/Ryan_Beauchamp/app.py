"""HW5 starter. Complete the contract, model loading, and prediction TODOs."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # TODO: Define required features and enforce the documented contract.


class OutputData(BaseModel):
    # TODO: Define a finite prediction and a model_version field.
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: Load the trusted fitted pipeline once and store it on app.state.
    # Resolve artifact paths relative to this file, not the caller's directory.
    yield
    # TODO: Release any resources that need explicit cleanup.


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=OutputData)
def predict(data: InputData, request: Request) -> OutputData:
    # TODO: Validate shape, preserve training feature order/representation,
    # predict with the saved pipeline, and return OutputData.
    # Unexpected inference failures must remain errors, not default predictions.
    raise NotImplementedError("Complete the /predict implementation")
