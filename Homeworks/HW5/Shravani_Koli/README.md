# HW5 Stock Return Prediction API

A FastAPI application serving the fitted HW4 Ridge regression model to predict 90-calendar-day stock returns.

## Model

- Ridge regression with alpha=100
- Inputs: return_20d, return_60d, volatility_20d
- Output: prediction and model_version
- Model artifact: hw4_model.joblib

## Run with Docker

Build:

docker build -t mfe-hw5:local .

Run:

docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local

Open http://127.0.0.1:8000/docs to test POST /predict.

## Example request

{
  "return_20d": 0.05,
  "return_60d": 0.10,
  "volatility_20d": 0.02
}

## Validation

- 14 automated tests passed.
- Ruff lint and formatting checks passed.
- Docker build and POST /predict succeeded.
- Invalid inputs return HTTP 422.
- Unexpected model failures return HTTP 500.

## Limitations

The model uses only three historical price features. Its predictions are estimates, not guarantees of future returns.

## AI Attribution

AI assisted with debugging, and documentation. The application was tested locally.
