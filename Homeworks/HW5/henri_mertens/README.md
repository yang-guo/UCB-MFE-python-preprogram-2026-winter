# HW5 - Henri Mertens

The service predicts a stock's 90-calendar-day adjusted-price return using the
frozen XGBoost model from my HW4 experiment. It loads the fitted model once at
startup and returns the prediction in decimal units. The example prediction
`0.04004279524087906` corresponds to approximately 4.0043%.

## Model and preprocessing

The included `artifacts/hw4_frozen.joblib` is the 127,480-byte artifact from HW4
commit `fca22b5cc90d70c0df6f707aaba30eed7185ceba`. Its SHA-256 is
`eeda5faa3f14cc59c718c0471475172dcd245b79c57355df1365eb3b245e5f23`.
The API reports version `hw4-fca22b5-xgboost-eeda5faa3f14`.

`model_service.py` follows the fitted HW4 inference sequence.

1. Arrange the 12 named inputs in their training order.
2. Apply the learned 1st/99th-percentile clipping bounds.
3. Apply the fitted median imputer and standard scaler.
4. Select the eight features retained by the fitted Boruta mask.
5. Predict with XGBoost and reverse the training target's scaling.

The selected features are `momentum_21`, `momentum_63`, `momentum_252_21`,
`volatility_63`, `drawdown_252`, `operating_roa`, `cash_to_assets`, and
`filing_age_days`. The full 12-column contract preserves the input representation
expected by the fitted preprocessing objects.

The model stack in `requirements.txt` matches the saved HW4 environment,
including NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1 and XGBoost 3.4.1.

## Input and output contract

`POST /predict` accepts one JSON object containing all 12 fields below. All
values must be finite JSON numbers. Numeric strings, booleans, nulls, missing
fields, extra fields and arrays are rejected with HTTP 422. Integer-valued
numbers are accepted. JSON field order can vary because the service explicitly
restores the training column order.

| Field, in training order | Meaning and units | Allowed range |
| --- | --- | --- |
| `momentum_21` | Adjusted-price return over 21 trading sessions, decimal | At least -1 |
| `momentum_63` | Adjusted-price return over 63 trading sessions, decimal | At least -1 |
| `momentum_252_21` | Adjusted-price return from 252 to 21 sessions ago, decimal | At least -1 |
| `volatility_63` | Standard deviation of 63 daily returns, annualized by sqrt(252), decimal | At least 0 |
| `drawdown_252` | Current adjusted price divided by its 252-session high, minus 1 | Between -1 and 0 |
| `net_margin` | Quarterly net income / revenue | Any finite number |
| `operating_roa` | Quarterly operating income / total assets | Any finite number |
| `debt_to_assets` | Total debt / total assets | Any finite number |
| `cash_to_assets` | Cash and equivalents / total assets | Any finite number |
| `fcf_to_assets` | Quarterly free cash flow / total assets | Any finite number |
| `accruals_to_assets` | (Quarterly net income - operating cash flow) / total assets | Any finite number |
| `filing_age_days` | Days since the filing's HW4 availability date, which is filing date + 1 day | At least 0 |

Return and drawdown bounds follow their definitions. Volatility and elapsed days
are nonnegative. The financial ratios retain the signed values used in HW4,
with extreme finite inputs handled by the fitted clipping bounds.

A valid request returns HTTP 200 with exactly `prediction` and `model_version`.
Unexpected inference failures return HTTP 500 with a generic error message and
record the exception in the server log. Validation errors include the affected
field, error type and explanation. Their raw input is omitted so that rejecting
NaN or infinity still produces a valid JSON error response.

## Run locally

Run these commands from this submission folder. Python 3.12.9 was used for the
recorded checks.

```powershell
uv venv --python 3.12.9 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv`,
install with `.venv/bin/python -m pip install -r requirements.txt`, and run
`.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000`.
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Tests and latency

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

The Makefile provides `test`, `lint`, `format`, `build`, and `run` targets.
For example, an activated environment can use `make test` and `make lint`.

The 33 tests cover agreement with a direct model call and the recorded HW4
reference, input validation before inference, expected HTTP 500 behavior, and
latency. They also check that the loaded model object is reused across requests.
`reference_prediction.json` identifies stock A on January 29, 2010 in the HW4
development panel. Its expected result was recorded by running the original
`research.py::predict_candidate` before the API was implemented. Comparisons
allow relative tolerance 1e-6 and absolute tolerance 1e-9.

The performance test warms up with five requests, then measures 30 sequential
successful requests with `time.perf_counter`. The recorded median is 11.56 ms
and p95 is 13.70 ms on Windows 11, Python 3.12.9, Intel64 Family 6 Model 140.
This measurement uses in-process FastAPI TestClient and includes request
handling and inference. Startup and network transport are excluded. The full
environment and measurements are in `evidence/latency.json`.

Passing test output is saved in `evidence/test_results.txt`. The course-pinned
HTTPX dependency produces a Starlette deprecation warning during test import.

## Build and exercise the container

The Dockerfile uses `python:3.12-slim`, installs the pinned dependencies and
the OpenMP runtime required by XGBoost, and copies the application and fitted
artifact into `/app`.

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

From another terminal, send the included example.

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{}'
```

In PowerShell, use `curl.exe` and put each request on one line. To check a
fresh start of the same image, stop the running container and start it again.

```sh
docker stop mfe-hw5
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

Repeat the valid request after restart. The response should contain the same
prediction and model version because the fitted artifact is stored in the image.

Verified on November 14, 2026 using Docker Desktop 4.99.0, Docker Engine 29.9.3
and the WSL 2 Linux backend. The image built successfully. The valid request
returned HTTP 200 with prediction `0.04004279524087906` and model version
`hw4-fca22b5-xgboost-eeda5faa3f14`. An empty JSON object returned HTTP 422.
After stopping and removing the first container, a new container from the same
image returned the identical prediction and model version.

The [request transcript](evidence/container_requests.txt) includes the input,
responses and container commands. The [verification record](evidence/container_verification.json)
records both container IDs, their matching image ID and the successful checks.

## Before a real deployment

- Add authentication, TLS, request-size limits and rate limits appropriate to the users.
- Monitor prediction latency, failures, input drift and changes in realized model performance.
- Track model versions and deployment dependencies, validate replacement models and support rollback.

## Adapted code

The application follows the official HW5 template. The serving transformation
order and target rescaling come from my HW4 `research.py::transform_features`
and `predict_candidate`. `course_data.py` includes the course's `QuantileClipper`
under its original import name so the existing artifact can be loaded. These
sources are available at the HW4 commit.

- [HW4 inference code](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/fca22b5cc90d70c0df6f707aaba30eed7185ceba/Homeworks/HW4/henri_mertens/research.py)
- [Course transformer](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/fca22b5cc90d70c0df6f707aaba30eed7185ceba/Lectures/modeling/course_data.py)
- [FastAPI lifespan tests](https://fastapi.tiangolo.com/advanced/testing-events/)
- [Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/)
