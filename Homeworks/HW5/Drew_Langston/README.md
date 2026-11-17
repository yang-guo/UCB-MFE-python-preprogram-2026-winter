# HW5 — Drew Langston

FastAPI service for my HW4 stock-return Ridge model, version
`hw4-ridge-2026-v1`.

## Model and artifact

`artifacts/model.joblib` contains the fitted Ridge model (alpha 0.1),
numeric imputation and scaling, and sector encoding in one pipeline.
It recreates the HW4 annual model used for 2026 predictions from
`hw4_stock_strategy_clean.ipynb`, using the original feature construction,
training selection, and pipeline settings.

The fit date is December 31, 2025, with 365,800 training rows and a latest
training exit of December 30, 2025. Predictions are return fractions for
HW4's 90-calendar-day holding horizon, using its next-session entry rule.

The artifact is included in this folder. The service loads it once during
startup, without retraining or accessing the HW4 database. No custom
transformer modules are needed. Model provenance and training-library
versions are recorded in `artifacts/model_metadata.json`; matching versions
are pinned in `requirements.txt` for local use and Docker.

`artifacts/reference_prediction.json` records an HW4 pipeline prediction
before API implementation. The example is symbol A on December 31, 2025,
with expected prediction -0.25636256564901405. The export also checked that
reloading the artifact preserved all 17,840 predictions in the 2026 batch.

## Input and output contract

`POST /predict` accepts one JSON object containing all seven named fields.
The service constructs a DataFrame in the following model feature order;
JSON key order itself does not matter.

| Feature | Type and allowed values | Definition / units |
| --- | --- | --- |
| momentum_126 | Number >= -1, or null | 126-session return fraction |
| momentum_252_21 | Number >= -1, or null | Return fraction from 252 to 21 sessions ago |
| volatility_63 | Number >= 0 | Annualized volatility from 63 daily returns |
| operating_roa | Finite number, or null | Quarterly operating income / total assets |
| accruals_to_assets | Finite number, or null | Quarterly (net income - operating cash flow) / assets |
| debt_to_assets | Finite number, or null | Total debt / total assets |
| sector | String, 1–100 characters | Sector label |

All numeric values must be finite. Numeric strings, booleans, extra fields,
missing fields, and an array in place of the object are rejected with 422.
Nullable fields must still be present; null uses the saved numeric imputer.

The -1 lower bound reflects a simple return's minimum. Volatility cannot be
negative. Financial ratios have no arbitrary upper or lower bounds because
extreme or negative accounting values can occur. Missing features retain
HW4's imputation behavior. Unseen sector labels use the saved encoder's
`handle_unknown="ignore"` behavior.

A valid request returns 200 with a finite numeric `prediction` and string
`model_version`, for example:

```json
{"prediction": -0.25636256564901394, "model_version": "hw4-ridge-2026-v1"}
```

Unexpected inference failures return 500 with
`{"detail": "Prediction unavailable"}`; detailed exceptions go to server
logs. Validation responses include field locations, messages, and error
types, omitting raw inputs so NaN/infinity cannot break JSON serialization.

## Local setup and checks

From this submission folder, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make lint
```

`make format` applies Ruff formatting. Without make, use
`python -m pytest -q`, `python -m ruff check .`, and
`python -m ruff format --check .`.

To run locally:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Interactive documentation is at http://127.0.0.1:8000/docs.
Stop the server with Control+C before running the container on port 8000.

## Docker

Install and start Docker, then run from this submission folder:

```bash
make build
make run
```

These wrap the assignment's commands:

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The image uses `python:3.12-slim`, installs requirements, and copies the API
and artifacts into `/app`. The build context excludes local environments,
Git files, environment files, and DuckDB databases.

In another Terminal, from this folder, send a valid request:

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Send an invalid request:

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary '{}'
```

Stop the container with Control+C, run `make run` again without rebuilding,
and repeat the valid request to verify the model is available after restart.

## Results and evidence

All 63 tests passed. Tests cover the real saved pipeline, response schema,
agreement with a direct call and the pre-API reference, invalid inputs that
never reach inference, missing-value imputation, and a simulated inference
failure. Prediction comparisons use relative tolerance 1e-10 and absolute
tolerance 1e-12 to allow floating-point differences.

The performance test uses sequential in-process TestClient requests timed
with `time.perf_counter`: 5 warmups followed by 30 measured successful
requests. The recorded median is 1.712 ms and p95 is 1.843 ms on macOS
26.6.2, ARM64, 10 logical CPUs, Python 3.12.9. These are not real HTTP
latencies; no universal latency cutoff is asserted.

Saved evidence:

- `evidence/pytest.txt`: 63 passing tests; one dependency deprecation warning.
- `evidence/lint.txt`: passing Ruff lint and formatting checks.
- `evidence/latency.json`: timing method, hardware, and results.
- `evidence/build.txt`: successful Docker build.
- `evidence/container_valid.txt`: real HTTP 200, prediction, and model version.
- `evidence/container_invalid.txt`: real HTTP 422 for missing features.
- `evidence/container_after_restart.txt`: HTTP 200 with the same prediction
  and version after restarting the same image.

## Before a real deployment

Add authentication and TLS, request-size and rate limits, health checks,
centralized logs and monitoring, and load tests with realistic concurrency.
Run the container as a non-root user, scan and update dependencies, and
version and verify trusted model artifacts. Monitor feature quality and
drift, and define a reviewed model update and rollback process.

## Sources

Application structure, lifespan loading, test fixtures, Dockerfile, Makefile,
and Ruff configuration are adapted from the course HW5 template and
Lecture 9 demo. The fitted model and feature definitions come from my HW4
experiment.

- [HW5 assignment](../hw_5.md)
- [HW5 template](../template/)
- [Lecture 9 demo](../../../Lectures/Lecture%209/demo/)
