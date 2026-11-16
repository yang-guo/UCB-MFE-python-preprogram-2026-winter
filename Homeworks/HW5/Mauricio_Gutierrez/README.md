# HW5: serve the HW4 stock-return model

Author: Mauricio Gutierrez. Prepared with AI assistance (Codex, then Claude Code for the Docker work).

## Status

All five milestones are complete with saved evidence.
The local API and required test groups pass. Real localhost HTTP returns 200 for valid input and 422 for invalid input.
The image builds from python:3.12-slim. A running container returns 200 for the valid request and 422 for
the invalid request, and returns the same 200 result after the container is stopped and the same image is started again.
The pull request for HW5 is the only remaining step.

| Milestone | Evidence |
|---|---|
| 1 API | app.py, model_service.py, artifacts/ |
| 2 Tests | tests/test_app.py, evidence/checks.txt, evidence/latency.json |
| 3 Container | Dockerfile, .dockerignore, build log in evidence/container.txt |
| 4 Tooling | Makefile, pyproject.toml, evidence/checks.txt |
| 5 Requests | evidence/container_curl.txt (curl), evidence/container.txt, evidence/container_status.json |

## Goal and system

Serve the HW4 fit dated April 1, 2016. Predict a 90-calendar-day adjusted-price return.
One request contains one stock/date observation. It contains calculated features, not raw prices.

```text
JSON features -> strict validation -> named DataFrame -> fitted pipeline -> JSON result
                     |
                     +-> invalid request: HTTP 422

Server startup -> verify package versions and artifact hash -> load model once
Unexpected inference failure -> server log + HTTP 500
```

An API is a boundary through which another program calls the model.
A pipeline is a sequence of fitted transformations followed by a predictor.
An artifact is the saved model file.
An image packages code, packages and the artifact. A container is a running image.

## Input contract

All ten named fields are required. Each value must be a finite JSON number.
Reject strings, booleans, null, missing fields, unknown fields and arrays.
Numeric integers are accepted. Filing age must be a whole number: 23 and 23.0 are valid.
The JSON key order can change. The DataFrame column order below cannot change.

| Order | Field | Meaning and units | Allowed range |
|---:|---|---|---|
| 1 | momentum_21 | Adjusted-price return over 21 sessions, decimal | > -1 |
| 2 | momentum_63 | Adjusted-price return over 63 sessions, decimal | > -1 |
| 3 | momentum_252_21 | Price at t-21 / price at t-252 - 1 | > -1 |
| 4 | volatility_63 | Sample std of 63 daily returns times sqrt(252), decimal | >= 0 |
| 5 | drawdown_252 | Price / maximum price over 252 sessions - 1 | > -1 and <= 0 |
| 6 | net_margin | Percentile rank of net margin | > 0 and <= 1 |
| 7 | operating_roa | Percentile rank of operating income/assets | > 0 and <= 1 |
| 8 | debt_to_assets | Percentile rank of debt/assets | > 0 and <= 1 |
| 9 | cash_to_assets | Percentile rank of cash/assets | > 0 and <= 1 |
| 10 | filing_age_days | Calendar days since report availability | >= 0; whole number |

Returns use decimal units: 0.05 means 5%. Positive adjusted prices imply returns greater than -1.
Volatility cannot be negative. Price divided by its rolling maximum cannot exceed 1.
Percentile ranks lie in (0, 1]. HW4 ranks within date and sector.
It uses the whole eligible date when the sector is unknown or has fewer than five eligible stocks.
Never send raw financial ratios under these rank field names.

Filing age starts at the report's available date, normally filing date plus one calendar day.
HW4's 550-day staleness rule uses fiscal date and blanks financial ratios; it is not an age cap.
No arbitrary upper cap is imposed on returns or volatility. The fitted clipper handles extreme finite values.
This service deliberately requires complete observations. The saved model retains its original imputer,
but the API does not accept null or non-finite features. Do not replace missing inputs with zero.
Input checks cannot prove that the caller used the correct sector, date or price history.

## Output contract

POST /predict returns HTTP 200 with exactly prediction and model_version.
The prediction is finite. It is the adjusted-price return from next-session entry to the exit
90 calendar days later (or the next market session when needed), not a daily return or a trade order.

Reference response:
```json
{"prediction": 0.033696732346221454, "model_version": "hw4-ridge-alpha100-2016-04-01"}
```

HTTP 422 means the input is invalid. HTTP 500 means an unexpected inference error occurred.
The 500 response contains only {"detail": "Prediction unavailable"}.
Detailed exceptions stay in server logs. Non-finite input values are omitted from error responses
so the validation error itself remains valid JSON.

## Setup

Use Python 3.12. Open PowerShell in this submission folder.

1. Create a separate environment: `py -3.12 -m venv .venv`.
2. Install packages: `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
3. Run checks: `.venv\Scripts\python.exe validate.py`.
4. Start the API: `.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000`.
5. Open http://127.0.0.1:8000/docs.

If the py launcher is absent, use the repository's existing Python 3.12 executable to create the environment:
`../../../.venv/Scripts/python.exe -m venv .venv`.
Activation is optional when commands use the environment's Python explicitly.
The supplied Makefile also supports test, lint, format, build and run after environment activation.
On Windows without make, use the equivalent Python and Docker commands in this README.

requirements.txt pins the tested environment, including transitive packages.
The six model-library versions match HW4. Startup refuses a different model-library version.
The current Starlette TestClient emits an httpx deprecation warning. Tests still pass.
Keep the course-required httpx interface for this submission.

## Saved model and provenance

artifacts/model.joblib includes the fitted clipper, median imputer, missing indicators, scaler and ridge model.
model_components.py contains the custom QuantileClipper class required during loading.
The service resolves paths from its own file location. It loads no database and does not import the HW4 project.
The artifact is small enough to include in Git (2,309 bytes at export).
The recorded SHA-256 detects accidental file changes; it is not a signature. Load only the trusted bundled file.

The HW4 runner saved forecasts but did not retain the fitted pipeline.
export_model.py reproduced the April 1, 2016 fit with the original HW4 functions:
2,959,875 training rows, alpha 100, and outcomes ending before the fit date.
All 380,101 forecasts from that fit matched the existing HW4 forecast file exactly.
Saving and reloading changed none of those predictions.
The portable clipper preserves the learned state and behavior of the HW4 clipper.

The reference example is symbol A on April 1, 2016.
Its expected prediction was read from HW4 outputs/final_test/forecasts.parquet before the API was built.
Tests use absolute tolerance 1e-10 and relative tolerance 1e-8 for floating-point differences.
This is stronger than comparing the API only with a direct call to the same loaded file.

To regenerate the artifact, use the original HW4 environment from the repository root:
`.venv/Scripts/python.exe Homeworks/HW5/Mauricio_Gutierrez/export_model.py`.
It needs the existing HW4 final_test filtered_features.duckdb, forecasts.parquet and training_splits.csv.
DuckDB 1.5.6 is an offline export dependency in that environment. It is not needed by the service.
Do not rerun training inside a request. Normal use needs only the committed artifact.

## Validation and evidence

validate.py runs pip check, Ruff lint, Ruff format check and pytest. It saves evidence/checks.txt.
Tests cover real-model/API/reference agreement; all required input failures; model failures;
non-finite model output; field order; one load per process; no fit during requests; and path independence.
Named required fields enforce width: missing or extra fields fail before inference.
Arrays are rejected because this API uses a named object, not a feature-list interface.

The performance test warms up five requests and measures 50 successful sequential requests with perf_counter.
It writes median and p95 in evidence/latency.json. This is an in-process TestClient measurement.
It does not measure a network round trip or concurrent load. There is no arbitrary 500 ms assertion.
Hardware details are in evidence/hardware.json.

Run `.venv/Scripts/python.exe verify_http.py --local` for a real local server on port 18765.
It saves valid and invalid HTTP exchanges in evidence/localhost_http.txt and stops its server.
That transcript is not container evidence.

## Container and restart check

Tested on November 15, 2026 with Docker Desktop (engine 29.9.3, WSL 2 backend, linux/x86_64).
Start Docker first. Use these commands from this folder:

```text
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

Send a request from another terminal:
```text
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@example_request.json"
```

An invalid request (negative volatility) is saved in invalid_request.json; send it the same way to see HTTP 422.

Two scripts repeat the whole check and save the evidence:

- `bash curl_transcript.sh > evidence/container_curl.txt 2>&1` (Git Bash, macOS or Linux) uses
  Docker and curl on port 8000: valid request, invalid request, `docker stop`, `docker start`
  of the same container, then a valid request again. It stops on unexpected command failures,
  checks HTTP status and the saved reference prediction, and verifies the image ID after restart.
  It removes only its own container. It uses the local Python environment for JSON checks;
  set `PYTHON` to a Python executable if that environment is elsewhere.
- `.venv/Scripts/python.exe verify_container.py` builds the image, runs valid/invalid real HTTP requests
  on port 18766, stops and starts the same container, checks its image ID, repeats the requests, and
  removes only its test container. It writes evidence/container.txt and evidence/container_status.json.

Both scripts passed. Observed result before and after the restart:
`HTTP/1.1 200 OK` with `{"prediction":0.033696732346221454,"model_version":"hw4-ridge-alpha100-2016-04-01"}`;
the invalid request returned `HTTP/1.1 422 Unprocessable Entity`.
The model is copied into the image, so every container started from it has the model without any volume or download.

The Dockerfile uses python:3.12-slim and /app. It includes the fitted artifact and custom module.
It runs as an unprivileged user. The process binds port 8000 inside the container.
The host mapping exposes it only on 127.0.0.1. Local environments, databases, secrets and report files
are excluded from the build context. The Python 3.12 base tag can move; record the built image ID.

## Before a real deployment

Add authentication, TLS at the gateway, request limits, timeouts, monitoring and a rollback plan.
Track errors, median/p95 latency, memory and changing input distributions.
Test Linux and concurrent requests under realistic load. Pin an image digest for a release.
Keep model provenance and artifact access controlled. Add automated checks on the target platform.
The old HW4 fit is a course artifact. Passing software tests does not establish investment quality.

## Submission

Docker evidence is complete. Submit only this homework folder in a new HW5 pull request.
Assign kevinramlal and raghuramkowdeed. Do not include the HW4 branch's unrelated changes.
Do not commit .venv, caches, databases, credentials or temporary files.
Optional GitHub Actions CI was not added; it is not required to pass.

## Sources and assistance

- Homeworks/HW5/hw_5.md and its template: assignment structure, required tests and tooling.
- Lectures/Lecture 9/demo/app.py: strict schema, lifespan load and logged inference failure patterns.
- Lectures/Lecture 9/demo/tests/: fixtures, direct-result checks and rejection before inference.
- Lectures/Lecture 9/demo/Dockerfile and Makefile: container and command patterns.
- HW4/Mauricio_Gutierrez/feature_engineering.py: actual pipeline and custom clipper.
  The clipper originally adapts Lectures/modeling/course_data.py.
- Claude Code built the image, ran the container checks, wrote curl_transcript.sh and updated the documents.
- Codex implemented, tested and documented this work with AI assistance.
  The README explains the implementation and validation choices.
