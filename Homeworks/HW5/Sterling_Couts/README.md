# HW5 — Sterling Couts

This service exposes the fitted stock-return pipeline used in HW4. It loads the
saved imputer, missing indicators, scaler and Ridge model once per process and
returns a prediction through `POST /predict`. It does not train, query a database,
construct features, select stocks or place trades.

Model version: **hw4-fixed-ridge-2012-84e526d28792-v1**. The unchanged artifact
comes from HW4 commit `9d94d05`, candidate `F00_fixed`, fitted as of December 31,
2012 with Ridge alpha 1000. Its SHA256 and exact model-library versions are in
[model_metadata.json](artifacts/model_metadata.json).

## Input and output contract

Send a JSON object with exactly these eight required keys. JSON object order does
not matter: `serving.feature_frame` explicitly restores the training column order.

Every numeric input is a **cross-sectional percentile rank in (0, 1]**, not a raw
return, volatility, accounting ratio or percentage from 0 to 100. Values such as
1 and 1.0 are numeric and valid; strings such as "0.5" and booleans are rejected.
Zero, values above one, infinity and NaN are rejected. The two financial fields
allow explicit `null`; their keys are still required. Other fields cannot be null.

| Order | Feature | Underlying HW4 quantity before ranking | Null allowed |
|---|---|---|---|
| 1 | `rank_momentum_21` | Adjusted close today / adjusted close 21 sessions ago − 1 | No |
| 2 | `rank_momentum_63` | Adjusted close today / adjusted close 63 sessions ago − 1 | No |
| 3 | `rank_momentum_252_21` | Adjusted close 21 sessions ago / adjusted close 252 sessions ago − 1 | No |
| 4 | `rank_volatility_63` | Sample standard deviation of 63 daily simple returns × √252 | No |
| 5 | `rank_drawdown_252` | Adjusted close / trailing 252-session maximum − 1 | No |
| 6 | `rank_momentum_126_21` | Adjusted close 21 sessions ago / adjusted close 126 sessions ago − 1 | No |
| 7 | `rank_operating_roa` | Available quarterly operating income / assets | Yes |
| 8 | `rank_fcf_to_assets` | Available quarterly free cash flow / assets | Yes |

Ranks use `rank(pct=True)` across HW4 eligible, risk-screened members on the same
signal date. Ties receive their average rank; missing observations are excluded
from each feature's denominator. This calculation belongs upstream. Ranking an
individual API request, or a user's arbitrary batch, would change feature meaning.

HW4 requires dated membership, causal market histories and financial availability
at filing date plus one calendar day. Invalid filing dates are excluded, and
financial observations older than 550 days are missing. The API cannot verify
these provenance rules from eight numbers; callers must supply HW4-compatible
features. It does preserve the learned preprocessing: financial nulls become
`np.nan`, and the saved training medians and missing indicators handle them.

A successful response has exactly:

```json
{
  "prediction": 0.03153401217040266,
  "model_version": "hw4-fixed-ridge-2012-84e526d28792-v1"
}
```

This is the result for [example_request.json](example_request.json), the saved
AAT observation dated December 31, 2012. The forecast is a **simple return
fraction**: about 3.1534%, from the next market session's close to the first
session on or after entry plus 90 calendar days. It is not annualized, a
probability or a trading recommendation. Training responses were winsorized
using training-only bounds; serving does not clip predictions or deduct costs.

| Status | Meaning |
|---|---|
| 200 | One finite numeric prediction and a model version |
| 422 | Missing/extra field, wrong width or body shape, invalid type/range, or non-finite input; no inference |
| 500 | Unexpected inference failure or invalid model output; generic error, detailed server log |

The service implements a JSON 500 body `{"detail": "Prediction unavailable"}`.
Successful predictions and inference errors have an `X-Request-ID` header.
Validation errors omit the invalid input value so non-finite values cannot break
error serialization. Startup errors fail the process before it serves requests.

## Setup and local execution

Use a **separate Python 3.12 environment**. The model-library versions in
`requirements.txt` match the actual HW4 environment, not the starter defaults.
Do not replace them with newer sklearn or numerical-library versions.

From this folder in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the input schema and interactive requests.
`GET /health` reports readiness and model version. The artifact path is resolved
relative to the service module, so it does not depend on the caller's directory.
Stop the server with Ctrl+C.

The artifact is small enough to include in Git. `hw4/preprocessing.py` supplies
the original import path for `TargetClippedRegressor`. Its fitted Ridge
estimator is already serialized. No training data or HW4 database is needed.
Only load a joblib artifact from a trusted source; a matching checksum alone
does not establish trust.

## Tests and timing evidence

```powershell
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
& .\.venv\Scripts\python.exe -m pytest -q
```

The Makefile retains `test`, `lint`, `format`, `build` and `run`; activate the
environment first if using `make`. The commands above work without Make.

The positive tests use the real pipeline and all 209 input–prediction pairs
recorded by HW4's `prepare_fixed_anchor.py` before this API existed. The original
CSV is copied unchanged. Tests compare API and direct predictions, and the saved
HW4 results provide an independent check. Absolute tolerance `1e-12` accommodates
floating-point differences; it is much smaller than any economically relevant
return change.

Validation tests check that rejected inputs do not reach the model. Failure
tests replace `predict` with a function that raises and verify an HTTP 500
response. Other tests cover non-finite model output, loading once per lifespan,
and requests leaving the fitted pipeline unchanged.

The performance test performs 5 warmups followed by 100 sequential successful
requests using `time.perf_counter`. It records the median, p95 (linear percentile),
raw observations and hardware in `evidence/latency_testclient.json`. This is
in-process TestClient timing, not a network measurement. A separate real
localhost/container measurement is recorded by the request demonstration.
There is no universal latency assertion: these measurements establish this
machine's observed behavior, not a production service-level objective.

See [VALIDATION.md](VALIDATION.md) for requirement coverage and observed results.

## Build, request and restart the container

Docker Desktop must be running in Linux-container mode. From this folder:

```powershell
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The server listens on port 8000 inside the container. The host binding limits
access to this computer. If port 8000 is busy, use `127.0.0.1:8875:8000`
and request host port 8875. The image runs as an unprivileged user.

In a second terminal in the homework folder:

```powershell
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@example_request.json"
& .\.venv\Scripts\python.exe -m scripts.request_demo --url http://127.0.0.1:8000 --label manual-container
```

On macOS/Linux use `curl` and `.venv/bin/python`. The Python demo records the
valid 200 response, a deliberately incomplete request returning 422, and real
HTTP latency. Stop the container and run the same image again; repeat the request.

The automated alternative builds, sends both requests, stops and starts the
same image, checks its ID and internal model hash, and removes only its own
temporary container:

```powershell
& .\.venv\Scripts\python.exe -m scripts.verify_container
```

It uses port 8875 and container name `mfe-hw5-sterling-check`, refusing to replace
an existing container with that name. Its actual transcript and verification
record go in `evidence/`. The Docker build context allows only the application,
requirements and model inputs; it excludes databases, credentials, local
environments, tests and evidence.

## Before a real deployment

This is a local coursework service. A deployed service would need authenticated
access, TLS, input-size/rate limits, measured concurrency and timeouts, dependency
and image vulnerability review, and a documented release/rollback process.
Version the complete feature contract and model artifact together. Pin and retain
an approved image digest; the `python:3.12-slim` tag can change over time.

Monitor request rate, 422/500 rates, tail latency, CPU/RAM, feature missingness and
rank distributions. Check live predictions against trusted reference examples
after releases. A health response proves readiness, not economic forecast
quality. Evaluate forecast quality separately once the 90-day outcomes mature.
Retain the prior image for rollback and avoid logging confidential request bodies.

One process is sufficient here. Extra workers hold extra model copies; measure
load before introducing worker pools, queues or microservices. The service makes
no trade or persistent state change, so repeating a prediction does not duplicate
an order.

## Sources, implementation and validation

- [HW5 assignment and starter](../hw_5.md): milestones, endpoint, tests, Makefile
  and Dockerfile structure.
- [Lesson 9 demo](../../../Lectures/Lecture%209/demo/): lifespan loading,
  TestClient fixtures, error handling and request IDs.
- Sterling's HW4 commit `9d94d05`: unchanged fitted artifact, reference CSV,
  custom estimator and feature definitions.
- [FastAPI lifespan testing](https://fastapi.tiangolo.com/advanced/testing-events/),
  [Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/),
  and [sklearn persistence](https://scikit-learn.org/stable/model_persistence.html).

This project adapts the HW5 starter and Lecture 9 examples to serve my fitted
HW4 model through FastAPI. The implementation preserves the model's
preprocessing, feature definitions, and prediction horizon, with separate
components for input validation and inference.

Verification includes 126 passing tests, agreement with 209 saved HW4 reference
predictions, error-handling checks, and successful container requests before
and after restart. The README explains the design decisions and provides
commands to reproduce these checks.

I reviewed the implementation and verification results against the assignment
requirements.
