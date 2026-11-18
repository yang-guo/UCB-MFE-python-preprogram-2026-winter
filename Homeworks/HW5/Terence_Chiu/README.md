# HW5 – Terence Chiu: serving the HW4 model

A FastAPI service, in a Docker container, that serves the fitted stock-return pipeline from my HW4 submission (PR #237).

## What is served

- Model: scikit-learn `Ridge` (alpha = 1000) inside the course preprocessing pipeline (`QuantileClipper` feature clipping, median imputation with missing indicators, standard scaling, one-hot sector). Trained on HW4 development rows (2010–2015) to predict the 90-calendar-day return; training targets clipped to their 1st/99th percentiles.
- Artifact: `artifacts/model.joblib` (5.5 KB), the fitted HW4 pipeline saved with `joblib.dump` from the HW4 notebook. It is loaded once at startup; the service never retrains or queries the HW4 database.
- Model version reported by the API: `hw4-ridge-alpha1000`.
- `course_data.py` contains only the `QuantileClipper` class, copied unchanged from `Lectures/modeling/course_data.py`. The pickled pipeline refers to `course_data.QuantileClipper`, so the module keeps that name; the course file's data-loading code (and its `duckdb` import) is not needed for serving.

## Input/output contract

`POST /predict` takes one stock on one signal date as a JSON object with exactly these 13 named fields, in training order. Returns and ratios are fractions (0.05 = 5%). Every field is required, numbers must be finite, numeric strings and booleans are rejected (strict mode), and extra fields are rejected.

| Field | Meaning | Allowed range | Why |
|---|---|---|---|
| `momentum_21` | 21-session return | > −1 | A price can't fall below zero |
| `momentum_63` | 63-session return | > −1 | Same |
| `momentum_252_21` | Return from 252 to 21 sessions ago | > −1 | Same |
| `volatility_63` | Annualized volatility of 63 daily returns | ≥ 0 | A standard deviation |
| `drawdown_252` | Price vs. its 252-session high, minus 1 | (−1, 0] | Price can't exceed its own high |
| `net_margin` | Net income / revenue | finite | Can be negative |
| `operating_roa` | Operating income / total assets | finite | Can be negative |
| `debt_to_assets` | Total debt / total assets | ≥ 0 | Both are nonnegative |
| `cash_to_assets` | Cash and equivalents / total assets | ≥ 0 | Both are nonnegative |
| `fcf_to_assets` | Free cash flow / total assets | finite | Can be negative |
| `accruals_to_assets` | (Net income − operating cash flow) / total assets | finite | Can be negative |
| `filing_age_days` | Days since the financial report was filed | ≥ 0 | A filing can't be in the future |
| `sector` | Company sector | one of the 11 training sectors | Unseen sectors would only get an all-zero encoding |

The 11 sectors: Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities.

Responses:

- 200: `{"prediction": <finite float>, "model_version": "hw4-ridge-alpha1000"}`, the predicted 90-day return as a fraction, with an `X-Request-ID` header.
- 422: invalid input (missing, extra, wrong type, non-finite, out of range, unknown sector, or not a JSON object). The body lists each failing field and reason, without echoing the rejected value.
- 500: an unexpected inference failure. The body is only `{"detail": "Prediction unavailable"}`; the full error is logged on the server under the same `X-Request-ID`.

Design decision: the HW4 pipeline can impute missing financial ratios, but the API requires all 13 values, so `null` is rejected. On 2016-03-31, 123 of 1,993 eligible stocks had at least one missing feature; serving them would need an explicit "missing allowed" contract.

`example_request.json` is a valid input: BOOT's HW4 features on 2016-03-31, the first HW4 signal date.

## Setup

Python 3.12, from this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The model-library versions in `requirements.txt` (numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.16.3, joblib 1.5.3) match the HW4 training environment exactly; loading a scikit-learn artifact across versions is unsupported.

## Commands to reproduce

With the environment active:

```sh
make test      # 17 tests
make lint      # ruff check + ruff format --check
make build     # docker build -t mfe-hw5:local .
make run       # serves on http://127.0.0.1:8000 (docs at /docs)
```

Local server without Docker: `python -m uvicorn app:app --host 127.0.0.1 --port 8000`.

Request to the running container:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

## Tests

`tests/test_app.py`, run in-process with FastAPI's `TestClient` against the real saved model:

- Valid prediction: 200, exact response schema, model version, request ID; the API equals a direct `pipeline.predict` call on the same row, and equals the reference prediction recorded from the HW4 code.
- Invalid input (10 cases): missing field, extra field, numeric string, boolean, null, unknown sector, and four out-of-range values each return 422, and the model's `predict` is never called.
- Non-finite values (NaN, Infinity, −Infinity) return 422 without calling the model.
- A bare list instead of named fields (wrong feature count) returns 422.
- Inference failure: `predict` is replaced with one that raises; valid input returns 500 with no internal details in the body.
- Latency: 5 warm-up requests, then 50 timed with `time.perf_counter`.

Reference result: `tests/reference_example.json` holds BOOT's features on 2016-03-31 and the saved HW4 pipeline's score for that row (0.0800623524957497), recorded with the HW4 data code (`course_data.prepare_panel` on `equities_top2000.duckdb`) before the model was wrapped in the API. BOOT was the top-scored stock that day, the first pick of the HW4 strategy. `scripts/make_reference.py` regenerates it (course environment and HW4 database required).

Results:

```text
$ make test
17 passed
$ make lint
All checks passed!
4 files already formatted
```

Latency: median 1.51 ms, p95 2.06 ms over 50 requests. This is an in-process `TestClient` measurement (no network), on an Apple M4 Pro MacBook Pro, macOS, Python 3.12.13. The test asserts p95 < 500 ms, a deliberately loose local budget that catches regressions such as reloading the model per request; real HTTP to the container adds network and serialization overhead.

The tests found a real bug: NaN input returned 500 instead of 422, because FastAPI's default 422 response echoes the rejected value and JSON cannot encode NaN. A custom 422 handler now reports the field and reason without the value.

## Container evidence

- Image: `python:3.12-slim`, built with `make build` (`mfe-hw5:local`).
- Valid request to the running container: `HTTP/1.1 200 OK`, `x-request-id: 4eda6f7a589d`, body `{"prediction":0.0800623524957497,"model_version":"hw4-ridge-alpha1000"}`, the same value as the HW4 reference.
- Full transcript in `container_evidence.txt`: the valid request (200), a rejected invalid request (`example_invalid_request.json`, 422), and the valid request again after stopping and restarting the same image (200).

## Before a real deployment

- Training/serving skew: the caller must compute the 13 features exactly as the HW4 code did. A real service should compute them itself from a shared feature library and data feed, not trust client-supplied features.
- Missing data: real stocks often lack some financial ratios; decide whether to allow and impute them (as in training) or keep rejecting them, and monitor how often requests fail validation.
- Model staleness: the model was fitted once on 2010–2015 data. It needs a retraining schedule, model versioning, and a registry or storage location instead of a file baked into the image.
- Monitoring: request rates, error rates, latency, and the distribution of inputs and predictions (drift) per model version, using the request ID to tie responses to logs.
- Security and operations: authentication and rate limiting, HTTPS, a health/readiness endpoint, running as a non-root user, and loading pickles only from trusted, checksummed artifacts.
- Scale: multiple uvicorn workers or replicas behind a load balancer; the model is small and stateless, so this is straightforward.

## Files

- `app.py`: contract, model loading, `/predict`, 422 handler
- `course_data.py`: the `QuantileClipper` class the pickled pipeline needs
- `artifacts/model.joblib`: the fitted HW4 pipeline
- `tests/test_app.py`, `tests/reference_example.json`: tests and the HW4 reference
- `scripts/make_reference.py`: regenerates the reference
- `example_request.json`, `example_invalid_request.json`: container requests
- `container_evidence.txt`: container transcript
- `Dockerfile`, `Makefile`, `requirements.txt`, `pyproject.toml`

## Reused code and AI assistance

- The app structure (lifespan loading, request ID, 500 handling) and test structure follow the Lecture 9 demo (`Lectures/Lecture 9/demo/`).
- `QuantileClipper` is copied from `Lectures/modeling/course_data.py`.
- AI assistance (Claude Code):
  - Wrote the `InputData`/`OutputData` contract and the slim `course_data.py`.
  - Built the reference example and `scripts/make_reference.py`.
  - Wrote the test suite and the 422 handler that fixed the NaN bug the tests exposed.
  - Fixed bugs in my `/predict` code and Dockerfile.
  - Created the example request files and drafted this README.
- My own work: model loading and `/predict`, the first Dockerfile, the environment setup, building and running the container, and collecting the container evidence.
