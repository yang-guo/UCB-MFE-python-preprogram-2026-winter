# HW5 — Serving the HW4 stock-ranking model (Oscar Sallusti)

A FastAPI service that loads my fitted HW4 model once at start-up and answers
`POST /predict`. It is packaged as a Docker image.

| File | Purpose |
|---|---|
| `app.py` | Request/response schemas, validation, HTTP error handling |
| `model_service.py` | Loads the artifact, rebuilds the training input, predicts |
| `artifacts/model.joblib` | Fitted HW4 pipeline (5.6 kB) with its version and feature contract |
| `artifacts/model_metadata.json` | Provenance: fit date, training rows, library versions, SHA-256 |
| `export_model.py` | How the artifact and the reference prediction were produced from HW4 |
| `example_request.json` | A real HW4 feature row (AAPL, signal date 2026-09-23) |
| `tests/` | 30 tests; `reference_prediction.json` is the HW4 score for the example |
| `measure_latency.py` | Latency over real HTTP against a running server or container |
| `evidence/` | Terminal transcripts of the runs described below |

## Which model is served

**`hw4-ridge-fit-2026-04-01`**: the last annual refit of the HW4 strategy model
(`Homeworks/HW4/Oscar_Sallusti`, locked `config.py`: `MODEL = "ridge"`).

- A scikit-learn `Pipeline`: `SimpleImputer(fill 0.5, add_indicator=True)` on the
  26 numeric features, `OneHotEncoder` on sector, then `Ridge(alpha=10)`. The
  imputer and encoder are the learned preprocessing and are inside the artifact.
- Fitted on 1,418,748 rows (signal dates from 2010 whose 90-day outcome was
  known before 2026-04-01). `export_model.py` asserts this count equals the one
  HW4 recorded in `results/refits.csv`.
- HW4 did not save fitted models, so `export_model.py` re-runs the final refit by
  calling HW4's own `build_panel` and `fit`. Ridge has a closed-form solution, so
  the refit is deterministic.
- The artifact contains only scikit-learn objects: no custom transformer
  modules are needed to load it.

## API contract

### `POST /predict`

One stock on one signal date. The body is a JSON object with exactly these 27
fields, in the training column order. No other fields are accepted.

**Units.** Each numeric feature is the stock's **percentile rank among that
date's eligible stocks**, exactly as HW4 `rank_features` computes it
(`rank(pct=True)`): the lowest of *n* stocks gets 1/*n* and the highest gets 1.
The allowed range is therefore **0 < x ≤ 1**. The service does not rank raw
values itself, because a rank needs the whole cross-section.

| # | Field | Rank of | Null allowed |
|---|---|---|---|
| 1 | `mom_1m` | 21-session return | yes |
| 2 | `mom_3m` | 63-session return | yes |
| 3 | `mom_6m_1m` | return from 126 to 21 sessions ago | yes |
| 4 | `mom_12_1` | return from 252 to 21 sessions ago | yes |
| 5 | `vol_63` | std. dev. of daily returns, 63 sessions | **no** |
| 6 | `vol_252` | std. dev. of daily returns, 252 sessions | yes |
| 7 | `max_ret_21` | largest daily return in 21 sessions | **no** |
| 8 | `dist_high_252` | price / 252-session high − 1 | yes |
| 9 | `beta_252` | 252-session beta to the S&P 500 TR | yes |
| 10 | `idio_vol_252` | 252-session idiosyncratic volatility | yes |
| 11 | `log_dollar_vol` | log mean dollar volume, 63 sessions | **no** |
| 12 | `log_amihud` | log Amihud illiquidity, 63 sessions | yes |
| 13 | `log_mcap` | log market capitalisation | yes |
| 14 | `earnings_yield` | TTM net income / market cap | yes |
| 15 | `book_to_market` | total equity / market cap | yes |
| 16 | `sales_to_price` | TTM revenue / market cap | yes |
| 17 | `cashflow_to_price` | TTM operating cash flow / market cap | yes |
| 18 | `gross_profit_to_assets` | TTM gross profit / total assets | yes |
| 19 | `roe` | TTM net income / total equity | yes |
| 20 | `asset_growth` | year-on-year total asset growth | yes |
| 21 | `sales_growth` | year-on-year quarterly revenue growth | yes |
| 22 | `earnings_change_to_price` | YoY change in quarterly net income / market cap | yes |
| 23 | `accruals_to_assets` | (TTM net income − TTM operating cash flow) / assets | yes |
| 24 | `debt_to_assets` | total debt / total assets | yes |
| 25 | `share_issuance` | year-on-year growth in shares outstanding | yes |
| 26 | `filing_age_days` | days since the latest report became public | **no** |
| 27 | `sector` | categorical, one of the 12 labels below | **no** |

`sector` is a string, declared separately from the numeric features: `Basic
Materials`, `Communication Services`, `Consumer Cyclical`, `Consumer Defensive`,
`Energy`, `Financial Services`, `Healthcare`, `Industrials`, `Real Estate`,
`Technology`, `Utilities`, `Unknown`.

**Why these constraints.**

- *Range (0, 1].* Anything outside cannot be a percentile rank. It is almost
  certainly a raw value sent by mistake (e.g. `filing_age_days: 45`), and a
  linear model would turn it into a confident, meaningless score.
- *Every field required, null only where training had missing values.* The
  fitted imputer has a missing-value indicator for 22 features, so for those an
  explicit `null` is a state the model has seen. The other four were never
  missing in training (HW4's eligibility screen guarantees `vol_63` and
  `log_dollar_vol`), so a null there is rejected. Requiring every key means a
  caller cannot drop a feature by accident. A test checks that the schema's
  nullable set equals the artifact's.
- *Strict types.* `"0.35"` and `true` are rejected, not coerced. JSON integers
  are accepted because the top rank is exactly `1`. `NaN` and `Infinity` are
  rejected.
- *Sector whitelist.* The encoder would silently ignore an unseen label; I
  reject it instead, so a typo such as `technology` is not scored as "no sector".

**Response 200**

```json
{"prediction": 0.5320347392488041, "model_version": "hw4-ridge-fit-2026-04-01"}
```

`prediction` is the predicted percentile rank (0 worst, 1 best) of the stock's
**90-calendar-day forward return** among that date's eligible stocks, measured
from the next session's close: the HW4 target and horizon. It is a linear
model's output, always finite but not clipped to [0, 1]; on the last HW4 signal
date predictions ranged from 0.29 to 0.59. Only the ordering across stocks on
the same date is meaningful.

**Errors**

| Status | When | Body |
|---|---|---|
| 422 | Missing or extra field, wrong type, out of range, null where not allowed, unknown sector, non-finite number, wrong shape (e.g. a list), malformed JSON | `{"detail": [{"type", "loc", "msg"}, ...]}` |
| 500 | Unexpected failure during inference, or a non-finite model output | `{"detail": "Prediction unavailable"}` plus an `X-Request-ID` header |

The 500 body never contains the exception. The traceback is written to the
server log with the same request id. Validation errors do not echo the rejected
input: FastAPI's default handler does, and a `NaN` in the body then makes the
422 response itself fail with a 500 (I reproduced this before replacing it).

### `GET /health`

`{"status": "ready", "model_version": "hw4-ridge-fit-2026-04-01"}`

## Setup and run

Python 3.12. From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # docs at /docs
```

`requirements.txt` pins the model stack to my HW4 training environment (the
course `uv.lock`): scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3,
joblib 1.5.3, on Python 3.12.9. The same file is installed in the image.

**Artifact loading.** `artifacts/model.joblib` is committed. The service loads it
once at start-up through the FastAPI lifespan, from a path relative to the code.
It never refits and never touches the HW4 database. It is a pickle, so only load
the copy from this repository; its SHA-256 is in `model_metadata.json`.

To rebuild the artifact (not needed to run the service), with the HW4 submission
and course database in place, from the repository root:

```sh
.venv/bin/python Homeworks/HW5/Oscar_Sallusti/export_model.py
```

## Tests, lint and latency

```sh
make test      # python -m pytest -q
make lint      # ruff check . && ruff format --check .
```

Output is in `evidence/tests_and_lint.txt`: **30 passed**, ruff clean.

1. **Valid prediction.** Uses the real artifact. The API result equals a direct
   call to the loaded pipeline, and equals the reference in
   `tests/reference_prediction.json`. That reference is the score HW4's own
   `model.predict()` gave for the AAPL row on 2026-09-23, recorded by
   `export_model.py` before any API code was involved. Tolerance is 1e-9
   absolute: predictions are ranks on a 0–1 scale, so this is far below any
   difference that matters while allowing for floating-point differences
   between macOS and the Linux image. Further tests cover nulls, a rank of
   exactly 1, and that the schema matches the fitted artifact.
2. **Input validation.** 22 invalid requests, each returning 422. A spy on the
   pipeline's `predict` confirms the model was never called.
3. **Inference failure.** `predict` is replaced by one that raises; a valid
   request returns 500 with no exception text in the body. A second test makes
   the model return `NaN` and also gets 500.
4. **Performance.** See below.

**Latency** (Apple M2, 8 GB RAM, macOS 14.6, Python 3.12.9, single worker, 200
sequential requests after 10 warm-up requests, timed with `time.perf_counter`):

| Measurement | Median | p95 |
|---|---|---|
| In-process `TestClient` (the test; no network) | 2.0 ms | 2.6 ms |
| Real localhost HTTP, uvicorn on the host (`measure_latency.py`) | 1.8 ms | 2.0 ms |
| Real localhost HTTP, Docker container | _pending, see below_ | |

The test asserts p95 < 250 ms. That is a regression guard rather than a target:
about 100× the measured value, loose enough for a slow shared CI runner, tight
enough to catch the model being reloaded on every request.

## Container

```sh
make build     # docker build -t mfe-hw5:local .
make run       # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The image is `python:3.12-slim` with `requirements.txt`, `app.py`,
`model_service.py` and `artifacts/`. It runs as a non-root user and listens on
port 8000. `.dockerignore` keeps the virtual environment, databases, `.env`
files, tests and the export script out of the build context. If host port 8000
is taken, map another one, e.g. `-p 127.0.0.1:8080:8000`.

Then, from this folder:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

### Evidence

> **Not done yet: container evidence.** Docker was not installed on the machine
> where this was written, so the image has not been built or run. The transcripts
> for the valid request, the rejected request and the request after a restart
> still have to be recorded in `evidence/container.txt`.

`evidence/local_uvicorn.txt` shows the same three requests against uvicorn
running directly on the host.

## Before a real deployment

- **Training/serving skew is the main risk.** The service expects features
  already ranked across the day's eligible universe. A caller that ranks over a
  different universe, or sends raw values that happen to fall in (0, 1], gets a
  plausible but wrong score. A real system would compute the ranks in one shared
  library used by both training and serving, or accept the whole cross-section
  and rank it inside the service.
- **The model is stale by design and weak.** HW4 refits once a year and this fit
  is from April 2026. Its test-period rank IC was about 0.03 and the strategy
  did not beat the S&P 500. It should not drive real trades.
- **No authentication, rate limiting or TLS.** The container must sit behind a
  gateway that provides them; it binds to localhost here.
- **One stock per request.** Ranking a 500-stock universe takes 500 calls. A
  batch endpoint would be the first change.
- **Artifact trust.** The model is a pickle. Deployment should verify the
  SHA-256 in `model_metadata.json` before loading and pull the artifact from a
  versioned store rather than the image.
- **Monitoring.** Track latency, error rate, the share of null inputs, and the
  distribution of inputs and predictions, so drift is noticed.
- **Reproducible image.** Pin the base image by digest and the transitive
  dependencies with a lock file; scan the image.
- **Rollout.** Keep the previous model version deployable, and run more than
  one worker behind a readiness check on `/health`.

## Code reuse and AI assistance

- **Adapted from the Lesson 9 demo** (`Lectures/Lecture 9/demo`): the
  try/except around inference with a request id and a generic 500 body, the
  `/health` endpoint, the bundle-with-version artifact layout, and the
  Dockerfile's layer order. The template supplied the lifespan skeleton, the
  test fixtures, the Makefile and the ruff settings.
- **AI assistance:** this submission was drafted with Claude Code (Anthropic),
  which wrote the code, tests and this README. The optional CI extension was not
  done.
