# HW5: Serving my HW4 stock-return model (Ananya Das)

This folder turns the model I fitted in HW4 into a small web service. A FastAPI endpoint,
`POST /predict`, scores one stock with my **saved HW4 pipeline**, which is tested, packaged
with Docker, and demonstrated with real requests to the running container.

> **What is served:** model version **`hw4-ridge-final-refit-2026-01-02`**. This is the last
> yearly refit from my HW4 final test, a Ridge regression (alpha = 1e7) with a constant 0.5
> imputer. It was saved from my HW4 code, not retrained or re-implemented. The service loads
> it once at startup and never fits anything.
>
> **Status (checked 2026-11-17):** 46 of 46 tests pass, Ruff lint and format checks pass, and
> the Docker image builds. The container returns the recorded HW4 prediction (identical before
> and after a restart) and rejects invalid input with HTTP 422.

## Contents

1. [Architecture and files](#1-architecture-and-files)
2. [The model being served](#2-the-model-being-served)
3. [Input and output contract](#3-input-and-output-contract)
4. [API reference](#4-api-reference)
5. [Local setup and running](#5-local-setup-and-running)
6. [Re-creating the model artifact](#6-re-creating-the-model-artifact)
7. [Tests, lint, and latency](#7-tests-lint-and-latency)
8. [Docker](#8-docker)
9. [Evidence from the running container](#9-evidence-from-the-running-container)
10. [Limitations and what a real deployment would need](#10-limitations-and-what-a-real-deployment-would-need)
11. [Citations and AI use](#11-citations-and-ai-use)
12. [Submission checklist](#12-submission-checklist)

## 1. Architecture and files

```
client ──HTTP JSON──▶ app.py (FastAPI)            model.py (inference)          artifacts/
                      • InputData: validates  ──▶ • StockReturnModel.load() ◀── model.joblib
                        all 8 features            • one named DataFrame row      model_metadata.json
                      • 422 handler (no echo)     • pipeline.predict()
                      • 500 on inference error    • finite check
                      • lifespan: load once
```

Request handling (`app.py`) is separate from model logic (`model.py`). The service does not
import `hw4.py`, DuckDB, or matplotlib, and does not need the HW4 database.

| File | Role |
|---|---|
| `app.py` | FastAPI app: request/response schemas, the sanitised 422 handler, `GET /health`, `POST /predict`, and the lifespan function that loads the model once per process |
| `model.py` | Inference layer: loads `artifacts/model.joblib`, builds a one-row DataFrame in the artifact's feature order (`null` becomes NaN), calls `predict`, and rejects non-finite output |
| `export_model.py` | One-off script, run with the HW4 environment: rebuilds the HW4 final-test model with my HW4 functions, saves the artifact, records reference predictions, and verifies them in a fresh process |
| `artifacts/model.joblib` | The fitted HW4 pipeline (1,924 bytes): `{"pipeline", "model_version", "features"}` |
| `artifacts/model_metadata.json` | Model provenance, derived from the fitted model and HW4 settings |
| `artifacts/reference_prediction.json` | Real HW4 test-period inputs with the predictions my HW4 pipeline made for them, recorded before any API code |
| `example_request.json` | A valid request: the complete reference input (stock A, 2026-08-31) |
| `tests/test_app.py` | The test suite (46 tests) |
| `Dockerfile`, `.dockerignore` | Container image definition and build-context exclusions |
| `requirements.txt` | Pinned service, test, and model-library versions |
| `Makefile` | `test`, `lint`, `format`, `build`, `run` |
| `pyproject.toml` | pytest and Ruff settings (unchanged from the template) |

## 2. The model being served

### Where it comes from

My HW4 (PR #296, folder `Homeworks/HW4/Ananya_Das`, commit `92ecd938`) ranks about 1,850
eligible stocks by predicted 90-day return relative to the median stock. In its final test it
refit the model at the first signal date and then every January, from 2016 to 2026. HW4 never
saved a model file, so `export_model.py` rebuilds the **last** refit with HW4's own code:

1. Imports the unchanged `hw4.py` and runs the same model steps as `python hw4.py final`:
   `check_frozen()` → `load(TEST_END)` → `FeatureBuilder.panel(...)` → `refit_dates(...)` →
   `fit(panel, 2026-01-02, alpha)`. HW4's final-test lock (`RUN_FINAL_TEST`) is switched on
   only inside that process; `hw4.py` is not edited.
2. Checks that the refit schedule matches the one recorded in HW4's `final_results.md`
   (2016-03-31, 2017-01-03, …, 2026-01-02), so this is the same model the final test used.
3. Saves the fitted pipeline with joblib, plus metadata and reference predictions.
4. Reloads the files in a fresh Python process, without `hw4.py`, and checks that the saved
   model reproduces the recorded predictions within 1e-12. They match exactly.

### Model details (from `artifacts/model_metadata.json`)

| | |
|---|---|
| Model version | `hw4-ridge-final-refit-2026-01-02` |
| Pipeline | `SimpleImputer(strategy="constant", fill_value=0.5, keep_empty_features=True)` → `Ridge(alpha=1e7)` |
| Fit date | 2026-01-02 |
| Training rows | 362,402 stock-dates on 189 month-ends, 3,661 stocks |
| Training signal dates | 2010-01-29 to 2025-09-30 (the last target ended 2025-12-30). HW4 only trains on 90-day targets that had ended before the fit date, so data loaded through 2026-09-30 does not enter this fit. |
| Target | 90-day return minus that date's median stock, capped at each date's 1st/99th percentiles for training |
| Library versions | Python 3.12.9, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3, threadpoolctl 3.6.0 |

Coefficients (features are ranks, so they can be compared directly) and intercept:

| Feature | Coefficient | | Feature | Coefficient |
|---|---:|---|---|---:|
| `mom_21` | +1.55e-06 | | `operating_roa` | +6.71e-05 |
| `mom_252_21` | +4.33e-05 | | `accruals_to_assets` | −1.62e-06 |
| `vol_63` | −1.94e-05 | | `debt_to_assets` | +1.43e-05 |
| `drawdown_252` | +3.87e-05 | | `log_cap` | +1.35e-05 |
| **Intercept** | **0.0048801** | | | |

The large penalty keeps every prediction close to the intercept: across all valid inputs it
ranges from about 0.00486 to 0.00506. HW4 used the score only to **rank** stocks. In HW4's
out-of-sample test, the strategy built on this model did not beat the S&P 500 (7.0% vs 15.2% a
year; see my HW4 README). This assignment grades the service, not investment performance.

## 3. Input and output contract

### Inputs: eight precomputed percentile ranks

The inputs are **not** raw prices or financial ratios. Each value is the stock's
**cross-sectional percentile rank** among that signal date's eligible stocks, exactly as HW4's
`FeatureBuilder.features_on()` computes it (`pandas` `rank(pct=True)`). A rank needs every
stock's value on that date, so it cannot be computed from one stock alone. The client must
send the ranks that HW4's feature pipeline produced.

All eight keys are required, in this order (the order of the trained pipeline):

| # | Key | Rank of … | Missing in training |
|---:|---|---|---:|
| 1 | `mom_21` | 21-session return | 0.001% |
| 2 | `mom_252_21` | return from 252 to 21 sessions ago | 3.10% |
| 3 | `vol_63` | annualized volatility of 63 daily returns | 0.06% |
| 4 | `drawdown_252` | price / 252-session high − 1 | 2.95% |
| 5 | `operating_roa` | operating income / total assets (latest filing, usable from filing date + 1 day, blank after 550 days) | 0.03% |
| 6 | `accruals_to_assets` | (net income − operating cash flow) / total assets | 0.06% |
| 7 | `debt_to_assets` | total debt / total assets | 0.03% |
| 8 | `log_cap` | log of the quarter's market-cap proxy | 0.00% |

**Rules, and why:**

| Rule | Reason |
|---|---|
| Each value is a finite number in **[0, 1]**, or **`null`** | Percentile ranks lie in (0, 1]; training values ranged from about 0.0005 to 1.0. |
| `null` is allowed for **any** feature | HW4 itself scored stocks with missing features. For example, stocks with less than a year of history have no `mom_252_21`/`drawdown_252`. The pipeline's imputer fills a missing rank with the middle rank, **0.5**. |
| All eight keys must be present | A missing key is an error, not the same as an explicit `null`. |
| No extra keys | Catches typos and wrong feature sets. A 7- or 9-feature request fails as a missing or extra key. |
| At least one value must be a number | An all-null row carries no information and would only return the intercept. |
| No strings, booleans, lists, or objects | `"0.5"` and `true` must not silently become numbers (strict mode). JSON integers `0` and `1` are accepted as 0.0 and 1.0. |
| No `NaN`, `Infinity`, `-Infinity`, or overflowing numbers such as `1e400` | Missing values must be sent as `null`, not as non-finite numbers. |

### Output

| Field | Type | Meaning |
|---|---|---|
| `prediction` | finite float | Predicted **90-day return minus the median eligible stock's return**, as a fraction (0.01 = 1 percentage point). It is not rounded. |
| `model_version` | string | The version stored in the loaded artifact: `hw4-ridge-final-refit-2026-01-02` |

## 4. API reference

Interactive documentation: `http://127.0.0.1:8000/docs`. The example on that page is the
contents of `example_request.json`.

### `GET /health`

```json
{"status": "ok", "model_version": "hw4-ridge-final-refit-2026-01-02"}
```

### `POST /predict`

Request (`example_request.json`):

```json
{
  "mom_21": 0.8199174406604747,
  "mom_252_21": 0.4978969505783386,
  "vol_63": 0.25748194014447884,
  "drawdown_252": 0.8934942287513117,
  "operating_roa": 0.7301341589267286,
  "accruals_to_assets": 0.8384932920536635,
  "debt_to_assets": 0.48348813209494323,
  "log_cap": 0.8555211558307534
}
```

Response, HTTP 200. This equals the prediction recorded from HW4 for this input:

```json
{"prediction": 0.004998615115138454, "model_version": "hw4-ridge-final-refit-2026-01-02"}
```

Real examples with missing values (from `artifacts/reference_prediction.json`):

| Example | Null features | Prediction (API = HW4) |
|---|---|---|
| `A`, 2026-08-31 (`example_request.json`) | none | 0.004998615115138454 |
| `ALMR`, 2026-08-31 | `mom_252_21`, `drawdown_252` | 0.00491501385259442 |
| `SPHR`, 2024-10-31 | `operating_roa`, `accruals_to_assets`, `debt_to_assets` | 0.00494366639922651 |

All three are real HW4 final-test observations. HW4's backtest scored the SPHR date with its
2024-01-02 refit, so for SPHR the recorded value is this served pipeline's prediction for
those real inputs. No real observation in the final test had a missing `log_cap`. The JSON
file records that this case was not found, rather than inventing one.

### Errors

| Status | When | Body |
|---|---|---|
| **422** | Any input that breaks the contract above, malformed JSON, or an empty body. **The model is never called.** | Only `type`, `loc`, and `msg` for each error. The submitted values are never echoed. |
| **500** | The model raises, or returns a non-finite value | `{"detail": "Prediction unavailable"}`, plus an `X-Request-ID` header. The exception and traceback go to the server log under the same id. |

Example 422 (a `NaN` value, from the running container):

```json
{"detail":[{"type":"finite_number","loc":["body","mom_21"],"msg":"Input should be a finite number"}]}
```

FastAPI's default 422 handler echoes the submitted input. A `NaN` or `Infinity` input cannot be
written back as JSON, so the default handler turned those validation errors into HTTP 500s.
`app.py` replaces it with a handler that returns only `type`, `loc`, and `msg`, so these cases
return 422.

## 5. Local setup and running

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Run these from this folder
(`Homeworks/HW5/Ananya_Das/`):

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

> On my Mac, `python3.12 -m venv .venv` (the assignment's suggested command) produced a broken
> environment, because `python3.12` is a symlink to a uv-managed interpreter. `uv venv` works.
> On a standard Python install, `python3.12 -m venv .venv` followed by
> `python -m pip install -r requirements.txt` is equivalent.

The model-library pins in `requirements.txt` already match my HW4 training environment exactly,
so no version changes were needed. sklearn does not support loading an artifact across versions.

| Task | Command (environment activated) | Without `make` |
|---|---|---|
| Run the API | — | `python -m uvicorn app:app --host 127.0.0.1 --port 8000` |
| Tests | `make test` | `python -m pytest -q` |
| Latency report | — | `python -m pytest -q -s -k performance` |
| Lint and format check | `make lint` | `python -m ruff check .` and `python -m ruff format --check .` |
| Format | `make format` | `python -m ruff format .` |

With the server running:

```sh
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

## 6. Re-creating the model artifact

The artifact is small (1,924 bytes) and is committed, so the service, tests, and Docker image
need nothing else. Rebuilding it is only needed to audit where it came from, and requires:

- the HW4 code at `../../HW4/Ananya_Das/` (this branch is based on my HW4 branch),
- the repository-root environment (`uv sync --locked` at the repo root; it has DuckDB), and
- `equities_top2000.duckdb` from bCourses in `Lectures/modeling/data/` (not in Git).

```sh
../../../.venv/bin/python export_model.py            # about 25 s and 2.6 GB peak memory on my M2
python export_model.py --verify                       # re-check the saved files only; no database
```

A rerun never overwrites `artifacts/model.joblib`. It rebuilds the model, checks that it is
identical to the saved one, and stops if anything differs. It also refuses to change the
original reference example. `--verify` loads the artifact without `hw4.py` and reproduces all
three reference predictions (absolute difference 0).

Joblib files are pickles. Only load artifacts you created or trust; this one was created by
`export_model.py` from my own HW4 code.

## 7. Tests, lint, and latency

### Results (run on 2026-11-17)

```
$ make test
.venv/bin/python -m pytest -q
..............................................                           [100%]
46 passed, 1 warning in 1.55s

$ make lint
.venv/bin/python -m ruff check .
All checks passed!
.venv/bin/python -m ruff format --check .
5 files already formatted
```

(These ran with `make test PYTHON=.venv/bin/python`, the same as `make test` with the
environment activated. The five formatted files are the four Python files and this README.) The one warning is Starlette's notice that `TestClient` with `httpx`
is deprecated in favour of `httpx2`. It does not affect the results, and `httpx==0.28.1` is
the version pinned by the template.

### What the 46 tests cover

| Group | Tests | Checks |
|---|---:|---|
| Valid predictions | 3 | All three reference examples: 200, exactly `prediction` and `model_version`, version equal to the artifact's, result within **1e-12** of the recorded HW4 value and of a direct `pipeline.predict` call, and the model receives a one-row DataFrame in artifact order with NaN exactly where the input had `null` |
| Reference coverage | 1 | The real examples include missing price and missing financial features |
| Invalid input | 19 | Missing or extra feature, empty object or body, `true`/`false`, numeric and text strings, list, object, below 0, above 1, `NaN`, `Infinity`, `-Infinity`, `1e400`, all null, array instead of object, malformed JSON. Each returns 422, the model is **never called** (spy), errors contain only `type`/`loc`/`msg`, and no submitted value appears in the response. Non-standard numbers are sent as raw request bodies. |
| Null per feature | 8 | `null` for each single feature returns 200, and only that column reaches the model as NaN |
| Required per feature | 8 | Omitting each feature returns a `missing` 422 and never calls the model |
| Inference failure | 1 | `predict` patched to raise an error containing a fake secret and path: 500, generic body, nothing leaked, and the exception logged once under the returned `X-Request-ID` |
| Non-finite output | 3 | The model returning NaN, inf, or −inf gives 500, never 200 |
| Health and lifespan | 2 | `/health` returns the artifact version. The model loads once across many requests, and once more for a new lifespan (a restart). |
| Latency | 1 | See below |

The 1e-12 tolerance is deliberately tight: predictions for different stocks differ by about
1e-4, so a loose tolerance could pass for the wrong input.

To check that the tests detect real defects, I ran the suite against scratch copies of
`app.py`/`model.py`, each with one deliberate bug. Tests failed for each of these: the default
422 handler, non-strict types, no all-null check, NumPy input instead of a DataFrame, reversed
feature order, imputing `null` as 0.5 in the API instead of NaN through the pipeline,
returning 0.0 on failure, leaking the exception text, and reloading the model per request.

### Latency

`test_predict_performance` sends 5 warm-up requests, then times 50 successful requests with
`time.perf_counter`.

| Measurement | Value |
|---|---|
| Method | **In-process FastAPI `TestClient`**: no network, no uvicorn, no Docker. It measures validation, DataFrame construction, and prediction inside one process. |
| Hardware | Apple M2, 8 GB RAM, macOS, Python 3.12.9 |
| Median | **1.21–1.22 ms** (three runs on 2026-11-17) |
| p95 | **1.39–1.50 ms** |
| Max | about 23 ms (one slow request per run) |
| Budget asserted | median < 50 ms, p95 < 200 ms: about 40× and 130× headroom, so slower machines do not fail, while gross regressions such as blocking calls or heavy per-request work still do |

These numbers are **not** network or production latency. Requests through uvicorn, Docker port
mapping, or a real network add serialization and transport time that this test does not
measure.

## 8. Docker

The image uses `python:3.12-slim`, works in `/app`, installs `requirements.txt`, and copies
only `app.py`, `model.py`, `artifacts/model.joblib`, and `artifacts/model_metadata.json`. It
runs as a non-root user, exposes port 8000, has a health check, and starts uvicorn. No HW4 code,
DuckDB, matplotlib, tests, data, or local environments go into the image; `.dockerignore` also
keeps `.venv`, caches, `.env` files, and databases out of the build context. The model is
loaded from the packaged artifact at startup and is never retrained in the container.

```sh
make build        # docker build -t mfe-hw5:local .
make run          # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

In a second terminal, or run the container in the background with `-d`:

```sh
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
docker stop mfe-hw5           # --rm removes the stopped container; the image stays
```

The container listens on port 8000. If another service already uses host port 8000, stop it
first or map a different host port, for example `-p 127.0.0.1:8001:8000`.

**Reproducing the build:** clone the repository and check out this branch, then run
`make build` from this folder. Docker and network access to Docker Hub and PyPI are the only
requirements; the artifact is in Git. My build:

| | |
|---|---|
| Base image | `python:3.12-slim`, digest `sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b` at build time. Pin it with `FROM python:3.12-slim@sha256:…` for a byte-identical base. |
| Image | `mfe-hw5:local`, linux/arm64, 740 MB on disk (161 MB compressed). It runs Python 3.12.16 with the same pinned library versions as the local environment. |
| Build time | about 15 minutes, almost all of it downloading wheels over a slow connection. Code-only changes reuse the cached dependency layer. |

The image includes `pytest`, `ruff`, and `httpx` because the template keeps service and
homework tools in one `requirements.txt` (see section 10).

## 9. Evidence from the running container

Copied from my terminal on 2026-11-17, image `mfe-hw5:local` (`531f8174db29`), container
started with `docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local`.

**Startup log** (`docker logs mfe-hw5`): the model is loaded once.

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     model_loaded version=hw4-ridge-final-refit-2026-01-02
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Valid request:**

```
$ curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
date: Tue, 17 Nov 2026 14:17:23 GMT
server: uvicorn
content-length: 86
content-type: application/json

{"prediction":0.004998615115138454,"model_version":"hw4-ridge-final-refit-2026-01-02"}
```

**Rejected request** (`NaN` for `mom_21`):

```
HTTP/1.1 422 Unprocessable Entity
{"detail":[{"type":"finite_number","loc":["body","mom_21"],"msg":"Input should be a finite number"}]}
```

**Other checks against the container** (all as expected, with no submitted value echoed):

| Request | Status | Result |
|---|---:|---|
| `log_cap: null` | 200 | 0.004993830501918713 |
| ALMR (real price nulls) | 200 | 0.00491501385259442 |
| SPHR (real financial nulls) | 200 | 0.00494366639922651 |
| Missing `vol_63` | 422 | `missing` |
| Extra key `sector` | 422 | `extra_forbidden` |
| All eight null | 422 | `value_error` |
| `drawdown_252: 1.5` | 422 | `less_than_equal` |
| `mom_21: -0.25` | 422 | `greater_than_equal` |
| `NaN`, `Infinity`, `1e400` | 422 | `finite_number` |
| `mom_21: true` | 422 | `float_type` |
| `vol_63: "0.4321"` | 422 | `float_type` |

Inside the container, the service runs as `appuser`. `/app` contains only `app.py`,
`model.py`, `requirements.txt`, and the two artifact files. `duckdb`, `matplotlib`, and `hw4`
are not importable.

**Restart:** `docker stop mfe-hw5` (container `d536adef0405`, removed by `--rm`; `/health`
then got no response), then a fresh container from the same image (`e3b427919525`):

```
INFO:     Started server process [1]
INFO:     model_loaded version=hw4-ridge-final-refit-2026-01-02
INFO:     Application startup complete.

$ curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
{"prediction":0.004998615115138454,"model_version":"hw4-ridge-final-refit-2026-01-02"}
```

(Response headers trimmed in this excerpt.)

| Request | Before restart | After restart | Abs. difference |
|---|---|---|---:|
| `example_request.json` | 0.004998615115138454 | 0.004998615115138454 | 0 |
| `log_cap: null` | 0.004993830501918713 | 0.004993830501918713 | 0 |
| ALMR | 0.00491501385259442 | 0.00491501385259442 | 0 |
| SPHR | 0.00494366639922651 | 0.00494366639922651 | 0 |

After the restart, all the checks above passed again, and Docker's health check reported the
container as `healthy`.

## 10. Limitations and what a real deployment would need

**Limitations of what is served:**

- **A frozen January 2026 model.** It learned from targets that ended by 2025-12-30. HW4
  refit yearly; this service has no refit or model-update process.
- **Ranks in, not raw data.** The client must compute HW4's cross-sectional percentile ranks
  for the whole eligible universe on the signal date. Sending raw ratios or returns gives
  meaningless predictions; only the [0, 1] range is checked. There is no check that a request
  came from a real cross-section or a valid signal date.
- **What the output means.** It is a relative 90-day score for ranking stocks, as a fraction
  over the median stock. It is not an absolute return forecast, a probability, or advice. With
  alpha = 1e7 all predictions lie between about 0.00486 and 0.00506.
- **HW4's data issues carry over.** The audit in my HW4 README found missing split adjustments
  and takeover targets without deal proceeds. HW4 features also use restated fundamentals and
  today's sector labels.
- **The financial-missing reference (SPHR) comes from 2024.** No date scored by this model had
  a missing financial feature. No real example with a missing `log_cap` exists.

**Not implemented here, and needed before real use:**

- **A feature service** that computes the ranks from point-in-time data, with checks that they
  match training (data validation and drift monitoring).
- **Security:** authentication, rate limiting, request-size limits, HTTPS/TLS, and a trusted
  artifact store with checksum or signature checks, since joblib is a pickle.
- **Operations:** metrics and alerting on latency, errors, and prediction distributions;
  structured logs; more than one worker or replica; CPU and memory limits; and a readiness
  check separate from liveness.
- **Model lifecycle:** a scheduled refit, versioned artifact storage, staged rollout and
  rollback, and a record of which version produced each prediction.
- **A leaner, reproducible image:** split runtime and development requirements (`pytest`,
  `ruff`, and `httpx` are not needed in production), pin the base image digest, and use a
  hash-locked dependency file.
- **CI:** the optional GitHub Actions workflow was not set up for this submission.
- **Real latency:** load-test through the network and container. The latency in section 7 is
  in-process only.

## 11. Citations and AI use

- **Lesson 9 demo** (`Lectures/Lecture 9/demo/`):
  - The service follows its structure: loading a joblib bundle with `pipeline`,
    `model_version`, and `features` in the lifespan function, the `uvicorn.error` logger, a
    request id with a generic 500, and the Dockerfile layout. I adapted these to my model.
  - The ranked-feature contract, null policy, sanitised 422 handler, inference layer, export
    script, and tests are my own.
- **HW4 code:** `export_model.py` imports and reuses my HW4 functions (`check_frozen`, `load`,
  `FeatureBuilder`, `month_end_sessions`, `refit_dates`, `fit`, `training_rows`) without
  modifying them.
- **Documentation:** FastAPI [lifespan in tests](https://fastapi.tiangolo.com/advanced/testing-events/),
  Starlette [TestClient](https://starlette.dev/testclient/), and scikit-learn
  [model persistence](https://scikit-learn.org/stable/model_persistence.html).
- **AI assistance:** I used Claude Code (Anthropic) as a coding assistant to help build the
  logic and explanations of the code, the tests, and this README. I reviewed the work and
  checked the results reported here.

## 12. Submission checklist

| Assignment item | Where |
|---|---|
| Application, tests, Dockerfile, Makefile, dependency versions | `app.py`, `model.py`, `tests/test_app.py`, `Dockerfile`, `Makefile`, `requirements.txt` |
| README with input/output contract, setup, artifact loading, and reproduction commands | Sections 3, 5, 6, 8 |
| Saved HW4 model and preprocessing; custom modules; `example_request.json` | `artifacts/model.joblib` (the imputer and Ridge in one sklearn pipeline; no custom transformers), `example_request.json` |
| Which HW4 model version is served, with an example input | Section 2; `example_request.json` |
| Passing test and lint output; latency report | Section 7 |
| Valid and invalid requests to the container, and a request after restart | Section 9 |
| What needs attention before a real deployment | Section 10 |
| Optional CI | Not done |
