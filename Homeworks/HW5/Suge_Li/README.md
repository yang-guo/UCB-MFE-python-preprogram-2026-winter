# HW5: serve the HW4 ridge-rank model

This service loads one fitted Homework 4 pipeline and returns its prediction. It does not retrain, open the equity database, or apply the backtest.

Homework 4 froze `mom_roa` for the reported backtest. The model served here is the `ridge_rank` candidate from that same experiment: `Ridge(alpha=100)` after the lecture preprocessing, fit on development labels only.

## Model version

`hw4-ridge-rank-dev-labels-before-2016-04-01`

| Choice | Value |
|---|---|
| Panel | `equities_top2000.duckdb` month-end rows through 2016-04-15, built by HW4 `batched_month_end_panel` |
| Signals | 2010-01-01 through 2015-12-31 |
| Labels | 90-calendar-day forward return, and the label must end before 2016-04-01 |
| Screen | Eligible name, universe rank ≤ 400, unadjusted close ≥ $5, dollar volume ≥ $1,000,000 |
| Row filter | Raw 90-day return inside that sample's 1st–99th percentile (`in_fold_keep`) |
| Target | Same-date percentile rank of the 90-day return (`rank_target`), not the return itself |
| Features | The 12 names below, in this order. Sector is not an input |

The checked-in example is one finite training row (the middle row after sorting by date and symbol). `train_artifact.py` called `pipeline.predict` on that row before this API existed and wrote `artifacts/reference_prediction.json`. `example_request.json` is that same feature dict.

Learned clip bounds, row count, and library versions are in `artifacts/model_metadata.json`.

## Input contract

`POST /predict` takes a JSON object with these fields and no others. Values are JSON numbers. Numeric strings and booleans are rejected. Missing fields, extra fields, nulls, non-finite numbers, and a bare list (any length) are rejected with **422**. Integers are accepted and read as floats.

| Field | Unit | Allowed range |
|---|---|---|
| `momentum_21` | 21-session adjusted-close return, fraction | (−1, 100] |
| `momentum_63` | 63-session adjusted-close return, fraction | (−1, 100] |
| `momentum_252_21` | return from 252 sessions ago to 21 sessions ago, fraction | (−1, 100] |
| `volatility_63` | std of 63 daily adjusted returns, times sqrt(252) | [0, 100] |
| `drawdown_252` | adjusted close / 252-session max close − 1 | [−1, 0] |
| `net_margin` | quarterly net income / revenue | [−1000, 1000] |
| `operating_roa` | quarterly operating income / total assets | [−100, 100] |
| `debt_to_assets` | total debt / total assets | [−10, 1000] |
| `cash_to_assets` | cash and equivalents / total assets | [−10, 100] |
| `fcf_to_assets` | free cash flow / total assets | [−100, 100] |
| `accruals_to_assets` | (net income − operating cash flow) / total assets | [−100, 100] |
| `filing_age_days` | calendar days since the filing became available | [0, 10000] |

These ranges are the domain of the definitions. The saved `QuantileClipper` then winsorizes each column to the 1st and 99th percentiles learned on the training rows. Sending a value inside the table and outside those percentiles is valid; the stored cutoff is applied, not refit.

## Output contract

**200**

```json
{"prediction": 0.5, "model_version": "hw4-ridge-rank-dev-labels-before-2016-04-01"}
```

`prediction` is a finite float: the model's score for the same-date percentile rank. It is not clipped to [0, 1] after the linear model. `model_version` is the string above.

**422** for a contract violation. The body is FastAPI's validation error. Inference is not called.

**500** `{"detail": "Prediction unavailable"}` if the loaded model raises or returns a non-finite number. The traceback stays in the server log.

## Setup

From this directory, with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`.

`requirements.txt` pins the HW4 training stack: scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3. Loading the artifact with a different scikit-learn is not supported.

The service reads `artifacts/model.joblib` relative to `app.py`. `preprocessing.py` must be importable as `preprocessing`, because that is where `QuantileClipper` lives in the file. No database and no credentials are required to serve or test.

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

`make test`, `make lint`, `make format`, `make build`, and `make run` call the same tools. Activate the virtual environment first so `python` is that interpreter. Interactive docs are at `http://127.0.0.1:8000/docs`.

## Reproduce the artifact

The artifact in `artifacts/` is already the fitted model. Regenerating it needs this repository, `Lectures/modeling/data/equities_top2000.duckdb`, and the `hw_4_Suge_Li` branch object (the script reads `features.py` from that revision). From the repository root, with the environment above:

```sh
python Homeworks/HW5/Suge_Li/train_artifact.py
```

That rewrites `artifacts/model.joblib`, `artifacts/model_metadata.json`, `artifacts/reference_prediction.json`, and `example_request.json`.

## Container

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The process listens on port 8000 inside the container. The host port is bound to 127.0.0.1 only. Stop any other process on host port 8000 first, or change the host port: `-p 127.0.0.1:8001:8000`.

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Build context is this directory. The image contains the application, `preprocessing.py`, and `artifacts/`. It does not contain `.venv`, the database, or tests. Transcripts of a valid request, an invalid request, and a request after restarting the same image are in `evidence/`.

## Latency

`tests/test_app.py::test_predict_performance` warms up once, then times 20 successful in-process `TestClient` calls with `time.perf_counter`. It writes `evidence/latency_report.txt`. On this Mac (macOS 14.3.1, arm64, 8 CPUs) the median was 0.001925 seconds and the p95 was 0.002198 seconds. The median < 1 second assertion is a local guard for this small Ridge, not a production SLO. This measurement does not include real localhost HTTP or Docker.

## What a real deployment would still need

The endpoint assumes the caller already built the 12 features with the Homework 4 definitions. It does not check point-in-time eligibility, the rank cap, or the top-15 portfolio rule. Clip bounds, medians, and coefficients stay frozen until someone refits, saves a new artifact, and rebuilds the image. The service has no authentication, rate limit, or request size limit. `joblib.load` should keep loading only this artifact; an untrusted pickle is code execution. Each worker process loads its own copy. Redeploy with the same scikit-learn 1.8.0.

## Code map

- `app.py` — request contract, startup load, `POST /predict`
- `scoring.py` — one-row DataFrame in training column order, finite prediction
- `preprocessing.py` — `QuantileClipper` and the Ridge pipeline. The clipper is copied from `Lectures/modeling/course_data.py` so the container does not import the lecture package
- `contract.py` — feature order, version string, and numeric bounds
- `train_artifact.py` — one-time fit. Row selection follows HW4 `fit_ridge` / `labeled_development`
- `tests/test_app.py` — valid prediction against the saved reference and a direct `pipeline.predict`, rejected input that does not reach the model, a mocked inference failure, and the latency measurement

The FastAPI lifespan and the 500-response test shape follow `Lectures/Lecture 9/demo/`. The served model is the Homework 4 Ridge pipeline, not the lesson's synthetic demo. An AI coding assistant helped structure the service, tests, and container files. The fitted coefficients and the reference prediction come from running `train_artifact.py` on `equities_top2000.duckdb`.
