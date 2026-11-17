# HW5: Serving the HW4 Stock-Return Model with FastAPI and Docker

Anjali Sudhakar Bandewar, UCB MFE Python Pre-Program 2026

`POST /predict` serves the fitted **HW4 HistGradientBoosting pipeline**, including its learned
`QuantileClipper` preprocessing. Each request describes one stock on one feature date, using the
11 HW4 features. The response is the model's predicted **cross-sectional percentile rank of
that stock's 90-calendar-day forward return**.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI app: input/output contract, lifespan model loading, `/predict` |
| `hw4_model.py` | Model logic: feature list, `QuantileClipper` (needed to unpickle), load and predict |
| `model/hw4_hgb_fold2015.joblib` | Fitted HW4 pipeline (206 KB, committed) |
| `model/reference_prediction.json` | HW4 reference input/prediction, hyperparameters, metrics, training versions |
| `export_model.py` | One-time script that produced the artifact from HW4 code (not used by the service) |
| `example_request.json` | Valid request: AAPL on 2015-01-02, a real HW4 holdout row |
| `tests/test_app.py` | Tests: valid prediction, validation, inference failure, latency |
| `evidence/` | Export log, test/lint transcript, HTTP transcripts, container demo script |

## Which HW4 model is served (artifact provenance)

HW4 compared momentum, Ridge and HistGradientBoosting (HGB). The backtest traded **plain
12-month momentum**, which is a ranking rule with nothing fitted, so there is no momentum
artifact to serve. The fitted model with learned preprocessing that HW4 selected is the HGB
pipeline (best ML model by mean 2012-14 rank IC: 0.023 vs Ridge 0.016). This service uses
that pipeline:

- **Pipeline:** `QuantileClipper(lower=0.01, upper=0.99)` → `HistGradientBoostingRegressor(
  learning_rate=0.05, max_leaf_nodes=15, max_iter=100, min_samples_leaf=1000,
  early_stopping=False, random_state=0)`
- **Fit:** the HW4 **2015 fold**, which was HW4's final and most recent development fit. The
  fit date is 2015-01-02. It uses 462,318 training rows with feature dates from 2010-01-04 to
  2014-09-25, and every label exited before the fit date. Training used every 5th session.
  The target is the within-date percentile rank of the forward return.
- **Model version:** `hw4-hgb-fold2015-lr0.05-leaves15-iter100`

**How it was serialized.** HW4 fitted its pipelines in memory and never saved them. Rather
than train a different model, `export_model.py` re-runs **HW4's own code**
(`model_dataset.make_fold`, `train_baselines.hgb_pipeline`) in the HW4 training environment,
using the HW4 feature and label outputs read-only. It then checks the reproduction in two
steps:

1. The recomputed 2015 holdout metrics must equal the values HW4 recorded in
   `HW4/.../results/models/selected_by_year.csv`. All four matched to full precision: IC
   0.06985711083527543, decile spread 0.04565494219581659, MSE 0.0825649134972883 and R²
   0.009220760896943903 (`evidence/export_model_log.txt`).
2. The pipeline is refitted with a verbatim copy of `QuantileClipper` placed in
   `hw4_model.py`, so the pickle doesn't need HW4's training script or duckdb. Its
   predictions must be **bit-identical** to the HW4-code pipeline on all 496,704 holdout rows.

Running the export a second time produced a byte-identical artifact (same SHA-256). HW4
files were not modified. To rebuild the artifact, run this from the repository root with the
HW4 data and `results/` present, in the HW4 environment
(`pip install duckdb==2.0.0 pyarrow==26.0.0` on top of `requirements.txt`):
`python Homeworks/HW5/Anjali_Sudhakar_Bandewar/export_model.py`.

The service only **loads** this trusted artifact, once, in the FastAPI lifespan. It never
fits, refits or queries the HW4 database.

## Input contract (`InputData`)

The body is a JSON object with **exactly these 11 keys**, all required. They are passed to
the model as a one-row pandas DataFrame with these column names in this order, which is the
HW4 training order. `null` means *missing* and becomes NaN, just as in HW4 training (the
fitted clipper and HGB handle NaN natively). Values are decimals, not percentages.

| # | Field | Type | Units / definition (HW4) | Allowed |
|---|---|---|---|---|
| 1 | `momentum_1m` | number or null | `adj_close / adj_close 21 sessions earlier − 1` | finite, > −1 |
| 2 | `momentum_3m` | number or null | same, 63 sessions | finite, > −1 |
| 3 | `momentum_6m` | number or null | same, 126 sessions | finite, > −1 |
| 4 | `momentum_12m` | number or null | same, 252 sessions | finite, > −1 |
| 5 | `universe_rank` | integer | dated quarterly top-2,000 universe rank (1 = largest) | 1 to 2000, not null |
| 6 | `profit_margin` | number or null | net income / revenue | finite |
| 7 | `ebitda_margin` | number or null | EBITDA / revenue | finite |
| 8 | `debt_to_equity` | number or null | total debt / total equity | finite |
| 9 | `asset_turnover` | number or null | revenue / total assets | finite |
| 10 | `fcf_margin` | number or null | free cash flow / revenue | finite |
| 11 | `fundamental_age_days` | number or null | days from the latest eligible filing's fiscal date to the feature date | 0 to 548 |

Fundamentals must come from the latest filing with `filing_date <= date`, as in HW4. That
point-in-time rule is the caller's responsibility; the API cannot check it.

**Why these constraints:**
- **Momentum > −1:** prices are positive. There is no upper bound, because HW4 data went up
  to +3,376 and the learned clipper caps extremes.
- **Ratios are only required to be finite:** HW4 data has very large values of both signs
  (for example, negative equity), and the clipper bounds them at the learned 1%/99% training
  quantiles.
- **Rank is 1 to 2000 and never missing:** it was never missing in HW4.
- **Age is 0 to 548:** HW4's eligibility rule.
- **Types:** the model uses `strict=True`, so numeric strings (`"0.1"`) and booleans are
  rejected rather than coerced. A float such as `4.0` for `universe_rank` is also rejected.
  JSON integers are accepted for number fields.
- **Width:** `extra="forbid"` plus the 11 required keys fixes the feature count at exactly 11.

**Returns 422 for:** a missing key, an extra key, wrong types, numeric strings, booleans,
NaN/±Infinity, values out of range, and a wrong body shape (a list, an empty body, or invalid
JSON). The 422 body lists each error's `type`, `loc` and `msg`. It deliberately doesn't echo
the input: FastAPI's default handler echoes it, and an echoed NaN can't be serialized, which
would turn a 422 into a 500.

## Output contract (`OutputData`)

```json
{"prediction": 0.4888339526811931, "model_version": "hw4-hgb-fold2015-lr0.05-leaves15-iter100"}
```

`prediction` is a finite float. It is the predicted percentile rank (about 0 = worst,
1 = best) of the stock's forward return among same-date universe stocks. The **prediction
horizon** is 90 calendar days: entry is at the next session's close, and exit is at the first
session on or after entry + 90 days. It is a **ranking score, not an expected return**. It is
useful only for comparing stocks on the same date.

**500:** any unexpected inference failure, including a non-finite model output, returns a
plain-text `Internal Server Error`. The traceback goes to the server log only, and no default
prediction is ever returned.

## Setup and commands

Requires Python 3.12 and Docker. From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the model stack to the HW4 training environment, which was Python
3.12.9 with scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, joblib 1.6.0 and
threadpoolctl 3.8.0. The Docker image installs the same pins.

| Command | What it does |
|---|---|
| `make test` | `python -m pytest -q` (the Makefile uses `.venv/bin/python` if present) |
| `make lint` | `ruff check .` and `ruff format --check .` |
| `make format` | `ruff format .` |
| `make build` | `docker build -t mfe-hw5:local .` |
| `make run` | `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` |
| `python -m uvicorn app:app --host 127.0.0.1 --port 8000` | Run without Docker; docs at http://127.0.0.1:8000/docs |

Example request:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The full container demo (build, valid request, invalid request, stop, restart the same image,
predict again) is scripted:

```sh
bash evidence/container_demo.sh 2>&1 | tee evidence/container_transcript.txt
```

**Docker image.** It is built from `python:3.12-slim`, works in `/app` and runs as a
non-root user. It contains only `requirements.txt`, `app.py`, `hw4_model.py` and `model/`.
`.dockerignore` keeps `.venv/`, `.git/`, `.env*`, databases (`*.duckdb`, `*.db`,
`*.sqlite`), `*.parquet`, keys, `data/`, `tests/`, `evidence/` and the export script out of
the build context.

## Docker results (verified 2026-11-17)

Docker Desktop 29.9.3 on an Apple M1 Mac (linux/arm64 image).

- **Build:** `make build` succeeded. The image is 770 MB, mostly the scientific Python
  stack.
- **Image contents:** inside the image, `/app` holds only `app.py`, `hw4_model.py`,
  `requirements.txt`, `model/hw4_hgb_fold2015.joblib` and `model/reference_prediction.json`.
  A filesystem search found no `*.duckdb`, `.env`, `.venv` or `*.parquet` files. The service
  runs as `appuser`.
- **Versions in the image:** scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1 and
  joblib 1.6.0, the same as HW4 training. Python is 3.12.16 from `python:3.12-slim`, versus
  3.12.9 in training. The pinned libraries are what determine whether the artifact loads,
  and the in-container prediction equals the HW4 reference exactly.

Full output is in `evidence/container_transcript.txt`:

| Step | Request | Result |
|---|---|---|
| 1 | `example_request.json` | **200** `{"prediction":0.4888339526811931,"model_version":"hw4-hgb-fold2015-lr0.05-leaves15-iter100"}` |
| 2 | `evidence/invalid_request.json` (numeric string, missing `fundamental_age_days`, extra `adj_close`) | **422** listing all three errors |
| 3 | `docker stop`, then `docker run` of the same image ID, then `example_request.json` | **200**, same prediction |
| 4 | 25 timed requests | see Latency below |

`make run` was also checked: the container started, served a 200 for `example_request.json`
and shut down cleanly on `docker stop`.

## Tests (`tests/test_app.py`, 31 passing)

1. **Valid prediction.**
   - The real artifact returns 200, the exact response schema, a finite prediction and the
     correct `model_version`.
   - The DataFrame passed to the model has the exact training columns, in training order.
   - The API prediction equals a direct pipeline call on an independently loaded artifact.
   - It also equals the **HW4 reference prediction** 0.4888339526811931 for AAPL on
     2015-01-02. `export_model.py` recorded that value from the HW4-code pipeline, after the
     HW4 metrics had reproduced and before any API code existed.
   - Tolerance is `abs=1e-12`, since the artifact, library versions and platform are the
     same, so only round-off could differ.
   - Further checks: a `null`-feature request matches the direct pipeline with NaN, and
     requests never reload or refit the model.
2. **Input validation:** 25 parametrized cases, all returning 422. A spy on the loaded
   model's `predict` shows it was **never called** for any of them.
3. **Inference failure:** with valid input, the loaded model's `predict` is replaced with one
   that raises. The test asserts a 500, that the exception text is not in the response, and
   that no prediction is returned. A model returning NaN also gives a 500.
4. **Performance:** 5 warm-up requests, then 50 sequential successful requests timed with
   `time.perf_counter`, reporting median and p95. The test asserts p95 < 250 ms. That is a
   loose regression guard for this laptop, more than 7× the slowest p95 measured here: reloading or refitting
   the model per request would take seconds and would trip it.

### Latency

| Measurement | Method | Median | p95 |
|---|---|---|---|
| In-process `TestClient` (not real HTTP) | 5 warm-up + 50 requests, `time.perf_counter` | 4.5 ms | 6.4 ms |
| **Real localhost HTTP to the Docker container** | 5 warm-up + 25 sequential `curl` requests, `time_total` | 4.6 ms | 8.8 ms |

Hardware: Apple M1 (8 cores), 8 GB RAM, macOS 26.5, Python 3.12.9. Requests were sequential
with one process and no concurrent load. Results vary between runs: an earlier run on the
same machine measured TestClient medians of 7-10 ms and p95s of 10-30 ms, and an earlier
container run measured a 6.0 ms median and a 33.9 ms p95. The tail is noisy at this sample size. The container
numbers include Docker Desktop's port forwarding.
Sources: `evidence/test_lint_transcript.txt` and `evidence/container_transcript.txt`.

## Evidence (`evidence/`)

- `export_model_log.txt`: reproduction of the HW4 metrics and the identical-predictions check
- `test_lint_transcript.txt`: `make lint`, `make test` and the latency report
- `local_uvicorn_transcript.txt`: valid (200) and invalid (422) requests over real localhost
  HTTP to a **local uvicorn** process, not the container
- `container_demo.sh` and `container_transcript.txt`: the Docker build, valid and invalid
  requests, a request after restarting the same image, and the container latency

## Before a real deployment

- **Feature pipeline:** callers must compute the 11 features exactly as HW4 did (point-in-time
  fundamentals, adjusted prices, universe rank). The service can't verify that. A real system
  would compute features server-side from a versioned feature store.
- **Model staleness:** the model was fitted on data up to 2014. It needs a retraining
  schedule, monitoring of input drift and score distributions, and a model registry rather
  than a file in Git. Its out-of-sample rank IC was small (0.07 in 2015, negative in 2013),
  and HW4 chose momentum over it for trading.
- **Output meaning:** the score is a same-date rank. It isn't a return forecast, and it isn't
  meaningful for a single stock in isolation.
- **Operations:**
  - The service has no authentication, TLS, rate limiting or request-size limits.
  - It has no health/readiness endpoint, and nothing records the model version in a
    structured log.
  - It runs a single uvicorn worker.
  - The image includes test and lint tools from `requirements.txt` (pytest, ruff), which
    should be split into a dev requirements file.
  - `python:3.12-slim` floats on the patch version, so it should be pinned by digest.
- **Security:** pickle/joblib can execute code on load, so only a trusted, checksummed
  artifact should be loaded.
- **Test tooling:** the test client prints a Starlette deprecation warning about the pinned
  `httpx`. It is harmless here, but it should be addressed when the pins are upgraded.

## Code reuse and AI assistance

- **Course code:**
  - `QuantileClipper` is copied from my HW4 `train_baselines.py`, which adapted it from
    `Lectures/modeling/course_data.py`.
  - The starting point was the course HW5 template (`app.py`, tests, Dockerfile, Makefile,
    `pyproject.toml`).
  - The 422 handler pattern follows FastAPI's documented `RequestValidationError` override.
- **AI assistance:** I used **Claude Code** (Anthropic) to:
  - inspect HW4 and identify the fitted model;
  - write `export_model.py`, `hw4_model.py`, `app.py`, the tests, the Dockerfile and the
    demo script;
  - run the checks;
  - draft this README.

  It also found and fixed the NaN-input bug that turned a 422 into a 500.
- **My role:** I am responsible for the submission and reviewed the outputs. Choosing the
  HGB 2015-fold pipeline over momentum was a judgment call: momentum has no fitted
  parameters to serve.
