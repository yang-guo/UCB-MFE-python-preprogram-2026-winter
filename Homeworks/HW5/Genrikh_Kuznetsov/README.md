# HW5 - Genrikh Kuznetsov: serving the HW4 Ridge model

A FastAPI service that scores one stock with the **final HW4 model**, tested with pytest and packaged with Docker.

## Which model is served

HW4 ([notebook](../../HW4/Genrikh_Kuznetsov/hw4_Genrikh_Kuznetsov.ipynb)) refit a Ridge pipeline every quarter. This service serves the **last refit, made at the 2026-06-30 close**. That refit produced the scores for the final two monthly purchases (signals 2026-07-31 and 2026-08-31).

| | |
|---|---|
| `model_version` | `hw4-ridge-rank-a1e6-fit2026-06-30` |
| Model | Lecture 7 preprocessing (quantile clip → median impute + missing indicators → scale; one-hot sector) → `Ridge(alpha=1e6)` |
| Training rows | 8,038,367 eligible stock-days, 2010-01-04 to 2026-03-30, using only labels known by 2026-06-30 |
| Target | Same-day percentile rank of the 90-calendar-day forward return, minus 0.5 |
| Libraries | Python 3.12.9, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, joblib 1.5.3 (the same versions as HW4, in the venv and in Docker) |

`train_model.py` rebuilds the model from the cached HW4 panel. It then checks the rebuilt model against **all 5,951 scores HW4 saved for this fit date**, and the maximum absolute difference is **0.0**. The artifact is `artifacts/model.joblib` (5.6 KB, committed). `artifacts/model_metadata.json` records its SHA-256 and the training details. The service only loads this file; it never retrains and never reads the HW4 database.

**Custom transformer.** The pipeline contains `QuantileClipper` from `Lectures/modeling/course_data.py`. That module also imports duckdb and the data loaders, so I copied the class unchanged into [`hw4_pipeline.py`](hw4_pipeline.py) and fitted the artifact with it. Unpickling therefore only needs `hw4_pipeline.py`, which is copied into the Docker image.

## API contract

`POST /predict` takes a JSON object with **exactly these 13 fields, all required, in HW4 training order**. Extra fields are rejected. The service builds a one-row DataFrame with these column names in this order, which is the same representation HW4 trained on.

| Field | Type | Allowed | `null`? | Meaning (fractions: 0.05 = 5%) |
|---|---|---|---|---|
| `momentum_21` | float | -1 to 100 | no | 21-session price return |
| `momentum_63` | float | -1 to 100 | no | 63-session price return |
| `momentum_252_21` | float | -1 to 100 | yes | Return from 252 to 21 sessions ago |
| `volatility_63` | float | 0 to 50 | no | Annualized std. dev. of 63 daily returns |
| `drawdown_252` | float | -1 to 0 | yes | Price / 252-session high − 1 |
| `net_margin` | float | ±1e6 | yes | Net income / revenue |
| `operating_roa` | float | ±1e6 | yes | Operating income / total assets |
| `debt_to_assets` | float | 0 to 1e6 | yes | Total debt / total assets |
| `cash_to_assets` | float | ±1e6 | yes | Cash and equivalents / total assets |
| `fcf_to_assets` | float | ±1e6 | yes | Free cash flow / total assets |
| `accruals_to_assets` | float | ±1e6 | yes | (Net income − operating cash flow) / total assets |
| `filing_age_days` | int | 0 to 1000 | no | Calendar days since the latest filing became usable |
| `sector` | string | one of 11 training sectors | no | e.g. `"Technology"` (exact spelling and case) |

Response `200`: `{"prediction": <finite float>, "model_version": "hw4-ridge-rank-a1e6-fit2026-06-30"}`, plus an `X-Request-ID` header. The prediction is a **relative score** (higher = expected to rank higher in 90-day return among that day's stocks). It is not a return forecast. HW4 used it only to rank stocks.

How I chose the constraints:
- **Missing values are part of the HW4 data.** About 3% of training rows have no 1-year history, and a few have no usable filing. Those fields accept `null`, which becomes NaN and is filled by the pipeline's imputer, exactly as in training. They must still be sent explicitly, so the payload always has 13 fields. Features that were never missing for eligible stocks do not accept `null`.
- **NaN / Infinity are always rejected** (`allow_inf_nan=False`). "Unknown" is spelled `null`, never NaN.
- **Strict types** (`strict=True`): `"0.02"`, `true`, and `30.5` for an int field are rejected rather than converted. Whole JSON numbers such as `0` are accepted for float fields.
- **Bounds reject impossible values or obvious garbage, not unusual ones.** A price can't fall more than 100%, volatility can't be negative, and a drawdown can't be positive. The upper limits are loose on purpose. The training panel contains real vendor outliers (the ACM reference example has cash/assets = 63), and the pipeline clips every feature to its 1st/99th training percentiles, so the model already handles extreme values that are still possible.

Status codes: **422** for missing or extra fields, wrong types, null in a non-nullable field, out-of-range values, NaN/Infinity, unknown sector, or malformed JSON. Validation runs before the endpoint, so invalid input never reaches the model. **500** (`{"detail": "Prediction unavailable"}`) if inference raises or returns a non-finite number. The traceback goes only to the server log, tagged with the request ID.

The 422 body lists where and why validation failed but does not echo the input. **I found this bug while testing:** FastAPI's default 422 handler echoes the bad value, but NaN can't be written as JSON, so the handler itself crashed and a NaN request returned **500 instead of 422**. A custom `RequestValidationError` handler in `app.py` fixes this, and `test_non_finite_values_rejected` covers it.

`GET /health` returns the loaded model version. The interactive docs are at `/docs`.

## Files

| File | Purpose |
|---|---|
| `app.py` | Request handling: Pydantic contract, lifespan loading, `/predict`, error handling |
| `model_service.py` | Model logic with no web code: load the bundle, build the DataFrame, predict |
| `hw4_pipeline.py` | Feature list, sectors, `QuantileClipper`, pipeline builder (adapted from course code) |
| `train_model.py` | Offline: rebuild the 2026-06-30 HW4 fit, check it against HW4's scores, save artifacts |
| `artifacts/` | `model.joblib`, `model_metadata.json`, `reference_examples.json` (inputs + HW4-recorded scores) |
| `example_request.json` | AAPL on 2026-08-31, a valid request |
| `tests/test_app.py` | The four test groups (36 tests) |
| `measure_latency.py`, `container_demo.sh` | Real-HTTP latency, and the script that produced the container transcript |
| `evidence/` | `test_and_lint.txt`, `container_demo.txt` |
| `requirements.txt` / `requirements-train.txt` | Service + test tools / adds `pyarrow` for `train_model.py` only |

## Reproduce

From this folder, with Python 3.12 (on Windows: `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test        # python -m pytest -q
make lint        # ruff check . && ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # local server, docs at /docs
```

**Container** (Docker Desktop 4.99 was used here; stop anything else on port 8000 first):

```sh
make build       # docker build -t mfe-hw5:local .
make run         # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
# in a second terminal:
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

The image copies only the service code, `hw4_pipeline.py`, and the artifact. `.dockerignore` excludes `.venv`, databases, parquet files, and `.env` files. It runs as a non-root user.

**Rebuilding the artifact (optional).** The committed artifact is all the service needs. To regenerate it, run the HW4 notebook once so that `Homeworks/HW4/Genrikh_Kuznetsov/outputs/` contains `panel_equities_top2000.parquet` and `test_scores.parquet` (both Git-ignored; they need `equities_top2000.duckdb` from bCourses). Then run:

```sh
python -m pip install -r requirements-train.txt
python train_model.py            # ~2 min; asserts the HW4 scores are reproduced
```

## Results

**Tests and lint** ([full output](evidence/test_and_lint.txt)): `ruff check` and `ruff format --check` pass, and **36 passed**.

1. **Valid prediction.** Returns 200 with the right schema and model version, and the API prediction equals a direct `pipeline.predict` on the same DataFrame. The API also reproduces the **scores recorded by the HW4 notebook before the API existed** (`outputs/test_scores.parquet`) for AAPL and ACM on 2026-08-31 (ACM has null features) within `abs=1e-9`. This is the same model with the same float64 inputs, so only rounding error is expected. 1e-9 is far below the ~1e-5 gaps between stocks' scores that decide the ranking. Observed: identical.
2. **Validation.** 21 malformed payloads, plus NaN/±Infinity in a required and a nullable field, plus malformed JSON, all return 422. The model's `predict` is replaced with a function that fails the test if called, which shows that invalid input never reaches inference.
3. **Inference failure.** With valid input and `predict` patched to raise, the response is 500. The exception text is not in the response but is in the server log with the request ID. A NaN model output also gives 500.
4. **Performance** (2019 MacBook Pro, Intel i9-9980HK 8-core 2.4 GHz, 32 GB, macOS 26.6.2, Python 3.12.9; one sequential client, timed with `time.perf_counter`, after a warm-up):

   | Measurement | Requests | Median | p95 |
   |---|---:|---:|---:|
   | In-process `TestClient` (no network), `test_predict_performance` | 200 | 9.2 ms | 11.6 ms |
   | Real localhost HTTP to the Docker container, `measure_latency.py` | 200 | 12.5 ms | 15.9 ms |

   The test asserts p95 < 100 ms for the in-process measurement. That is roughly 10× the observed value, enough to catch a gross regression such as reloading or refitting per request without being flaky. Most of the time goes to pandas/sklearn overhead for a one-row DataFrame, not to the Ridge model itself.

**Container** ([full transcript](evidence/container_demo.txt)):
- The valid request returns `200` with `{"prediction":0.034079615723295494,"model_version":"hw4-ridge-rank-a1e6-fit2026-06-30"}`, identical to HW4's recorded AAPL score.
- Two invalid requests return `422`: a numeric string plus an unknown sector, and NaN plus a missing field.
- After `docker stop` and a fresh `docker run` of the same image, `/health` reports the same model version and the same request returns the same prediction. The model is packaged in the image, so restarting never retrains it or loses it.

## Before a real deployment

- **Features are not computed by the service.** Clients must compute 13 features exactly as HW4 did: market-calendar returns, filing-date-aware ratios, and the 550-day staleness rule. A real system would compute them server-side from a point-in-time data store, or share one feature library with training, to avoid training/serving skew.
- **Model staleness and monitoring.** HW4 refit quarterly, so this fit should be replaced after 2026-09-30. A real deployment needs a model registry, a scheduled retrain, and monitoring of input distributions (for example, the share of clipped or null features) and of the live rank IC.
- **Batch interface.** The strategy scores ~2,000 stocks per month. A batch endpoint, or an offline batch job, would fit that better than 2,000 single-row requests.
- **Pickle trust and versions.** joblib/pickle can execute code, so the artifact must come from a trusted, checksummed source (the SHA-256 is in the metadata). The serving image must pin the training library versions, as it does here, and should use pinned hashes (`pip --require-hashes`) and a pinned base-image digest.
- **Operations.** Auth, rate limits, HTTPS, multiple workers or replicas behind a load balancer, readiness/liveness probes, structured logs with request IDs, and CI that runs these tests and builds the image (the optional GitHub Actions extension was not done).
- **Model quality.** HW4 lost to the S&P 500 (7.4% vs 15.2% CAGR). Serving a model does not make it a good model.

## Sources and AI use

The app structure (lifespan loading, request ID, 500 handling) was adapted from the Lesson 9 demo (`Lectures/Lecture 9/demo/app.py`). `QuantileClipper` and the pipeline come from `Lectures/modeling/course_data.py`. I used Claude Code to help write the code, tests, and this README, and to run the Docker demo. I reviewed the code and the results, including the HW4 score reproduction and the NaN → 500 bug it found.
