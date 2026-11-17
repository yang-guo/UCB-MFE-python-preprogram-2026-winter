# HW5: Serving the HW4 relative-return model — Yiting Zhao

A FastAPI service that serves my HW4 model from a Docker container.

## Which model is served

The model is `final_relative_model` from `Homeworks/HW4/HW4.ipynb`, cell "Train the final Ridge model using all matured development data".

- **Pipeline**: `ColumnTransformer[SimpleImputer(median, add_indicator=True) → StandardScaler]` → `Ridge(alpha=10000)`. Preprocessing and model are saved together in one artifact.
- **Training data**: monthly signals from 2010-01-04 to 2015-12-01, 142,539 observations. The target is clipped at the 1st and 99th percentiles of the development data.
- **Target**: the stock's 90-calendar-day forward return, entered on the next trading day after the signal, **minus the median 90-day return of the universe on the same signal date**. It is a decimal (0.01 = 1%).
- **Version string**: `hw4-final-relative-ridge-a10000+sha256:<first 12 hex of artifact>`. It is returned with every prediction, so you can tell which artifact answered.

## Artifact

- `artifacts/ridge_model.joblib` is the fitted Pipeline (small, committed to Git).
- `artifacts/reference_prediction.json` holds one HW4 input, the prediction HW4 produced for it, and the library versions used in training.

Both were written by a cell I ran in HW4.ipynb immediately after fitting `final_relative_model`. The reference row is the first stock, by symbol, with all 18 features present on signal date 2016-04-01 (the first out-of-sample month).

The service loads this file once at startup in the FastAPI `lifespan`. It never refits and never touches the HW4 database. Set `MODEL_PATH` to load a different file.

joblib files run code when loaded, so load only artifacts you created or trust. The model library versions in `requirements.txt` match the HW4 environment exactly, because sklearn does not support loading pickles across versions.

## Input / output contract

`POST /predict`, JSON object with exactly these 18 keys, in this order:

| # | Field | Definition (HW4) | Allowed range |
|---|---|---|---|
| 1 | book_to_market | total_equity / (adj_close × shares_outstanding) | [-1000, 1000] |
| 2 | earnings_yield | net_income / market cap | [-1000, 1000] |
| 3 | sales_to_market | revenue / market cap | [-1000, 1000] |
| 4 | cash_flow_to_market | operating_cash_flow / market cap | [-1000, 1000] |
| 5 | momentum_1m | adj_close / adj_close 21 trading days earlier − 1 | [-1, 100] |
| 6 | momentum_3m | same, 63 trading days | [-1, 100] |
| 7 | momentum_6m | same, 126 trading days | [-1, 100] |
| 8 | momentum_12m | same, 252 trading days | [-1, 100] |
| 9 | momentum_12m_1m | price 21d earlier / price 252d earlier − 1 | [-1, 100] |
| 10 | roe | net_income / total_equity | [-1000, 1000] |
| 11 | roa | net_income / total_assets | [-1000, 1000] |
| 12 | gross_profitability | gross_profit / total_assets | [-1000, 1000] |
| 13 | operating_cash_flow_to_assets | operating_cash_flow / total_assets | [-1000, 1000] |
| 14 | net_margin | net_income / revenue | [-1000, 1000] |
| 15 | debt_to_assets | total_debt / total_assets | [0, 1000] |
| 16 | volatility_21d | std of daily returns over 21 days × √252 | [0, 50] |
| 17 | volatility_63d | same, 63 days | [0, 50] |
| 18 | volatility_126d | same, 126 days | [0, 50] |

All features are unitless decimals, computed point-in-time on the signal date. Fundamentals come from the latest statement filed on or before that date.

**Rules (otherwise 422):**

- Every key must be present.
- No extra keys are allowed (`extra="forbid"`). Because fields are named, a wrong feature count is always a 422.
- Values must be JSON numbers (`StrictFloat`). Numeric strings such as `"0.1"` and booleans are rejected rather than coerced.
- `NaN` and `±Infinity` are rejected.
- Values must fall inside the range shown in the table.
- `null` is allowed and means "not available". HW4 trained on missing values (it replaced ±inf with NaN), and the fitted median imputer with missing indicators handles them, so `null` reproduces the training representation exactly. The key itself must still be present.

**Why these ranges:** a price cannot fall more than 100%, so momentum is ≥ −1, and volatility is ≥ 0. The upper bounds are sanity limits far above the HW4 99th percentiles (momentum_12m p99 ≈ 2.6, volatility p99 ≈ 1.5, value ratios p99 < 6). The HW4 values beyond these bounds (e.g. momentum ≈ 96,000, volatility ≈ 450,000) came from bad prices or near-zero denominators, so the API treats them as input errors.

**Response (200):**

```json
{"prediction": 0.0061, "model_version": "hw4-final-relative-ridge-a10000+sha256:…"}
```

**Other status codes:**

- 422 responses list each error's `type`, `loc`, and `msg`, but do not echo the rejected value. FastAPI's default handler echoes it, and echoing `NaN` produced invalid JSON and crashed with a 500; my test suite caught this.
- 500 is returned when inference fails or produces a non-finite number. The response body is a generic `{"detail": "Inference failed"}`; the traceback goes to the server log only.
- `GET /health` reports the loaded model version.

The input is converted to a one-row pandas DataFrame with the training column names and order (`model_service.to_frame`). That is what the fitted `ColumnTransformer` expects.

## Code layout

- `model_service.py`: feature list, artifact loading (which checks that the artifact's `feature_names_in_` matches), and DataFrame construction and inference. It contains no HTTP code.
- `app.py`: lifespan, Pydantic schemas, and the endpoint.
- `tests/test_app.py`: the four required test groups.

## Setup and reproduce

From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

make lint     # ruff check + ruff format --check
make test     # pytest -q
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # docs at /docs
```

Docker (anyone with this folder can rebuild the identical image; the artifact is copied in):

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

`.dockerignore` is an allow-list, so only `requirements.txt`, the two modules, and the model file enter the build context. `.venv`, databases, and credentials never do. The container runs as a non-root user.

## Results

### Tests and lint

See `evidence/test_and_lint.txt`.

### Latency

Measured with `tests/test_app.py::test_latency_report`. The test makes 5 warm-up requests, then times 50 successful requests with `time.perf_counter`. This is an **in-process `TestClient`** measurement: no uvicorn and no network.

| Hardware | Median | p95 |
|---|---|---|
| Windows laptop, Intel 11th-gen Core (Intel64 Family 6 Model 140), Python 3.12.9 | 5.24 ms | 6.49 ms |

The test asserts a median under 100 ms. A single Ridge predict on 18 features takes well under 1 ms, so 100 ms is a loose bound on a laptop. It still catches a serious regression such as refitting inside a request, which takes seconds on 140k rows.

### Container requests

See `evidence/container_transcript.txt`. It shows:

- a valid request returning 200 with a prediction and model version,
- an invalid request returning 422,
- the same image stopped, restarted, and returning the same prediction again.

## Before a real deployment

- **Feature pipeline**: clients must compute the 18 factors exactly as HW4 did. In production the service should build them itself from a point-in-time feature store, to avoid train/serve skew.
- **Model quality**: the HW4 test-period predictions had extreme outliers (min ≈ −3.6e8) caused by unbounded raw ratios. The input bounds here reduce the risk. A real deployment needs winsorized features at training time and monitoring of the input distribution and prediction drift.
- **Artifact management**: use versioned storage or a model registry, signed or checksummed artifacts, and a non-pickle format if artifacts can come from untrusted sources.
- **Operations**: authentication, rate limiting, HTTPS, structured logging and metrics, multiple workers behind a load balancer, health and readiness probes, pinned base-image digest and lockfile, and CI that runs lint, tests, and the Docker build.

## AI use and citations

I used Claude (Anthropic) to draft the service, tests, and Docker setup, following the Lesson 9 demo structure (lifespan model loading, `TestClient` fixtures, and `raise_server_exceptions=False` for the 500 test). I checked the code against my HW4 notebook: the feature list and order, the preprocessing, and the target. I then ran the tests, lint, and container requests myself; the outputs are in `evidence/`.
