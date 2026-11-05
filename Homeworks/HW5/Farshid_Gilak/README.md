# HW5: Serving the HW4 stock-return model (Farshid Gilak)

This service serves the fitted model and learned preprocessing from my [HW4](../../HW4/Farshid_Gilak/README.md) experiment through a FastAPI `POST /predict` endpoint. The service is tested and packaged with Docker.

## Model being served

| | |
|---|---|
| Model version | `hw4-ridge-alpha1e7-rank-fit2026-07-06` |
| What it is | The last quarterly walk-forward refit of the HW4 final test. This is the model that scored the HW4 signals from July 6 to September 28, 2026. |
| Pipeline | `ColumnTransformer` [numeric: `SimpleImputer(median, add_indicator=True)` → `StandardScaler`; sector: `OneHotEncoder(handle_unknown="ignore")`] → `Ridge(alpha=1e7)` |
| Target | Same-date percentile rank of the 90-calendar-day holding return (next-session close to the first close on or after 90 days) |
| Training data | 1,671,038 rows from HW4 signals dated January 4, 2010 to March 30, 2026. Every label ended before the fit date: the last on June 29, 2026, the fit date was July 6, 2026. |
| Artifact | [artifacts/model.joblib](artifacts/model.joblib) (6 KB, in Git): a dict with `pipeline`, `model_version`, and `features`. [artifacts/model_metadata.json](artifacts/model_metadata.json) records provenance and library versions. |

The pipeline uses only built-in sklearn transformers, so no custom modules are needed.

**Regenerating the artifact.** [export_model.py](export_model.py) imports the HW4 modules `hw4_data.py` and `hw4_model.py` without changing them. It rebuilds the HW4 panel from `equities_top2000.duckdb` and refits only the last walk-forward model. HW4's `walk_forward` selects that model's training rows from the fit date alone (signals since 2010, `label_end` < fit date), so the result is the same model as in the HW4 notebook. The script also writes [example_request.json](example_request.json) and [tests/hw4_reference.json](tests/hw4_reference.json). It must run in the HW4 training environment (the repository `.venv`, which includes `duckdb`), with the HW4 code checked out:

```sh
../../../.venv/bin/python export_model.py --hw4-dir ../../HW4/Farshid_Gilak   # about 30 s
```

**Versions.** HW4 was trained with Python 3.13.5. The assignment requires Python 3.12 for serving. The libraries that matter for loading the artifact are identical in both environments: scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3, and threadpoolctl 3.6.0. They are pinned in [requirements.txt](requirements.txt). The service runs on Python 3.12 both locally and in Docker. Its prediction for the reference input equals the HW4 value exactly.

## Input/output contract

`POST /predict` takes one JSON object with **24 named fields**: 23 numeric features plus `sector`. Extra fields are rejected. The order below is the training column order. The service builds a one-row `DataFrame` with these column names in this order, which is what the HW4 pipeline was fitted on. At startup, the service also checks that the artifact's `feature_names_in_` matches this order exactly.

**Units.** Each numeric field is the stock's **percentile rank among the same signal date's eligible top-2000 members**, computed as in HW4 with pandas `rank(pct=True)`. This is the representation the pipeline was trained on. Ranking needs the whole cross-section for that date, so the client computes it. It is a same-date transformation and nothing in it is learned.

| Field | Underlying HW4 feature | Type | Allowed | `null`? |
|---|---|---|---|---|
| `mom_21` | 1-month (21-session) return | number | 0 < x ≤ 1 | no |
| `mom_63` | 3-month return | number | 0 < x ≤ 1 | no |
| `mom_126` | 6-month return | number | 0 < x ≤ 1 | yes |
| `mom_252_21` | 12-1 month momentum | number | 0 < x ≤ 1 | yes |
| `vol_63` | 63-session annualized volatility | number | 0 < x ≤ 1 | no |
| `drawdown_252` | Distance from 252-session high | number | 0 < x ≤ 1 | yes |
| `beta_252` | 252-session beta to `^SP500TR` | number | 0 < x ≤ 1 | yes |
| `max_ret_21` | Largest daily return in 21 sessions | number | 0 < x ≤ 1 | no |
| `gp_to_assets`, `roa`, `net_margin`, `accruals`, `debt_to_assets`, `cash_to_assets`, `asset_growth`, `sales_growth`, `share_growth` | Financial-statement ratios (filing date + 1 day) | number | 0 < x ≤ 1 | yes |
| `earnings_yield`, `book_to_market`, `sales_to_price`, `fcf_yield`, `log_mcap` | Value and size | number | 0 < x ≤ 1 | yes |
| `filing_age_days` | Days since latest filing became available | number | 0 < x ≤ 1 | yes |
| `sector` | Snapshot sector (categorical) | string | One of: Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities | no |

**Why these constraints.**

- **Ranks lie in (0, 1].** A percentile rank cannot be 0, can be exactly 1, and nothing outside this range was seen in training. The range check also catches a common client error: sending raw values, such as a log market cap of 23.5, instead of ranks.
- **Nulls.** Every field is required, but `null` is allowed exactly for the 19 features that had missing values in training. These are the features for which the fitted imputer learned medians and missing-value indicators. The other four features (`mom_21`, `mom_63`, `vol_63`, `max_ret_21`) are always present for an HW4-eligible stock, because eligibility requires 63 trailing returns. For them, `null` is rejected.
- **Strict types.** Numeric strings such as `"0.5"` and booleans are rejected rather than converted. JSON integers are accepted, because `1` is a valid rank. `NaN` and `±Infinity` are rejected.
- **Sector.** Only the 11 categories the fitted encoder knows are accepted. Any other value, including HW4's `"Unknown"` placeholder, which never occurred in training, would be silently encoded as all zeros, so it is rejected.

**Responses.**

- **200** `{"prediction": float, "model_version": str}`. `prediction` is finite. It is the predicted percentile rank of the stock's 90-day return among that date's members: 0.5 is average and higher is better. It is a cross-sectional ranking score, not an expected return.
- **422** for any contract violation. Errors list `type`, `loc`, and `msg`. The rejected input is not echoed back: FastAPI's default handler echoes it, and a `NaN` input would then make the 422 response itself fail to serialize and become a 500.
- **500** with `{"detail": "Prediction unavailable"}` when inference raises or produces a non-finite value. The traceback is logged on the server and never sent to the client.

## Setup and running

From this folder, with Python 3.12 (if `python3.12` is not on your `PATH`, `uv python find 3.12` prints its location):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000     # docs at http://127.0.0.1:8000/docs
```

The model is loaded once per process in the FastAPI lifespan, from `artifacts/model.joblib`. The path is resolved relative to `app.py`. The service never retrains and never queries the database.

## Tests ([tests/test_app.py](tests/test_app.py))

1. **Valid prediction.** Using the real saved model, `test_predict_valid_input` checks:
   - status 200, the exact response keys, and a finite float prediction;
   - the API prediction agrees with a direct `pipeline.predict` on the same `DataFrame` (tolerance 1e-12);
   - the prediction matches the **HW4 reference**: 0.5118015993564016 (tolerance 1e-9).

   The reference comes from `hw4_model.walk_forward`, which `export_model.py` called before any API code existed. It is the HW4 score of ALL (Allstate), the top-ranked stock on the last HW4 signal date, September 28, 2026, and the input is that stock's ranked HW4 feature row ([example_request.json](example_request.json)). The tolerance allows for floating-point differences between Python and BLAS builds. On a 0–1 rank scale, an error of 1e-9 is negligible. `test_predict_missing_features_are_imputed` sends `null` for all 19 nullable features and checks that the API still agrees with the pipeline.
2. **Input validation.** 19 parametrized cases must each return 422:
   - empty body; list body;
   - one feature missing or one extra; a fixed-length feature list;
   - a numeric string; a boolean; `null` for a required feature;
   - `NaN`, `Infinity`, `-Infinity`, sent as raw JSON tokens;
   - ranks of 0, 1.5, and −0.2; a raw log market cap;
   - an unknown sector, a wrong-case sector, a numeric sector, a missing sector.

   In each case, the loaded model's `predict` is replaced with a function that fails the test, which proves that invalid input never reaches inference.
3. **Inference failure.** Valid input is sent with the loaded model's `predict` patched to either raise or return `NaN`. Both cases must return 500 with no prediction in the body.
4. **Performance.** After 5 warm-up requests, 50 successful requests are timed with `time.perf_counter`.

**Test and lint output** (`make lint`, `make test`, with `.venv` active):

```text
python -m ruff check .
All checks passed!
python -m ruff format --check .
3 files already formatted
python -m pytest -q
.......................
In-process TestClient latency over 50 requests on macOS-27.2-arm64-arm-64bit (arm64), Python 3.12.9: median 3.51 ms, p95 4.03 ms
.                                                 [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1
  .venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
24 passed, 1 warning in 1.26s
```

The single warning comes from the course-pinned `fastapi`/`httpx` versions in the template, not from this code.

**Latency report.**

| | |
|---|---|
| Method | In-process `TestClient`, not real localhost HTTP. The timing covers request validation, the one-row `DataFrame`, the pipeline, and response serialization, with no network. |
| Run | 5 warm-up requests, then 50 timed requests, all with status 200 |
| Hardware | Apple M1 Max (10 cores, 32 GB), macOS 27.2, Python 3.12.9 |
| Result | **median 3.51 ms, p95 4.03 ms** |

The test asserts p95 < 50 ms. That is about 12 times the observed p95, which leaves room for a busy laptop or a slower CI runner. It still fails if, for example, someone moves model loading or extra work into the request path.

## Container

```sh
make build    # docker build -t mfe-hw5:local .
make run      # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The [Dockerfile](Dockerfile) uses `python:3.12-slim` and `/app` as its working directory. It installs the pinned `requirements.txt` and copies only `app.py` and `artifacts/`. [.dockerignore](.dockerignore) keeps `.venv`, `.git`, caches, `.env` files, and databases out of the build context.

To reproduce the build, another person needs only this folder and Docker. The artifact is in Git, so neither the database nor HW4 is required. The image was built and tested with Docker 28.4.0 on Apple silicon (linux/arm64). Stop any other service on host port 8000 first, or map a different host port, for example `-p 127.0.0.1:8001:8000`.

**Requests to the container.** See [container_transcript.txt](container_transcript.txt), recorded against the running image:

- a valid request: 200, with prediction 0.5118015993564016 and the model version;
- an invalid request with `mom_21` sent as the string `"0.257"`: 422;
- `docker stop`, then a fresh `docker run` of the same image, then the same valid request: 200 with the identical prediction.

The container was started in detached mode (`-d`) so that the whole session fits in one transcript. Otherwise the command is the same as `make run`.

## Before a real deployment

- **Feature pipeline.** Clients must send HW4's same-date percentile ranks. A production system should compute these itself, from the full cross-section of point-in-time data, in one shared, versioned feature job. That would remove the risk of training/serving skew and of clients sending raw values that happen to fall inside (0, 1].
- **Model lifecycle.** HW4 refits quarterly. Each refit needs:
  - a versioned artifact registry, with checksums or signatures, because joblib loads arbitrary code;
  - a rollback plan;
  - monitoring of input distributions, prediction drift, and realized rank IC once 90-day outcomes complete.
- **Known data issues from HW4.** Non-common instruments and stale delisted names affect the inputs. They should be screened upstream.
- **Operations.**
  - Run as a non-root user with a smaller runtime image, and keep `pytest` and `ruff` out of it.
  - Add health and readiness endpoints, structured logs with request IDs, authentication, rate limits, and request-size limits.
  - Set worker counts and resource limits.
  - Measure latency over real HTTP under concurrent load.
- **Batch interface.** The strategy scores about 2,000 stocks per signal date. A batch endpoint, or an offline scoring job, would suit that better than one request per stock.

## Reused code and AI assistance

- `app.py` adapts the structure of [Lectures/Lecture 9/demo/app.py](../../../Lectures/Lecture%209/demo/app.py): lifespan loading, a strict `ConfigDict`, a `FiniteFloat` output, and a logged 500 response. The tests adapt the "invalid input never reaches the model" pattern from the [demo tests](../../../Lectures/Lecture%209/demo/tests/test_app.py). `export_model.py` follows the bundle-plus-metadata layout of the demo's `train_model.py`. The model code itself is my HW4 code, used unchanged.
- **AI assistance.** I developed this submission with the help of Claude Code. It helped write and debug the service, tests, Dockerfile, and export script, and it helped prepare this README.
