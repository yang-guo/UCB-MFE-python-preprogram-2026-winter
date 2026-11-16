# HW5 – Serving the HW4 model

Antoine Amiel

A FastAPI service, tested and packaged with Docker, that serves the stock-return model of my HW4 submission ([`Homeworks/HW4/Antoine_Amiel`](../../HW4/Antoine_Amiel/README.md)).

## Model served

| | |
|---|---|
| Model version | `hw4-ridge-fit-2026-07-01` |
| Pipeline | `SimpleImputer(strategy="median", add_indicator=True)` then `Ridge(alpha=1000)` (HW4 `make_model("ridge")`) |
| Fit | The last quarterly refit of the HW4 final test, made after the close of 2026-07-01 and used for every signal from 2026-07-01 to 2026-09-30 |
| Training rows | 1,608,045 (weekly signal dates from 2010-01-04 whose 90-day outcome ended by 2026-06-24), equal to the count in HW4 `results/final/fit_audit.csv` |
| Target | Same-date percentile rank of the 90-day forward return |
| Artifact | [`artifacts/model.joblib`](artifacts/model.joblib) and [`artifacts/model_metadata.json`](artifacts/model_metadata.json) (feature order, nullable features, SHA-256, library versions) |

The learned preprocessing (training medians and missing-value indicators) is inside the pipeline, so it is saved with the model. The model uses only scikit-learn classes, so the service needs no custom module.

[`export_model.py`](export_model.py) rebuilds this fit with the unchanged HW4 code (`WalkForwardScorer.model_for`) and checks that its number of training rows equals the one HW4 recorded. It also records two reference predictions computed by the HW4 `predict` function. To reproduce the artifact, put `equities_top2000.duckdb` in `Lectures/modeling/data/` and run, from the repository root, in the course environment that trained HW4 (`uv sync --locked`):

```sh
uv run --locked python Homeworks/HW5/Antoine_Amiel/export_model.py
```

The service only loads this artifact. It does not retrain or query the database. At startup it checks the file's SHA-256 against the metadata, the feature order, the scikit-learn version and the model's input width, and refuses to start on any mismatch.

## Input and output contract

`POST /predict` takes one JSON object with the 28 HW4 features, all required, by name. Each value is the stock's same-date percentile rank of an HW4 feature among the stocks scored on that signal date (HW4 `FeatureStore.cross_section`, top-1000 screen). Ranks are unitless and lie in (0, 1]: for n stocks, `rank(pct=True)` gives 1/n to 1.

| Field | Underlying HW4 feature | null allowed |
|---|---|:---:|
| `r_rev_5` | 5-session return (fraction) | no |
| `r_mom_21` | 21-session return (fraction) | no |
| `r_mom_63` | 63-session return (fraction) | no |
| `r_mom_126` | 126-session return (fraction) | yes |
| `r_mom_252_21` | return from 252 to 21 sessions ago (fraction) | yes |
| `r_vol_63` | annualized volatility of daily returns, 63 sessions | no |
| `r_vol_252` | annualized volatility of daily returns, 252 sessions | yes |
| `r_drawdown_252` | price / 252-session high - 1 (fraction) | yes |
| `r_max_ret_21` | largest daily return over 21 sessions (fraction) | no |
| `r_beta_252` | beta to the S&P 500 total return index, 252 sessions | yes |
| `r_idio_vol_252` | annualized residual volatility vs the index, 252 sessions | yes |
| `r_log_dollar_volume_63` | log of the median daily dollar volume, 63 sessions | yes |
| `r_gross_margin` | TTM gross profit / TTM revenue | yes |
| `r_operating_margin` | TTM operating income / TTM revenue | yes |
| `r_roa` | TTM net income / total assets | yes |
| `r_roe` | TTM net income / equity | yes |
| `r_accruals` | (TTM net income - TTM operating cash flow) / total assets | yes |
| `r_fcf_to_assets` | TTM free cash flow / total assets | yes |
| `r_debt_to_assets` | total debt / total assets | yes |
| `r_cash_to_assets` | cash / total assets | yes |
| `r_sales_growth` | year-over-year quarterly revenue growth (fraction) | yes |
| `r_asset_growth` | year-over-year total asset growth (fraction) | yes |
| `r_earnings_yield` | TTM net income / market capitalization | yes |
| `r_book_to_market` | equity / market capitalization | yes |
| `r_fcf_yield` | TTM free cash flow / market capitalization | yes |
| `r_filing_age` | days since the latest filing became available | yes |
| `r_log_size` | log market capitalization ($) | yes |
| `r_universe_rank_pct` | rank in the quarterly top-2000 list (1 = largest) | no |

Rules:

- Values must be JSON numbers with 0 < value ≤ 1. Strings (`"0.5"`), booleans, NaN and infinities are rejected rather than converted.
- `null` means missing and is allowed only for the features that were missing in some HW4 training rows. The pipeline replaces it with its training median and sets the matching missing-value indicator, as in HW4. The other features were never missing in training, because HW4 eligibility requires them, so `null` is rejected for them.
- A missing field or an extra field returns 422, so a request with 27 or 29 features is rejected before inference.
- The service rebuilds the training representation: a float32 numpy row in the order above (`model_service.FEATURE_ORDER`, equal to HW4 `RANK_FEATURES`), with NaN for null.

Response: `{"prediction": <finite float>, "model_version": "hw4-ridge-fit-2026-07-01"}`. The prediction is the predicted same-date percentile rank of the stock's 90-day return; higher means better expected relative performance. It is the raw Ridge output and is not clipped to [0, 1].

| Status | When |
|---|---|
| 200 | Valid input |
| 422 | Invalid input (rules above). The body lists the location, type and message of each error. It does not echo the input values. |
| 500 | Unexpected inference failure or non-finite prediction. The body is `{"detail": "Inference failed"}`; the traceback goes to the server log. |

[`example_request.json`](example_request.json) is AAPL on 2026-09-29, the last signal date of the HW4 test, taken from the HW4 scoring cross-section. Its HW4 prediction is recorded in [`tests/reference_predictions.json`](tests/reference_predictions.json). `GET /health` returns the loaded model version.

## Setup and commands

Python 3.12, in a separate environment. [`requirements.txt`](requirements.txt) pins the model libraries to the HW4 training environment (the course `uv.lock`: scikit-learn 1.8.0, numpy 2.3.5, scipy 1.16.3, joblib 1.5.3, pandas 2.3.3). From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test      # python -m pytest -q
make lint      # python -m ruff check . && python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # docs at http://127.0.0.1:8000/docs
```

Docker (stop anything else on host port 8000 first):

```sh
make build     # docker build -t mfe-hw5:local .
make run       # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

The image is `python:3.12-slim` with the pinned requirements, `app.py`, `model_service.py` and `artifacts/`, and it runs as a non-root user. `.dockerignore` keeps `.venv`, databases, `.env` files, tests and evidence out of the build context. `bash evidence/container_demo.sh` runs the whole Milestone 5 sequence and writes the transcript.

## Code layout

| File | Role |
|---|---|
| `app.py` | HTTP layer: `InputData` / `OutputData`, lifespan loading, `/predict`, `/health`, 422 handler |
| `model_service.py` | Model logic without HTTP: artifact checks and loading, feature order, training representation, inference |
| `export_model.py` | Builds the artifact and the reference predictions from the HW4 code |
| `tests/test_app.py` | The four test groups and a contract check |
| `tests/reference_predictions.json` | HW4 reference predictions (AAPL and a stock with missing fundamentals) |

## Tests

The lint and test output, with the latency line, is in [`evidence/test_and_lint.txt`](evidence/test_and_lint.txt) (`make evidence`).

1. Valid prediction: the real artifact returns 200 with exactly `prediction` and `model_version`. The API prediction equals a direct pipeline call and the HW4 reference values written by `export_model.py`, within 1e-9: both paths feed the same float32 row to the same pipeline. The second reference case has null fundamentals. A contract test checks that the `InputData` fields, their nullability, the feature order and the artifact metadata all agree.
2. Input validation: 12 invalid payloads (missing or extra field, 27 or 29 features, numeric string, boolean, null in a never-missing feature, 0, 1.5, a negative value, a list, an empty object) and NaN, Infinity and -Infinity tokens all return 422. The model's `predict` is replaced by a function that fails the test, which shows that no invalid request reaches inference. This test found a bug: FastAPI's default 422 handler echoes the input, and a NaN input cannot be serialized to JSON, so it returned 500. The custom handler fixes this.
3. Inference failure: with valid input, the loaded model's `predict` is patched to raise (and, in a second test, to return NaN). The service returns 500 without the exception text.
4. Performance: 5 warm-up requests, then 50 timed requests with `time.perf_counter`, in process with `TestClient` (no network and no server process). The test reports the median and p95 with the hardware in the test output. It asserts that all 50 predictions are identical and that p95 < 100 ms. One-row Ridge inference plus validation takes a few milliseconds, so this budget only catches gross regressions such as refitting or reloading the model per request. On a slower machine, override it with `LATENCY_BUDGET_MS`.

## Evidence

`make evidence` writes both files:

- [`evidence/test_and_lint.txt`](evidence/test_and_lint.txt): `ruff check`, `ruff format --check` and `pytest -q -s`, including the latency report.
- [`evidence/container_transcript.txt`](evidence/container_transcript.txt): from [`evidence/container_demo.sh`](evidence/container_demo.sh). It builds the image, runs it, sends `example_request.json` (200, prediction and model version) and [`evidence/invalid_request.json`](evidence/invalid_request.json) (`r_mom_21` sent as the string `"0.5"`: 422), stops the container, starts the same image again and repeats the valid request (200, same prediction). It ends with the container log.

## Before a real deployment

- Features: the client must send ranks computed exactly as in HW4, against the same cross-section. In production the service should compute the features itself from a point-in-time data source, so that clients cannot send inconsistent inputs.
- Model freshness: HW4 refits every quarter, and this model is the 2026-07-01 fit. A real service needs the refit pipeline, a model registry with versioned artifacts, and a rollout and rollback procedure.
- Monitoring: request rates, latency, error rates, input distributions (share of nulls, rank coverage) and the realized rank IC of the predictions, with alerts.
- Security and operations: authentication, rate limits, TLS, a request ID in logs, several workers behind a load balancer, resource limits, and a production image without the test tools (pytest and ruff are in `requirements.txt` only because the template uses one file).
- Artifacts: signed artifacts in an access-controlled store instead of Git, because `joblib.load` can run code from an untrusted file. The checksum here only detects a mismatch with the metadata.
- The HW4 model did not beat its benchmark in the final test (9.0% vs 15.2% CAGR). Serving it is an engineering exercise, not an investment recommendation.

## Reused code and AI assistance

- The structure of `app.py` (lifespan loading, strict Pydantic models, generic 500 with the traceback in the log) and of the tests (patching `predict` to show that invalid input never reaches the model) follows the Lesson 9 demo (`Lectures/Lecture 9/demo`). The contract, model loading and checks, export script and tests are specific to my HW4 model.
- This submission was written with the help of an AI assistant (Claude). I reviewed the contract, the export, the tests and the evidence, and I can explain every part of the code.
