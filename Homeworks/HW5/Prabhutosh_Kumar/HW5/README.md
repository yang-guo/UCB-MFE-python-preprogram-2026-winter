# HW5 — Prabhutosh Kumar

This service predicts the gross adjusted-price stock return over the HW4 strategy's **90-calendar-day holding period**, entering at the next market session's close and exiting at the first observed close on/after maturity (HW4 training labels allow at most five sessions of exit delay). Predictions are fractional returns: `0.03` means 3%. This is the frozen HW4 experiment, not a newly selected model.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI schemas, validation, lifespan and error handling |
| `model_service.py` | Artifact integrity/version checks and DataFrame inference |
| `preprocessing.py` | Importable HW4 quantile clipper |
| `artifacts/model.joblib` | Small fitted preprocessing + Ridge pipeline, included in Git |
| `artifacts/metadata.json` | Training versions, feature order, checksum and historical reference |
| `example_request.json` | Real HW4 input: symbol A, March 31, 2016 |
| `tests/test_app.py` | Real-model correctness, validation, failure, lifecycle and latency tests |
| `Dockerfile`, `.dockerignore` | Python 3.12 image and restricted build context |
| `verify_container.py` | Build, valid/invalid requests, stop/start, and transcript capture |
| `export_model.py`, `research/` | Offline reproduction of the locked HW4 fit; not used by the service |
| `evidence/` | Recorded checks and environment limitations |

## Model provenance

Source: branch `hw_4_Prabhutosh_Kumar`, commit `7f958c5cecf7dd066eb63704dfae7bcc4369068b`. HW4 did not persist the fitted model, so the export command reproduced its final fit using the original local database and locked configuration. It did not redo model selection or change any hyperparameters. The resulting pipeline was compared against **all 249,796 predictions previously saved by HW4**, with maximum absolute difference **1.3189319428286161e-16**. It is serialized as `hw4-ridge-alpha10-fit20160401-v1`.

Ridge uses alpha 10 and no sector encoding. Final fit date: April 1, 2016; 139,554 completed training rows; latest signal November 30, 2015; latest label completion February 29, 2016. The saved pipeline contains training-only 1st/99th percentile clipping, median imputation with missingness indicators, standard scaling, and fitted Ridge coefficients. Target clipping also follows HW4's training-only percentiles.

The trusted joblib is loaded **once per server lifespan**, relative to the module location. Its SHA256, package versions, and DataFrame feature names/order are checked at startup. A bad/missing artifact stops startup. Inference never fits a model or accesses a database. The custom clipper is copied from `Lectures/modeling/course_data.py` into an importable module so serving does not require DuckDB. Its algorithm is unchanged.

The reference prediction `0.0303447807130536` comes from the pre-existing HW4 `outputs/results/predictions.csv` row for A on March 31, 2016. The positive test compares both a direct saved-pipeline call and this separately recorded historical value. Tolerances (`rtol=1e-9`, `atol=1e-10`) permit floating-point/CSV round-trip differences while detecting material changes.

## API contract

`POST /predict` accepts exactly one required top-level field, `features`, containing **14 JSON numbers in this order**. Integers are accepted as numeric values; numeric strings, booleans, nulls, nested objects, extra fields, NaN and infinities are rejected. No feature is optional. This API requires complete observations even though the saved training pipeline retains its learned missing-value imputation.

`P(t)` below is the adjusted close on the signal session; lags count market sessions. Volatility uses sample standard deviation of daily fractional returns, annualized by `sqrt(252)`. Financial ratios use the causal filing joins from HW4; no percentages should be multiplied by 100 before sending.

| Position (1-based) | Feature | Definition / units | Allowed range |
|---:|---|---|---|
| 1 | `momentum_21` | `P(t)/P(t-21)-1`, fraction | >= -1 |
| 2 | `momentum_63` | `P(t)/P(t-63)-1`, fraction | >= -1 |
| 3 | `momentum_252_21` | `P(t-21)/P(t-252)-1`, fraction | >= -1 |
| 4 | `volatility_63` | Annualized 63-session daily-return volatility | >= 0 |
| 5 | `drawdown_252` | `P(t)/max(P over last 252 sessions)-1`, fraction | [-1, 0] |
| 6 | `net_margin` | Net income / revenue | Any finite number |
| 7 | `operating_roa` | Operating income / total assets | Any finite number |
| 8 | `debt_to_assets` | Total debt / total assets | Any finite number |
| 9 | `cash_to_assets` | Cash and equivalents / total assets | Any finite number |
| 10 | `fcf_to_assets` | Free cash flow / total assets | Any finite number |
| 11 | `accruals_to_assets` | (Net income - operating cash flow) / total assets | Any finite number |
| 12 | `filing_age_days` | Signal date minus filing availability date, calendar days | >= 0 |
| 13 | `reversal_5` | `-(P(t)/P(t-5)-1)`, fraction | <= 1 |
| 14 | `volatility_21` | Annualized 21-session daily-return volatility | >= 0 |

All numbers must be finite. Bounds follow price-return definitions, nonnegative volatility/age, and the drawdown definition. Financial ratios have no artificial [0, 1] bound: losses, negative equity effects, and unusual accounting values can produce negative or large ratios; the fitted clipper already handles training-relative extremes. `filing_age_days` accepts finite nonnegative numeric values (historical examples use whole days); no 550-day cap is imposed on filing age because HW4's stale-report rule uses fiscal age instead.

A valid request returns **200** with exactly `prediction` (finite number) and `model_version` (nonempty string):

```json
{"prediction": 0.0303447807130536, "model_version": "hw4-ridge-alpha10-fit20160401-v1"}
```

Invalid requests return **422** before model inference. Unexpected prediction errors, nonfinite model outputs, or an unexpected output shape return **500** with `{"detail":"Prediction failed"}`. Exceptions remain in server logs. Validation errors expose locations/messages/types without echoing nonfinite raw input into the JSON response.

## Setup and local run

From this folder, use Python 3.12 and a separate environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make lint
make test
make serve
```

On Windows: `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1`. Without make, use `python -m ruff check .`, `python -m ruff format --check .`, `python -m pytest -q`, and `python -m uvicorn app:app --host 127.0.0.1 --port 8000`.

Open `http://127.0.0.1:8000/docs` or send:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data '{"features":[0.1]}'
```

## Container build and evidence

Start Docker Desktop (or another Docker engine). Stop the local server before using the same host port.

```sh
make build
make run
```

These build `mfe-hw5:local` with `python:3.12-slim` and bind host `127.0.0.1:8000` to container port 8000. The image runs as an unprivileged user. Only service modules, dependency requirements and the fitted artifact enter the build context; local environments, research, databases and credentials are excluded.

For an automated, checked demonstration, stop the manually started container, then run:

```sh
python verify_container.py
# If host port 8000 is occupied:
python verify_container.py --port 8001
```

This builds once, records the image ID, runs the image, saves valid 200 and invalid 422 responses, stops and starts the same container, then asserts the repeated prediction is identical. It removes only the container it created. Actual command output and HTTP requests/responses go to `evidence/container.txt`. A failed check exits nonzero; inspect the transcript rather than treating a partial file as success.

**Container verification passed on November 16, 2026.** The image built successfully on Docker Desktop (Linux arm64); valid input returned 200, wrong-width input returned 422, and the prediction remained identical after stopping and starting the same container. The full build, image ID, requests, responses, restart, logs and cleanup are recorded in [evidence/container.txt](evidence/container.txt). The verified prediction was `0.030344780713053604` with model version `hw4-ridge-alpha10-fit20160401-v1`. The test container was removed after verification; the image remains available as `mfe-hw5:local`.

On this Mac, Docker Desktop installed its CLI outside the current terminal PATH. If `docker` is not found, open a new terminal or run `export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"` before the commands above.

## Tests and latency

Run `python -m pytest -q -s` to show the latency report. Tests cover real model predictions, historical-reference agreement, all documented invalid cases with an inference spy, deliberate model failure, malformed model outputs, one load per lifespan, and checksum verification. The latency test warms up five times and times 50 successful sequential requests with `time.perf_counter`; median and p95 include validation, inference and response serialization. This is **in-process TestClient**, not localhost networking or container latency. There is no hardware-independent performance cutoff. Recorded output and hardware details are in `evidence/`.

## Reproducing the export (optional; not part of serving)

The included artifact is sufficient for testing and serving. Re-exporting needs the course root environment (including DuckDB), the bCourses database at `Lectures/modeling/data/equities_top2000.duckdb`, the unchanged course `course_data.py`, and the original local HW4 `outputs/results/predictions.csv`. From the repository root:

```sh
.venv/bin/python Homeworks/HW5/Prabhutosh_Kumar/HW5/export_model.py
```

On Windows replace the Python path with `.venv\Scripts\python.exe`. The copied HW4 research data loader only adjusts its repository-root path to account for this folder's depth. The exporter uses the locked alpha, reuses the exact HW4 training function, checks the historical predictions, and saves only the small fitted artifact, metadata and example. It does not modify HW4 outputs. The `research/results` files preserve the original locked config, fit audit and training-run metadata.

## Before a real deployment

Add authentication, TLS, request limits and timeouts, structured monitoring and model/data drift checks. Promote immutable, verified artifacts through a trusted release process; joblib must never load user-supplied files. Pin the image digest and resolved dependencies, scan dependencies, and benchmark concurrent real HTTP traffic on the deployment hardware. Review stale financials, feature availability, corporate actions, market-calendar alignment, and missing-observation policy. The HW4 model's weak predictive evidence and market-data limitations remain; API correctness does not establish investment usefulness.

## Sources and assistance

Adapted from this assignment's `Homeworks/HW5/template`, my HW4 code at the commit above, and `Lectures/modeling/course_data.py` (feature engineering and preprocessing). OpenAI Codex assisted with implementation, reproduction checks, testing and documentation. No synthetic lecture model was substituted. Review the feature definitions, fit provenance and limitations before submitting.
