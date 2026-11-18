# HW5 — Serve the HW4 return model

This submission wraps the fitted HW4 90-calendar-day stock-return model in a FastAPI service. The service loads a saved sklearn pipeline once at startup. Requests do not train, refit, or access DuckDB.

## Model and saved artifact

- Version: `hw4-ridge-alpha1000-fit-2026-06-30`
- Fit date: 2026-06-30; Ridge alpha 1000, selected using the HW4 chronological development validation.
- Training rows: 340,637, from 2010-01-29 through signal date 2026-02-27. The latest training target ended 2026-06-01, before the fit date.
- Artifact: `artifacts/hw4_ridge_pipeline.joblib` (about 3 KiB). It includes the fitted preprocessing and estimator. `course_data.py` provides the importable `QuantileClipper` class required to load the pickle. No database is needed to serve predictions.
- The pipeline was fit under Python 3.12.9, NumPy 2.3.5, pandas 2.3.3, and scikit-learn 1.8.0. Those versions are pinned in `requirements.txt` and used in the Docker image.
- `example_request.json` and `artifacts/reference_prediction.json` record the 2026-06-30 example and prediction from a direct pipeline call before adding the API wrapper. Its provenance is documented in `artifacts/model_metadata.json`.

`train_model.py` recreates the export from the HW4 source and the full course database. It is an offline export step only; it is not copied into or run by the service. The file expects HW4 at `Homeworks/HW4/Mrityunjay_Balkrishnan/` and `equities_top2000.duckdb` at `Lectures/modeling/data/`. The generated model is already included, so these inputs are not required for running the API or tests.

## API contract

`POST /predict` accepts one JSON object with exactly these named fields, in this order. Numeric fields accept finite JSON numbers or `null`; `null` is passed to the pipeline's learned imputer. Booleans and numeric strings are rejected. `sector` is a nonempty string. Bounds reject implausible or unsupported inputs before inference.

| Field | Type and units | Accepted range |
| --- | --- | --- |
| `momentum_21` | float, 21-session adjusted-price return | -1 to 1,000 |
| `momentum_63` | float, 63-session adjusted-price return | -1 to 1,000 |
| `momentum_252_21` | float, 252-session return excluding the latest 21 sessions | -1 to 1,000 |
| `volatility_63` | float, annualized 63-session standard deviation | 0 to 1,000 |
| `drawdown_252` | float, fraction below the 252-session adjusted-price high | -1 to 0 |
| `net_margin` | float, net income / revenue | -1,000 to 1,000 |
| `operating_roa` | float, operating income / assets | -1,000 to 1,000 |
| `debt_to_assets` | float, debt / assets | -1,000 to 1,000 |
| `cash_to_assets` | float, cash / assets | -1,000 to 1,000 |
| `fcf_to_assets` | float, free cash flow / assets | -1,000 to 1,000 |
| `accruals_to_assets` | float, (net income - operating cash flow) / assets | -1,000 to 1,000 |
| `filing_age_days` | float, days since the financial filing became available | 0 to 3,650 |
| `sector` | string, sector category used by HW4 preprocessing | 1–64 characters |

The finite bounds allow extreme values (up to a 1,000x return, volatility, or ratio) while rejecting values outside a defensible domain; a ten-year filing-age limit makes stale inputs explicit. Missing feature values remain allowed because HW4's fitted imputer handles them. An invalid or extra/missing field returns 422 without calling the model. Unexpected inference errors are logged server-side and return a generic 500. A valid response has `prediction` (finite decimal return over the HW4 target horizon) and `model_version`.

## Setup and run

From this directory, use Python 3.12 and a separate environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. In another terminal, submit the saved example:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Stop the server with Ctrl+C. The API loads the model during application startup and releases it at shutdown.

## Checks and latency

With the environment active, run:

```sh
make lint
make test
```

The test suite checks a saved-model prediction against both a direct pipeline call and the recorded HW4 reference, rejects invalid requests without inference, simulates an inference failure, and times 30 successful requests after one warm-up. On the Apple M2 MacBook Air (8-core CPU, 8 GB memory), the in-process FastAPI `TestClient` run measured median **3.097 ms** and p95 **3.477 ms**. This excludes network, container, and deployment overhead; it is not a production latency claim. Details are in `results/latency_report.md`.

## Container

The Dockerfile uses `python:3.12-slim`, installs the pinned dependencies, copies the application, transformer module, and fitted artifact, and runs as a non-root user on port 8000. The build context excludes virtual environments, databases, and credentials. Reproduce on a machine with Docker Engine:

```sh
make build
make run
```

Then repeat the valid and invalid `curl` requests above against the running container. Stop it, run `make run` again, and submit the valid request once more to verify startup reloads the artifact.

Docker image build and runtime verification completed on Docker Desktop. A valid request returned 200, the invalid numeric-string request returned 422, and the valid request returned 200 again after stopping and restarting the same image. The transcript is `results/container_demo.txt`. The earlier local API transcript is retained separately as `results/local_api_demo.txt`.

## Results, limitations, and deployment follow-up

The 200 response matched the saved direct-pipeline prediction (`0.046910665365911985`) and returned the expected model version. An invalid numeric string received 422, and the valid request succeeded again after restarting the local service. The automated failure test confirmed an inference exception returns 500 without disclosing its private exception text.

Before a real deployment, add authentication/authorization, HTTPS termination, request and rate limits, structured monitoring, model/data drift checks, and a process manager/health checks. Pin the Docker base image to a specific digest for byte-for-byte environment reproducibility. Validate external feature calculations and constraints against their originating feeds; this teaching API accepts already-engineered HW4 features and does not construct them from raw market data. Keep the artifact trusted and versioned, since joblib/pickle loading can execute code.

Implementation was produced with AI assistance and reviewed against the HW5 requirements. The custom-transformer import class is adapted from `Lectures/modeling/course_data.py`; model features and training source come from this student's HW4 implementation. The FastAPI contract, service behavior, tests, and container configuration were written for this submission.
