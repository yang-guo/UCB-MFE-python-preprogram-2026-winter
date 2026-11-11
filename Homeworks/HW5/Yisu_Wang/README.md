# HW5 — Serve the 2026 HW4 models

This service loads the four models selected for the 2026 rolling window of HW4. They were trained on 2020–2024 signal years, validated on 2025, and predict a 60-market-session return **ranking score**. The model files and `selection.json` came from my HW4 run. `feature_schema.json` contains only the ordered feature lists and runtime settings needed to serve them; no database or training rows are packaged.

## Input and output

`POST /predict` accepts one JSON object with `features`: exactly **115** values in the order in `feature_schema.json` under `model_features`. The input represents one stock at a signal close, using features already calculated by HW4's `01_data_and_feature.ipynb`. This API does not reconstruct 240-session windows from raw prices.

`FUN_filing_age_days` and `FUN_fiscal_age_days` are calendar days; other `FUN_*` features are dimensionless ratios or changes in ratios. `PV_*` are return, volume, turnover, momentum, or correlation statistics from the market-session windows documented in HW4. `SECTOR_*` are one-hot indicators and must be 0 or 1. All numbers must be finite; JSON `null` represents a missing non-sector feature, which LightGBM handles natively. Return-like values are fractions (for example, `0.05` is 5%). We impose no global numeric cap on the other features because financial ratios and rolling statistics have different natural ranges. Strings, booleans, additional fields, and arrays of the wrong length receive HTTP 422.

The response has a finite numeric `prediction` (the raw output of `model_01`, the highest-validation-IC selected model), `model_version`, and the four raw `model_scores`. These are **not** calibrated return percentages. HW4's final trading score was obtained by clipping and standardizing each model's output across **all stocks on the same signal date**, then averaging. A single-stock request cannot reproduce that cross-sectional operation; this endpoint reports the four genuine constituent outputs without pretending to be the final portfolio score.

## Setup and checks

From the repository root, create a small input and independent reference predictions from a real 2026 HW4 feature row using the existing course environment:

```sh
./.venv/bin/python Homeworks/HW5/Yisu_Wang/make_example.py
```

This writes `example_request.json` and `tests/reference_prediction.json`. The latter records direct predictions from the four original model files **before** calling the API. The helper needs the local prepared HW4 DuckDB; neither the service nor the Docker image reads it.

Then run from `Homeworks/HW5/Yisu_Wang/`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q -s
python -m ruff check .
python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The first test compares the API with both a direct model call and the saved reference. Other tests check 422 responses without inference, a forced 500 inference failure, and 20 warmed-up requests measured in-process with `TestClient`. Copy the printed median, p95, and machine details into the results section below after running.

## Docker and live request

From this submission folder:

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

In another terminal, run:

```sh
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data '{"features":[]}'
```

The first request should return 200 and the four model scores; the second should return 422. Stop and start the **same image** again, then repeat the valid request. Save a short terminal transcript or screenshot of these three requests in the submission.

## Results and limits

On a MacBook Pro with an M1 Pro and 10 logical CPU cores, all four pytest tests passed. `ruff check .` and `ruff format --check .` passed. The warmed-up, in-process `TestClient` measurement over 20 successful requests gave a median latency of **5.57 ms** and p95 of **6.56 ms**. These figures do not include a network round trip.

`docker build -t mfe-hw5:local .` succeeded. The running container returned HTTP 200 for `example_request.json`, with model version `hw4-2026-top4-60-sessions` and prediction `0.0035800756786954642`; an invalid 0-feature request returned HTTP 422. After stopping and restarting the same image, the valid request returned HTTP 200 with the same prediction and four model scores. The three HTTP responses are recorded in `container_requests.txt`.

This is a fixed 2026 model version, not a service that automatically rolls forward each year. Production deployment would additionally need authenticated access, request-size controls, logging/monitoring, an artifact release process, and a separately specified source for live engineered features. The service does not have access to training data or a retraining path.

The feature panel and selected models are from my HW4 work. The service structure was adapted from the Lecture 9 FastAPI/Docker template; ChatGPT helped implement and review the endpoint, tests, and documentation.
