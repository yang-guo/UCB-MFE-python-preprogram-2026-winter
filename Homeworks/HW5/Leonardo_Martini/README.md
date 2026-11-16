# HW5 - Leonardo Martini

FastAPI service for my HW4 model, with tests and a Docker image.

## Model served

- File: `model/model.joblib`, the last model fitted in my HW4 run (`Homeworks/HW4/Leonardo_Martini/results/model.joblib`, PR #245). It is the full sklearn pipeline: preprocessing (quantile clipping, median imputation, scaling, sector one-hot) + `HistGradientBoostingRegressor`, refit on 2026-01-30.
- Version returned by the API: `hw4-boosting-rank90d-refit-2026-01-30-854fd7bce9f9` (the last part is the start of the file's SHA-256).
- Saved with scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.7.0. `requirements.txt` and the Docker image use the same versions.
- The pipeline uses `QuantileClipper` from the course file `Lectures/modeling/course_data.py`. Pickle needs it under the name `course_data`, so `course_data.py` here is a copy of that class only.
- The service only loads the file (once, at startup). It does not retrain or read the HW4 database.

## Input / output contract

`POST /predict` with one stock on one month-end date. Features are computed as in HW4 (`course_data.py` of the course).

| Field | Type | Allowed | Meaning |
|---|---|---|---|
| momentum_21 | number | -1 to 100 | 21-session return (0.05 = +5%) |
| momentum_63 | number | -1 to 100 | 63-session return |
| volatility_63 | number | 0 to 100 | annualised volatility, 63 sessions (0.3 = 30%) |
| momentum_252_21 | number or null | -1 to 100 | 12-1 month return |
| drawdown_252 | number or null | -1 to 0 | drawdown from the 252-session high |
| net_margin, operating_roa, debt_to_assets, cash_to_assets, fcf_to_assets, accruals_to_assets | number or null | -1000 to 1000 | ratios from the last filed report |
| filing_age_days | number | 0 to 3650 | days since that report was filed |
| sector | string | one of the 11 HW4 sectors | e.g. "Technology" |

- All fields are required. `null` is allowed only where HW4 had missing values; the pipeline imputes them.
- Returns are decimals, not percent. A return below -100% or a negative volatility is impossible, so it is rejected. Ratios beyond ±1000 are data errors (the model clips extreme values anyway).
- Numeric strings, booleans, NaN/inf, extra fields and unknown sectors are rejected with **422**.
- Output: `{"prediction": float, "model_version": str}`. The prediction is the expected rank (0 = worst, 1 = best) of the stock's 90-day return among the month's stocks, because HW4 trained on the rank.
- **500** if the model fails during inference (details only in the server log).

Example: `example_request.json` (AAPL, 2026-08-31, from the HW4 panel). Expected prediction: 0.5361978083235462.

## Setup and tests

With Python 3.12 (in Git Bash on Windows, using uv):

```bash
uv venv -p 3.12 .venv
source .venv/Scripts/activate      # Linux/Mac: source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest -q                # make test
python -m ruff check . && python -m ruff format --check .   # make lint
python -m uvicorn app:app --host 127.0.0.1 --port 8000      # docs at /docs
```

Tests (`tests/test_app.py`):

1. Valid input: 200, response schema, same value as calling the pipeline directly, and same value as the reference recorded from HW4 (`tests/reference.json`). Tolerance 1e-9, because the library versions are the same, so only floating point noise is possible.
2. Invalid input (17 cases + NaN/inf): 422, and the model's `predict` is never called.
3. Model failure: `predict` replaced with a function that raises, valid input, 500.
4. Latency: 5 warm-up requests, then 50 timed requests with `time.perf_counter`.

Results: `24 passed`, ruff check and format pass. Latency (in-process `TestClient`, not real HTTP, x86_64 Linux, Python 3.12): median 9.2 ms, p95 12.4 ms. The test fails only if p95 is above 1 s, which is a generous limit for a laptop.

## Docker

```bash
docker build -t mfe-hw5:local .     # make build
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local   # make run
```

The image (`python:3.12-slim`) contains only `requirements.txt`, the app code and the model. `.dockerignore` keeps out `.venv`, databases, secrets and the tests.

`bash demo_requests.sh` builds the image, sends a valid and an invalid request, restarts the container and sends the valid request again. The output is saved in `evidence/container_requests.txt`.

## Before a real deployment

- The features must be built by the same code as HW4 on fresh data; now the client has to send them.
- Retrain on a schedule, keep model versions in a registry and monitor the input distribution and the predictions.
- Add authentication, HTTPS, rate limits, a health endpoint and structured logs.
- Run several workers and set resource limits; run the container as a non-root user.
- The pickle file must come only from trusted sources.

## Sources and use of AI

- `course_data.py`: copied from the course (`Lectures/modeling/course_data.py`).
- Structure based on the course template and the Lesson 9 demo.
- I used Claude to help me write and test the code and this README. I checked every step and can explain it.
