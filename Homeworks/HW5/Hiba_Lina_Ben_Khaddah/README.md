# HW5 — Stock-ranking model service

This project serves the fitted model from my HW4 experiment through a FastAPI endpoint and a
reproducible Docker image. The service loads the saved pipeline once at startup; it does not
retrain the model or query the HW4 database.

## Model and artifact

- **Version:** `hiba-hw4-rank-tree-v1`
- **Model:** `HistGradientBoostingRegressor` with its fitted preprocessing pipeline
- **Training cutoff:** March 31, 2016
- **Output:** predicted within-date percentile rank of the future 90-calendar-day return, minus 0.5. A higher score means the stock ranks more favorably; it is not a predicted return.

The trusted fitted pipeline is in `artifacts/model.joblib`. `course_data.py` contains the `QuantileClipper` required to load that pipeline. `artifacts/model_metadata.json` records the feature order and library versions. The API resolves these paths relative to `app.py`, so it works independently of the caller's current directory.

The reference input comes from stock `A` at the March 31, 2016 signal date in my HW4 `02_final_backtest.ipynb`. Before the API was written, its prediction was recorded as `0.02147808903157589` in `artifacts/reference_prediction.json`. The serialized pipeline and API reproduce that value.

## Input contract

Send one JSON object to `POST /predict`. All fields are required. Numeric fields are strict, finite numbers; numeric strings and booleans are rejected. Fractions use `0.10` to mean 10%.

| Field | Type | Meaning and units | Allowed range |
|---|---|---|---|
| `momentum_21` | number | Trailing 21-session return, fraction | −1 to 20 |
| `momentum_63` | number | Trailing 63-session return, fraction | −1 to 20 |
| `momentum_252_21` | number | Return from session −252 to −21, fraction | −1 to 20 |
| `volatility_63` | number | Annualized 63-session volatility, fraction | 0 to 10 |
| `drawdown_252` | number | Drawdown from the 252-session high, fraction | −1 to 0 |
| `net_margin` | number | Net income / revenue, fraction | −10 to 10 |
| `operating_roa` | number | Operating income / assets, fraction | −10 to 10 |
| `debt_to_assets` | number | Debt / assets, fraction | 0 to 10 |
| `cash_to_assets` | number | Cash / assets, fraction | 0 to 10 |
| `fcf_to_assets` | number | Free cash flow / assets, fraction | −10 to 10 |
| `accruals_to_assets` | number | Accruals / assets, fraction | −10 to 10 |
| `filing_age_days` | number | Days from filing availability to signal date | 0 to 550 |
| `sector` | string | Learned sector category | See list below |

Allowed sectors are `Basic Materials`, `Communication Services`, `Consumer Cyclical`, `Consumer Defensive`, `Energy`, `Financial Services`, `Healthcare`, `Industrials`, `Real Estate`, `Technology`, and `Utilities`. The ranges reject impossible or implausibly extreme API inputs; the fitted pipeline still applies its learned training-set clipping.

Extra or invalid fields return HTTP `422`. Unexpected inference failures return HTTP `500`.
A successful response is:

```json
{
  "prediction": 0.02147808903157589,
  "model_version": "hiba-hw4-rank-tree-v1"
}
```

## Local setup and checks

From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test
make lint
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The interactive documentation is at <http://127.0.0.1:8000/docs>.

The test result was **4 passed**, and Ruff reported **All checks passed** with all files formatted. On my MacBook Pro, 25 successful in-process `TestClient` requests after one warm-up request had a median latency of **6.26 ms** and p95 latency of **13.95 ms**. This measures Python request handling and inference in one process, not network or container latency, and will vary by machine.

## Docker and requests

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 \
  -p 127.0.0.1:8000:8000 \
  mfe-hw5:local
```

Stop the local Uvicorn server first if it is already using port `8000`.

In another terminal:

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The verified valid, invalid, and post-restart requests are saved in
[`evidence/container_requests.txt`](evidence/container_requests.txt).

## Before a real deployment

A real service would need authentication, HTTPS, request and model monitoring, rate limits, and managed logs. The upstream system must calculate the 13 features with exactly the same definitions and point-in-time rules as HW4. The model and data would also require drift and performance monitoring, a controlled update process, and secure artifact storage. Joblib artifacts must only be loaded from trusted sources. Finally, this ranking score is research
output, not investment advice or a guarantee of future returns.

## Attribution and AI use

The API structure follows the HW5 scaffold and the Lecture 9 FastAPI demo. `QuantileClipper` was adapted from `Lectures/modeling/course_data.py`. I used OpenAI Codex to help implement and review the service, tests, Docker configuration, and documentation; I ran the checks and verified the model and container responses locally.
