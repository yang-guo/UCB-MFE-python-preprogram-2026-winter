# HW5: Gobind Singh

Serves the HW4 Ridge pipeline `ridge-alpha100-top2000-2010-2015`. The artifact is `ridge_90d_full.joblib`, fit on development signals from 2010 through 2015. The service loads that file once and does not retrain or open the HW4 database. `course_data.py` contains `QuantileClipper`, the custom transformer the pickle imports.

The example row is symbol A on 2016-03-01 from the HW4 score panel. A direct `pipeline.predict` call on that row returns about `-0.20290557`. `example_request.json` is that same row.

## Contract

`POST /predict` takes the HW4 features in this order. Momentum, volatility, drawdown, and the statement ratios are decimals. `filing_age_days` is a nonnegative number of days, at most 550. `sector` is a non-empty string. Extra fields, strings or booleans in numeric fields, and non-finite numbers return **422**. A valid request returns **200** with a finite `prediction` and `model_version`. An unexpected failure inside `predict` returns **500**.

| Field | Meaning | Accepted range |
| --- | --- | --- |
| momentum_21, momentum_63, momentum_252_21 | Trailing adjusted-price returns | -1 to 5, 10, and 20 |
| volatility_63 | Annualized 63-day volatility | 0 to 5 |
| drawdown_252 | Price over trailing high, minus 1 | -1 to 0 |
| net_margin, operating_roa, fcf_to_assets, accruals_to_assets | Filing-dated ratios | -5 to 5 |
| debt_to_assets | Filing-dated ratio | 0 to 20 |
| cash_to_assets | Filing-dated ratio | 0 to 5 |
| filing_age_days | Days since the filing became usable | 0 to 550 |
| sector | Snapshot sector label | 1 to 64 characters |

## Run

From this folder, with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

`requirements.txt` matches the HW4 training stack: numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.16.3, joblib 1.5.3.

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

## Checks

`pytest` uses the real saved model for the valid request and compares it with a direct pipeline call and the reference value above. Invalid payloads are rejected with 422 before `predict` runs. The failure test replaces `predict` with a function that raises and expects 500.

Latency was measured in-process with FastAPI's `TestClient` on this machine: 20 requests after one warmup, timed with `time.perf_counter`. Median about 0.013 seconds, p95 about 0.014 seconds. The test allows a p95 under 0.5 seconds for that in-process measurement.

Container evidence is in `success.png` (200 from the running image), `invalid.png` (rejected request), and `restart.png` (the same image started again, then the same valid request).

## Before a real deployment

Pin the image digest, do not ship the model inside an unauthenticated public endpoint, and add a request limit. The feature bounds are wide enough for the training ranges; they are not a trading control. A prediction is a 90-day return score, not an order.
