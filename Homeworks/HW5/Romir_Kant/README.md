# HW 5: Romir Kant

A FastAPI service that serves my HW4 stock-ranking model from a Docker container. `POST /predict` takes one stock's 20 ranked features and returns the model's prediction and the model version.

## Files

- `app.py`: the API (input and output models, startup load, `/predict`)
- `model_service.py`: loads the saved model and predicts
- `artifacts/`: the saved HW4 model, its details, and three reference predictions
- `export_model.py`: created the files in `artifacts/` from the HW4 code (not needed to run the service)
- `example_request.json`: a valid request body
- `tests/test_app.py`: the four test groups
- `evidence/`: test and lint output, and the requests made to the container

## The model

The service serves the last refit from my HW4 final test (PR #276, commit 93c2441), version `hw4-ridge-rank-fit-2026-03-27`:

- Ridge regression, alpha = 1000, fit on 2026-03-27 on 1,580,288 stock-weeks whose outcomes had already ended.
- It predicts the stock's percentile rank, among that week's stocks, of its return from the next session's close to the first close at least 90 days later.

`export_model.py` repeats that refit by calling the HW4 function `research.fit_predict` on the same rows, then saves the fitted model with joblib. It also saves three real HW4 inputs with the predictions that HW4 function gave them (AAPL, MSFT, and ACHR, which had a missing feature). Those are the reference results the tests use. They were recorded before any API code touched the model.

Preprocessing: HW4 turns each feature into its percentile rank among that week's stocks, minus 0.5, and sets a missing feature to 0. Nothing in that step is learned from the training data, and one request describes one stock, so the service cannot rank it against the others. The caller sends the ranked values. This is the main limitation of the service.

Versions: HW4 trained with scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3 and joblib 1.5.3 on Python 3.14. The template's `requirements.txt` already pinned those library versions, so I did not change it. The service and the image run them on Python 3.12, and the service refuses to start if the scikit-learn version differs from the one saved with the model.

## Input and output

The request body is a JSON object with exactly these 20 fields. The order in the table is the training order; the service puts the fields in this order itself, so the order in the request does not matter.

Every field is a required number from -0.5 to 0.5: the feature's percentile rank among that week's eligible stocks, minus 0.5. A missing feature is sent as 0, the middle rank. Ranks cannot fall outside this range in training, so anything outside it is rejected.

| # | Field | Feature that is ranked |
|---:|---|---|
| 1 | `mom_1m` | Return over the last 21 sessions |
| 2 | `mom_3m` | Return over the last 63 sessions |
| 3 | `mom_6m_skip1m` | Return from 126 to 21 sessions ago |
| 4 | `mom_12m_skip1m` | Return from 252 to 21 sessions ago |
| 5 | `vol_3m` | Annualized volatility of daily returns, 63 sessions |
| 6 | `vol_12m` | Annualized volatility of daily returns, 252 sessions |
| 7 | `drawdown_12m` | Price relative to its 252-session high, minus 1 |
| 8 | `max_ret_1m` | Largest one-day return in the last 21 sessions |
| 9 | `beta_12m` | Beta to the S&P 500 total return, 252 sessions |
| 10 | `idio_vol_12m` | Annualized volatility not explained by beta, 252 sessions |
| 11 | `log_dollar_volume` | Log of average dollars traded a day, 63 sessions |
| 12 | `roa` | Trailing-12-month net income / total assets |
| 13 | `accruals` | (Net income - operating cash flow, 12 months) / total assets |
| 14 | `earnings_yield` | Trailing-12-month net income / market value |
| 15 | `sales_yield` | Trailing-12-month revenue / market value |
| 16 | `fcf_yield` | Trailing-12-month free cash flow / market value |
| 17 | `debt_to_assets` | Total debt / total assets |
| 18 | `cash_to_assets` | Cash and equivalents / total assets |
| 19 | `asset_growth` | Total assets vs. the same quarter a year earlier, minus 1 |
| 20 | `sales_growth` | Quarterly revenue vs. the same quarter a year earlier, minus 1 |

Responses:

| Status | When | Body |
|---|---|---|
| 200 | Valid input | `prediction` (a finite number) and `model_version` |
| 422 | Missing or extra field, a string, boolean or null, NaN or infinity, a value out of range, or a list in place of the named fields | The field and the reason |
| 500 | The model fails | `{"detail": "Prediction failed"}`. The full error is only in the server log |

`prediction` is the predicted percentile rank, minus 0.5, of the stock's 90-day return among that week's stocks. Higher is better and 0 is the middle. It is a ranking score, not a return.

Example, using `example_request.json` (AAPL, signal date 2026-03-27):

```json
{"prediction": 0.043031901081589566, "model_version": "hw4-ridge-rank-fit-2026-03-27"}
```

## Setup and run

From this folder, with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The API documentation is then at http://127.0.0.1:8000/docs. The model loads once at startup from `artifacts/model.joblib`, which is in this folder (1.5 KB). The service does not retrain and does not use the HW4 database.

Checks (`make test` and `make lint` run the same commands):

```sh
python -m pytest -q -s             # -s prints the latency report
python -m ruff check .
python -m ruff format --check .
```

Recreating the model file is optional. It needs the HW4 code from PR #276 and the panel that HW4's `validate.py` caches, and it runs in the HW4 environment: `python export_model.py --hw4-dir ../../HW4/Romir_Kant`.

## Tests

All four tests pass, and both ruff checks pass. The output is in `evidence/test_and_lint_output.txt`.

1. Valid prediction. The API returns 200 with the right fields. Its prediction matches a direct call to the saved model, and matches the three predictions recorded from HW4. The tolerance is 1e-12: the model is a sum of 20 small terms, so a different Python build can only change the last few bits.
2. Input validation. 15 invalid requests each get 422, and the model's `predict` is never called.
3. Inference failure. With `predict` replaced by one that raises, a valid request gets 500 and the error text is not in the response.
4. Performance. After 10 warm-up requests, 200 requests took a median of 8.5 ms and a p95 of 11.6 ms. This is an in-process `TestClient` measurement with `time.perf_counter`, so it has no network and no separate server. Hardware: Intel Core i7-11800H, 16 GB RAM, Windows 11, Python 3.12.9. The test fails if the median is above 50 ms, which leaves room for a slower machine but would catch reloading the model on every request.

While writing the tests I found that a request containing NaN returned 500, because FastAPI's default 422 response echoes the bad value and NaN cannot be written as JSON. `app.py` now returns the 422 without the value.

## Container

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

Then, from another terminal in this folder:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The image uses `python:3.12-slim`, installs `requirements.txt`, and copies only `app.py`, `model_service.py` and `artifacts/`. Stop it with `docker stop mfe-hw5`.

`evidence/container_requests.txt` is the transcript of my run:

- The valid request returned 200 with prediction 0.043031901081589566 and the model version. This is the same number HW4 gave.
- An invalid request (`mom_1m` sent as the string "0.1") returned 422.
- After stopping the container and starting the same image again, the valid request returned 200 with the same prediction.

## Before a real deployment

- Ranking. Callers must rank features against the whole week's stocks themselves. A real service should take the week's raw features for all stocks and rank them itself, so callers cannot rank differently from training.
- Access. There is no authentication, HTTPS or rate limit. The container also runs as root.
- Image size. The image includes pytest and ruff because they are in `requirements.txt`. A deployed image should install only the serving packages.
- Monitoring. Add a health check, request IDs in the logs, and tracking of latency, error rates and how the inputs drift.
- Model updates. HW4 refits every March, so the service needs a way to publish a new model version and roll back.
- Model file. joblib files can run code when loaded, so the file must come from a trusted, checked source.
- The model itself did not beat the S&P 500 in HW4. Serving it does not change that.

## Code reuse and AI help

`app.py` and the Dockerfile follow the pattern of the Lesson 9 demo (`Lectures/Lecture 9/demo/`): the lifespan load, the strict Pydantic input, and the logged 500. The tests fill in the template's four test functions. The model code is from my HW4. I used Claude to help write and debug the code, find the NaN bug, run the container checks, and draft this README. I reviewed the design, the tests and the results.
