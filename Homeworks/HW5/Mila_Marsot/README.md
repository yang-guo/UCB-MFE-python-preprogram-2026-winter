# HW5: Mila Marsot

I turn my HW4 model into an API (`POST /predict`) and run it in Docker.

## 1. The model

- **Model:** the Ridge model from my HW4 notebook (`test_model`): `course_data.make_pipeline(Ridge(alpha=100))`.
- **Version:** `hw4-ridge-alpha100-trained-before-2016-03-31`
- **Prediction:** the stock return over the next 90 days (0.05 = +5%).

The prediction for NVDA (3.75) is not realistic because of some wrong prices in the database (see HW4). Model performance is not graded, so I keep the HW4 model as it is.

## 2. Input and output

The API receives the 13 HW4 features of one stock, with the same names and order as in HW4.

| # | Field | Null allowed? | Range | Meaning |
|---|---|---|---|---|
| 1 | `momentum_21` | no | −1 to 100 | return over 21 days |
| 2 | `momentum_63` | no | −1 to 100 | return over 63 days |
| 3 | `momentum_252_21` | yes | −1 to 100 | return from 252 to 21 days ago |
| 4 | `volatility_63` | no | 0 to 50 | volatility over 63 days, per year |
| 5 | `drawdown_252` | yes | −1 to 0 | price / 252-day high − 1 |
| 6 | `net_margin` | yes | ±1,000,000 | net income / revenue |
| 7 | `operating_roa` | yes | ±1,000,000 | operating income / assets |
| 8 | `debt_to_assets` | yes | 0 to 1,000,000 | debt / assets |
| 9 | `cash_to_assets` | yes | ±1,000,000 | cash / assets |
| 10 | `fcf_to_assets` | yes | ±1,000,000 | free cash flow / assets |
| 11 | `accruals_to_assets` | yes | ±1,000,000 | accruals / assets |
| 12 | `filing_age_days` | yes | 0 to 3,650 | days since the last report |
| 13 | `sector` | no | a training sector or `"Unknown"` | company sector |

All fields are numbers, except `sector` (text).

Units: returns, drawdown and ratios are fractions (0.05 = 5%). Volatility is per year. `filing_age_days` is in days.

**Why these rules:**
- All 13 fields are required and extra fields are refused. So 12 or 14 features always give a 422.
- `null` means "unknown", like in HW4. The model replaces it with the training median.
- A number sent as text (`"0.05"`), true/false, NaN and Infinity are refused.
- The ranges refuse impossible values (a price cannot fall more than 100%, a volatility cannot be negative). I chose the upper limits by looking at the training data (for example, some ratios go up to about 950,000).

**Answers:**
- **200**: `{"prediction": 3.752976926401481, "model_version": "hw4-ridge-alpha100-trained-before-2016-03-31"}`
- **422**: invalid input. The model is not called.
- **500**: the model fails. The full error is only in the server log.

Example input: `example_request.json` (NVDA on 2016-03-31, a real HW4 row).

## 3. Setup and run

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/docs to try the API. The library versions are the same as in HW4 (scikit-learn 1.8.0).

## 4. The saved model

- `artifacts/hw4_ridge.joblib` (5.5 KB) is loaded once when the server starts. The API never trains the model and never reads the database.
- The server stops if the scikit-learn version is not 1.8.0.
- I made the file with `export_hw4_model.py`, which repeats my HW4 steps. It needs the course database and environment.
- The model uses the course class `QuantileClipper`. I copied it into `hw4_model.py`, which is also in the Docker image.

## 5. Tests and latency

```sh
make test
make lint
```

19 tests pass, and ruff finds no problems (`evidence/test_and_lint_output.txt`).

1. **Valid input:** the answer is 200. The prediction is the same as a direct call to the model (tolerance 1e-12). It is also the same as a reference saved from my HW4 code before the API existed (`artifacts/reference_predictions.json`, tolerance 1e-10). I use a tolerance because the last digit can change between computers.
2. **Invalid input:** 14 cases give 422, and the test shows the model is never called.
3. **Model failure:** the model's `predict` is replaced by a function that raises an error. The answer is 500.
4. **Latency:** measured with the in-process `TestClient` (no real network): 10 warm-up requests, then 100 timed requests with `time.perf_counter`. Computer: Apple M3 Pro, 18 GB. Result: median **2.08 ms**, p95 **2.70 ms**. The test checks p95 < 100 ms, so it still passes on a slower computer.

## 6. Docker

With Docker Desktop open:

```sh
make build
make run
```

The image uses `python:3.12-slim` and copies `app.py`, `hw4_model.py` and `artifacts/`. `.dockerignore` keeps `.venv`, databases and passwords out of the image.

## 7. Requests to the container

```sh
./demo_container.sh 2>&1 | tee evidence/container_transcript.txt
```

The script sends a valid request (200) and an invalid one (422). Then it restarts the container and sends the valid request again (200, same prediction). The output is in `evidence/container_transcript.txt`.

## 8. Before a real deployment

- Retrain the model with clean prices, and give each new model a new version.
- Compute the features in the service, not by the user, to avoid input mistakes.
- Add authentication, HTTPS, logs, alerts and CI tests.

## Sources and use of AI

- The structure comes from the Lesson 9 demo. `QuantileClipper` is copied from `course_data.py`.
- I used AI to help me write the code. It found one bug: NaN in the input gave a 500 instead of a 422. I fixed it in `app.py`. I ran the tests and the container myself.