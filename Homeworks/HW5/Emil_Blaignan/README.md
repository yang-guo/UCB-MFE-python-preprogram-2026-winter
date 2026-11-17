# HW5 submission, Emil Blaignan

I turned a fitted stock model from my HW4 work into a FastAPI service with one endpoint, `POST /predict`. It takes the 16 stock characteristics for one stock and week and returns the predicted 90 day return relative to the stock's sector. I tested it, put it in a Docker image and sent real requests to the running container.

## Which model I chose

My final HW4 strategy used a fixed four feature score, so nothing in it was fitted and there was no model to save. For this homework I chose the Fama-MacBeth linear model, which was one of the four alpha models I compared in HW4 (`benchrel.py`). It runs one regression per week on the 16 characteristics, averages the slopes, and shrinks each average by `t² / (t² + 4)` using Newey-West t statistics.

The saved pipeline has four steps.

1. `QuantileClipper` clips each feature to its 1st and 99th percentile from training.
2. `StandardScaler` standardizes with the training mean and standard deviation.
3. `ZClip` clips the standardized values to plus or minus 3.
4. `FamaMacBethRegressor` multiplies the features by the shrunken premia.

HW4 z scored features within each sector and date. That needs a whole week of stocks, which one request does not have. So the served pipeline uses the per row steps above, and I re-estimated the premia on those features. This is the main difference from HW4.

I trained on the 300 most traded stocks each week, using signal dates through 2015-12-31 whose 90 day outcome had already finished by that date. That gave 85,034 rows after I dropped 1,349 with missing features. The target is the 90 day return minus the sector median, clipped as in HW4. The model version string is `hw4-fm-linear-2015-12-31-v1`.

The premia are small. The largest t statistic in absolute value is 1.4 (see `premia.csv`), so none is significant. This matches HW4, where this model had almost no out of sample rank correlation. The homework does not grade investment performance.

## Input and output

The request is a JSON object with exactly these 16 fields, all required. Every field is a JSON number (a float, and a whole number such as 0 is also accepted). The order matches `FEATS` in my HW4 `helpers.py`, and the service builds a DataFrame with these column names in this order before predicting.

| Field | Meaning | Allowed range |
|---|---|---|
| `momentum_21` | return over the last 21 trading days, as a fraction | -1 to 5 |
| `momentum_63` | return over the last 63 trading days, as a fraction | -1 to 10 |
| `momentum_252_21` | return from 252 to 21 trading days ago, as a fraction | -1 to 100 |
| `volatility_63` | annualized volatility of daily returns over 63 days, as a fraction | 0 to 10 |
| `drawdown_252` | close divided by the 252 day high, minus 1 | -1 to 0 |
| `net_margin` | net income divided by revenue | -1000 to 1000 |
| `operating_roa` | operating income divided by total assets | -100 to 100 |
| `debt_to_assets` | total debt divided by total assets | 0 to 1000 |
| `cash_to_assets` | cash divided by total assets | -1 to 100 |
| `fcf_to_assets` | free cash flow divided by total assets | -10 to 10 |
| `accruals_to_assets` | accruals divided by total assets | -100 to 100 |
| `beta_mkt` | trailing beta to the market | -5 to 5 |
| `beta_mom` | trailing beta to the momentum factor | -10 to 10 |
| `resid_mom` | residual momentum, in standard deviations | -10 to 10 |
| `idisc` | path smoothness of the past return | -1 to 1 |
| `adv_log` | natural log of the 63 day median dollar volume in USD | 10 to 30 |

The bounds are wide on purpose. They catch typos and unit mistakes, such as dollar volume sent without the log, but they allow rare real values. The model also clips every feature at its training percentiles, so an extreme value cannot dominate the prediction. `feature_ranges.csv` shows the observed minimum, maximum and 0.1 and 99.9 percentiles of each feature in the training data, and I set each bound by comparing against it.

- **Limits that come from the definition.** A return cannot go below -1, so `momentum_21`, `momentum_63` and `momentum_252_21` start at -1. A drawdown must be between -1 and 0. Volatility and debt to assets cannot be negative.
- **Upper limits above the largest value I observed.** The training maximum was 1.6 for `momentum_21`, 6.9 for `momentum_63`, 50 for `momentum_252_21`, 5.8 for `volatility_63`, 9.2 for `fcf_to_assets`, 3.2 for `beta_mkt`, 6.1 for `beta_mom` and 4.2 for `resid_mom`, so each bound sits above the largest value. `net_margin` ranged from -636 to 75, so its bound of plus or minus 1000 covers it.
- **Ranges with a clear scale.** `adv_log` is the log of dollar volume. Training values were between 17.9 and 24, so a range of 10 to 30 covers dollar volumes from about 22 thousand to 10 trillion dollars. A value outside it means dollar volume was sent without the log. `idisc` is a smoothness measure that stayed within -0.3 to 0.2, so -1 to 1 is generous.
- **Ratios with tiny denominators.** `operating_roa`, `debt_to_assets` and `accruals_to_assets` have a few extreme rows in the data, such as debt to assets of 269,812. Their 0.1 to 99.9 percentile ranges are narrow (for example -0.17 to 0.50 for operating ROA), so I chose bounds far outside those ranges but did not try to cover the extreme rows. The service rejects those rows with a 422.

The response is `{"prediction": <float>, "model_version": "<string>"}`. The prediction is the expected 90 day return relative to the sector median, as a fraction.

| Situation | Status |
|---|---|
| Valid input | 200 |
| Missing field, extra field, wrong feature count, null, string, boolean, NaN, infinity or out of range value | 422 |
| Failure inside the model | 500 with the message "Prediction unavailable" |

Numeric strings and booleans are rejected on purpose, because they should not silently become features. The 422 body lists the field and the error type but not the submitted value. FastAPI's default body repeats the value, and a NaN cannot be written as JSON, so the default turned a NaN request into a 500. The 500 response hides the details and the full traceback goes to the server log.

## How to run it

Use Python 3.12. The versions in `requirements.txt` match the course environment that I trained in (scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3).

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make lint
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The interactive docs are at `http://127.0.0.1:8000/docs`. A valid request is saved in `example_request.json`.

```
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

## The saved model

`model.joblib` is small enough to commit. It holds the fitted pipeline, the model version and the feature list. The service loads it once when the server starts, using the lifespan function, and never trains or touches the HW4 database. It needs `fm_model.py` on the import path because the pipeline uses my custom classes. Joblib files are pickles, so only load files you made or trust.

`train_model.py` rebuilds the file. It needs the generated HW4 outputs and the course DuckDB file, so I run it from the repo root with the course environment.

```
.venv/bin/python Homeworks/HW5/Emil_Blaignan/train_model.py
```

It also writes `example_request.json`, `reference.json`, `premia.csv` and `feature_ranges.csv`. `reference.json` holds one real HW4 row and the prediction I got by calling the fitted pipeline directly, before the API existed. The tests compare the API against that number.

## Docker

The Dockerfile uses `python:3.12-slim`, sets `/app` as the working directory, installs `requirements.txt`, and copies `app.py`, `fm_model.py` and `model.joblib`. The `.dockerignore` keeps `.venv`, `.env` files and database files out of the build.

```
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

My Mac runs macOS 13, which the current Docker Desktop does not support. I installed Colima and the Docker command line with Homebrew, then ran `colima start` before building. Anyone with Docker Desktop can use the same two commands above.

## Results

| Check | Result |
|---|---|
| `make test` | 13 tests pass |
| `make lint` | `ruff check` and `ruff format --check` pass |
| Latency | median 1.04 to 1.06 ms and p95 1.12 to 1.20 ms across my runs |
| Container | valid request returns 200, invalid requests return 422, and a new container from the same image returns the same prediction |

The latency test warms up with 3 requests and then times 20 successful requests with `time.perf_counter`. It uses the in process `TestClient`, so there is no network and no real HTTP. I ran it on an Apple M2 with macOS 13.3 and Python 3.12. I did not set a time limit in the test because the numbers depend on the machine.

The tests cover the four required groups.

- **Valid prediction.** The response has the right fields and a finite number, and it matches both a direct pipeline call and the reference value from `reference.json`.
- **Invalid input.** Ten cases return 422. They are a missing field, an extra field, a string, a boolean, null, NaN, infinity, an out of range value, a short object and a list. In each one I replace `predict` with a function that records calls and check that it was never called.
- **Inference failure.** I replace `predict` with one that raises and send valid input. The response is 500 and does not contain the error text or a prediction.
- **Performance.** The latency measurement described above.

`test_and_lint_output.txt` is the saved output of `make lint` and `python -m pytest -v -s`, including the latency line from that run. `container_transcript.txt` is a terminal transcript from the running container. It shows the build, a valid request with the 200 response, prediction and model version, a rejected request with an extra field, a rejected request with NaN, then a stop and restart of the same image followed by the same request again.

## Before a real deployment

- The service has no authentication, rate limiting or HTTPS.
- A few real rows in my HW4 data have extreme ratios from tiny denominators, such as debt to assets above 1000. The service would reject them with a 422, so a real system needs a rule for those.
- Callers have to compute the 16 features exactly as in HW4 and without look ahead. The service cannot check that.
- There is no monitoring for input drift or prediction drift, and no retraining schedule. The model is trained only through 2015.
- Loading a pickle is unsafe for an untrusted file. A real system should use a checksummed artifact store or a format that is not a pickle.
- The image should pin the base image, run as a non root user, leave out the test and lint tools, and have a health endpoint.

## Help and sources

I used an AI coding assistant (Claude Code) in three stages.

1. At the start, it helped me work out which HW4 model to choose and wrote the model code (`fm_model.py` and `train_model.py`). It also saved the model and set up the environment. The Fama-MacBeth steps are adapted from my HW4 `benchrel.py`.
2. I wrote my own code for the request and response models, the model loading and the prediction in `app.py`, and my own first version of the tests. I based the `Field` bounds, `ConfigDict` settings and lifespan loading on the Lecture 9 demo in `Lectures/Lecture 9/demo/app.py`. Claude Code then reviewed my code. It added the handler that returns 422 for NaN and infinity, a `try` and `except` that turns model failures into a 500, and the extra invalid input cases and checks in the tests.
3. For the Docker part, it wrote the Dockerfile, helped me install Docker through Colima, ran the build and the container requests, saved the transcript, and helped draft this README.

## What is in this folder

| File | What it is |
|---|---|
| `app.py` | The FastAPI service |
| `fm_model.py` | Custom scikit-learn classes used by the saved model |
| `train_model.py` | Rebuilds the model and the reference files from the HW4 data |
| `model.joblib` | The fitted pipeline with its version and feature list |
| `example_request.json` | A valid request from a real HW4 row |
| `reference.json` | The same row with the prediction computed before the API existed |
| `premia.csv` and `feature_ranges.csv` | The fitted premia and the observed range of each feature |
| `tests/test_app.py` | The tests |
| `Dockerfile`, `.dockerignore` | The container build |
| `Makefile`, `pyproject.toml`, `requirements.txt` | Commands, tool settings and pinned versions |
| `test_and_lint_output.txt` | Saved output of the lint and test commands |
| `container_transcript.txt` | Terminal transcript from the running container |
