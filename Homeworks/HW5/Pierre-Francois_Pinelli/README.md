# HW5: Pierre-Francois Pinelli

For this assignment, I took the fitted model from my HW4, served it with FastAPI, added tests, and packaged it in a Docker image.

## Model and preprocessing

My HW4 model is an elastic net using 18 ranked stock characteristics. It predicts a stock’s 90-day return minus the average 90-day return of the stocks in the same month.

In HW4, I refit the model every quarter. For this API, I use the last refit from the HW4 test:

- Model version: `hw4-elastic-net-fit-2026-06-30`
- Fit date: 2026-06-30
- Parameters: `alpha = 0.0003`, `l1_ratio = 0.8`
- Training sample: 192,133 rows

All training rows have a 90-day outcome ending before the fit date. The latest outcome ends on 2026-06-01.

`export_model.py` runs the existing HW4 functions (`load_panel`, `add_ranks`, and `walk_forward`) and saves:

- `artifacts/model.joblib`: the fitted model, its version, and the ordered feature names. The file is about 2 KB and is included in Git.
- `artifacts/model_metadata.json`: the model settings, coefficients, and library versions.
- `artifacts/reference_prediction.json`: an input from HW4 and the prediction recorded by the HW4 code.
- `example_request.json`: the same input, ready to send to the API.

The only preprocessing in HW4 is the ranking. Each month, each characteristic is ranked across the eligible stocks, with values between -0.5 and 0.5. Missing values are replaced with 0.

Nothing is learned during this step: there is no fitted scaler, imputer, or custom transformer to save. The ranking is implemented in `add_ranks`, in my HW4 `hw4_model.py`.

Computing these ranks requires the other stocks in the same month. I therefore designed the API to take already computed ranks. Sending the raw characteristics of a single company would not reproduce the HW4 input.

To build a valid input, the client takes the eligible stocks of HW4 for a month-end date: in the course top-2000 list of the entry quarter, among the 1,000 largest by the dated rank of this list, with a price at the signal date, 63 complete daily returns, and no one-day price move above x5 or below /5 in the last 252 sessions. For each characteristic, it ranks these stocks from the lowest to the highest value (ties get the average rank), computes `(rank - 1) / (n - 1) - 0.5` where n is the number of stocks with a value, and sends 0 when the value is missing.

HW4 ran on Python 3.13 with scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, and scipy 1.16.3. For HW5, I use Python 3.12 with the same library versions in `requirements.txt` and in Docker. The artifact was exported in this environment. Its coefficients and example prediction match the full HW4 walk-forward exactly, with a difference of 0.

The service only loads the saved model. It never retrains and never queries the HW4 database. Since joblib can execute code when loading a file, the service uses only this trusted artifact.

## Input and output

`POST /predict` expects a JSON object with exactly the following 18 fields. This is also their order during training.

| # | Field | Characteristic being ranked |
| --- | --- | --- |
| 1 | `z_momentum_21` | 1-month return, over 21 sessions |
| 2 | `z_momentum_63` | 3-month return, over 63 sessions |
| 3 | `z_momentum_252_21` | 12-month return, excluding the last month |
| 4 | `z_volatility_63` | Volatility of daily returns over 63 sessions |
| 5 | `z_drawdown_252` | Price / highest price over the last year - 1 |
| 6 | `z_beta_252` | Beta against the S&P 500 total return index over 252 sessions |
| 7 | `z_log_dollar_volume_63` | Log of median dollar volume over 63 sessions |
| 8 | `z_net_margin` | Net income / revenue |
| 9 | `z_operating_roa` | Operating income / assets |
| 10 | `z_debt_to_assets` | Debt / assets |
| 11 | `z_cash_to_assets` | Cash / assets |
| 12 | `z_fcf_to_assets` | Free cash flow / assets |
| 13 | `z_accruals_to_assets` | (Net income - operating cash flow) / assets |
| 14 | `z_asset_growth` | Asset growth over one year |
| 15 | `z_gross_profitability` | Gross profit / assets |
| 16 | `z_book_to_market` | Book equity / market value |
| 17 | `z_earnings_yield` | Net income over the last four quarters / market value |
| 18 | `z_log_size_rank` | Log of the rank in the top-2000 list; a higher value means a smaller company |

All fields follow the same rules:

- The value must be a JSON number between -0.5 and 0.5, including both bounds.
- Ranks have no unit. -0.5 is the lowest rank of the month and 0.5 is the highest.
- Every field is required. If the underlying characteristic is missing, the client must send 0, as in HW4.
- Strings such as `"0.1"`, booleans, null, NaN, and infinity are rejected rather than converted.
- Missing or additional fields are rejected.

The range comes from the ranking formula:

```text
(rank - 1) / (n - 1) - 0.5
```

A value outside this range does not match the training input. For example, it could mean that raw data was sent instead of a rank. The linear model could still produce a number, so I reject the request before inference.

The order of the JSON keys does not matter. The service puts the values back into the training order before calling the model.

A successful response contains:

```json
{
  "prediction": 0.014235583262313726,
  "model_version": "hw4-elastic-net-fit-2026-06-30"
}
```

`prediction` is expressed as a fraction. A prediction of 0.014 means a 90-day return 1.4 percentage points above the average stock return in the universe. The horizon is unchanged from HW4: 90 calendar days from the close after the signal.

The API returns:

- **200** for valid input, with a finite prediction and the model version.
- **422** for invalid input, without calling the model. The response identifies the field and the reason without repeating the rejected value.
- **500** if inference fails or produces a non-finite prediction. The response is `{"detail": "Prediction unavailable"}`. Detailed exceptions are kept in the server logs.

`GET /health` returns the service status and the model version.

The example input is AAPL at the 2026-08-31 signal, the last signal of my HW4 test. Its HW4 score is `0.014235583262313726`. The API should return the same value within floating-point rounding.

In the Docker container on my Mac (arm64), it returns `0.01423558326231373`. Only the last digit differs.

`example_invalid_request.json` contains the same input with `z_beta_252` set to 0.9, which should be rejected.

## Setup and run

Use Python 3.12 and run these commands from the submission folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The interactive documentation is available at http://127.0.0.1:8000/docs.

From another terminal, send the example request:

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The service loads `artifacts/model.joblib` once at startup. Its path is resolved relative to `model_service.py`, so it does not depend on the terminal’s working directory.

The saved artifact is already included. To regenerate it, the course repository must contain my HW4 folder (`Homeworks/HW4/Pierre-Francois_Pinelli`) and the HW4 database:

```bash
python -m pip install duckdb==1.5.6
python export_model.py
```

The main files are:

- `app.py`: request validation, response models, and API endpoints.
- `model_service.py`: artifact loading and prediction logic, without FastAPI.
- `export_model.py`: export from the HW4 code.
- `tests/test_app.py`: API tests.
- `Dockerfile` and `requirements.txt`: completed versions of the template files.
- `Makefile`, `.dockerignore`, `.gitignore`, and `pyproject.toml`: unchanged template files.
- `container_demo.sh`: builds the image and records the demonstration.
- `evidence/`: saved test, lint, and container outputs.

## Tests and latency

Run:

```bash
make test
make lint
```

`make test` runs `python -m pytest -q`. Add `-s` to display the latency measurements. `make lint` runs both Ruff checks and the formatting check.

All 21 tests pass, and the lint passes. The saved outputs are in `evidence/test_output.txt` and `evidence/lint_output.txt`.

The tests cover four groups.

**Valid predictions.** The positive test uses the actual saved model. It checks HTTP 200, the response fields, and agreement with:

- A direct call to the saved model.
- The reference prediction recorded from `hw4_model.walk_forward`, before the API existed.
- A manual calculation using the intercept and coefficients in `model_metadata.json`.

The manual calculation is:

```text
intercept + sum(coefficient × rank)
```

The tolerance is `1e-9`. These are predictions from the same model, with scores around 0.01, so only small floating-point differences are expected.

Two additional tests check that setting all ranks to 0 gives the intercept and that changing the JSON key order does not change the prediction.

**Input validation.** Fifteen invalid requests must return 422: a missing field, an extra field, an empty object, a list of 18 numbers, a numeric string, a boolean, null, a list as a field value, values above and below the range, NaN, positive and negative infinity, a number too large for a float, and a non-JSON body.

During these tests, the loaded model’s `predict` method is replaced with a recorder. It must record no calls, which checks that invalid requests never reach inference.

**Inference failures.** With valid input, the model’s `predict` method is replaced with a function that raises an exception. The API must return 500, keep the exception details out of the response, and write them to the server logs.

Another test makes the model return NaN. This must also produce 500.

One issue found during testing was that a NaN input could produce 500 with FastAPI’s default validation response. Validation rejected the input, but the response included the rejected NaN value, which could not be serialized as JSON. I added a handler in `app.py` that returns 422 without repeating that value.

As an extra check, four deliberate changes were tried: removing this handler, returning a default prediction after an inference error, removing the range check, and removing the feature reordering. Each change caused at least one test to fail.

**Performance.** The latency test uses FastAPI’s in-process `TestClient`, without a network connection or a running HTTP server. It sends 10 warm-up requests, then measures 200 requests with `time.perf_counter`. Each measured request must return 200.

The saved measurements are:

- Linux x86_64 VM, 2 vCPU Intel Xeon, 7 GB RAM, Python 3.12.3: median **3.03 ms**, p95 **3.52 ms**. Output: `evidence/test_output.txt`.
- Inside the Docker image on my Mac, arm64, Docker 29.9.2: median **1.20 ms**, p95 **1.32 ms**. Output: the end of `evidence/container_transcript.txt`.

The test requires p95 below 100 ms. Inference is a dot product of 18 values, and the measured p95 is only a few milliseconds. This budget leaves room for slower machines while checking for large performance regressions. The model is loaded once at startup in the `lifespan` function of `app.py`; latency alone would not prove it.

## Docker

Build and run the image:

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

`make build` and `make run` run the same commands. The server listens on port 8000 inside the container. Send the example curl request from another terminal.

The image uses `python:3.12-slim` and includes the requirements, `app.py`, `model_service.py`, and `artifacts/`. It does not include the HW4 database or training code. `.dockerignore` excludes local environments, `.env` files, and databases from the build context.

To reproduce the demonstration:

```bash
bash container_demo.sh
```

This requires Docker and curl. It records the following in `evidence/container_transcript.txt`:

1. Build the image and start the container.
2. Send a valid request and show HTTP 200, the prediction, and the model version.
3. Send an invalid request and show HTTP 422.
4. Record the server logs.
5. Stop the container and start the same image again.
6. Repeat the valid request and show the same prediction.
7. Run the lint and tests inside the image.

If port 8000 is already in use:

```bash
PORT=8001 bash container_demo.sh
```

## Before a real deployment

The first point is the input preparation. The client currently computes the ranks, and the API cannot check whether they come from the correct universe or date. For a real service, I would move this step into a controlled data pipeline, using the raw data for the full stock universe.

The model also needs updating. HW4 refits it every quarter. A deployed service would need a process to export and version new models, switch between versions without downtime, and monitor inputs and predictions.

The API has no authentication, HTTPS, or rate limiting. The container runs as root with one process. Before deployment, I would review the process setup, use a non-root user, connect a health check to the platform, set resource limits, and add request IDs to the logs.

The artifact also depends on the library versions, and joblib files must come from a trusted source. Since this is a linear model, storing its coefficients in JSON would avoid loading an executable joblib artifact, provided the same feature contract and prediction logic are preserved.

Finally, the image includes pytest and Ruff for the assignment checks. A deployment image would only need the service dependencies.

## Sources and assistance

`app.py`, `tests/test_app.py`, `Dockerfile`, `Makefile`, `pyproject.toml`, and `requirements.txt` started from the HW5 template.

The inference failure handling and the check that invalid input never reaches the model were adapted from the Lesson 9 demo in `Lectures/Lecture 9/demo`.

The model, feature definitions, and export use my HW4 code.

I used Claude as an AI coding assistant: it wrote most of the code and the tests, and it ran the checks. The service builds on my HW4 model and feature definitions.
