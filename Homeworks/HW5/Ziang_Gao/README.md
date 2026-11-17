# HW5: Serve and containerize my HW4 return model

Author: Ziang Gao

This service gives the prediction of my HW4 model through a FastAPI endpoint, `POST /predict`. It runs in a Docker container. It loads a saved model. It does not train, and it does not read the HW4 database.

## Model that is served

| Item | Value |
|---|---|
| Model version | `hw4-round1-ridge-fit-2016-03-31` |
| Source | HW4 round 1, `hw4.ipynb` section 7. Round 2 uses the same return model. |
| Model | Ridge regression (`alpha=100`) on feature set C (momentum and financial features) |
| Preprocessing (learned, saved with the model) | Clip each feature at its 1% and 99% training quantiles, impute the median, standardize |
| Training data | 573,544 signals up to 2015-12-24 whose 90-day outcome ended before the fit date, 2016-03-31 |
| Prediction | Return from the next session's close to the first close on or after 90 calendar days, as a fraction (0.05 = 5%) |

HW4 did not save the fitted model, so `export_model.py` fits it again with the HW4 code and saves it. My HW4 notebook kernel had a different pandas version, and its cache file cannot be read in the course environment. Thus the script builds the data again from the course database with the HW4 function `build_panel`, in the locked course environment (`uv sync --locked`). The result has the same 573,544 training rows and the same Ridge coefficients as `hw4.ipynb`. The versions in `requirements.txt` are the versions of that environment (scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3). `inference.py` refuses to load the model if the installed scikit-learn version is different.

The pipeline uses a custom transformer, `QuantileClipper`, from the lecture file `course_data.py`. That file also imports duckdb and the data loaders. I copied the transformer into `preprocessing.py`, so the service and the container need only that small module. `export_model.py` fits the pipeline two times, one time with the lecture function that `hw4.ipynb` calls and one time with the copy, and it stops if the predictions are not equal (the largest difference was 0.0).

## Input contract

`POST /predict` takes one JSON object with exactly these 12 fields. All are required JSON numbers. Returns and ratios are fractions, not percent. The order below is the training order. The service builds a one-row DataFrame with these column names, because the pipeline selects columns by name.

| # | Field | Unit | Allowed range | Meaning | Reason for the range |
|---|---|---|---|---|---|
| 1 | `momentum_21` | fraction | -1 to 5 | Return over the last 21 sessions | A price cannot fall more than 100%. Training maximum: 3.5. |
| 2 | `momentum_63` | fraction | -0.8 to 5 | Return over the last 63 sessions | The HW4 screen rule. The model never saw a value outside it. |
| 3 | `momentum_252_21` | fraction | -1 to 100 | Return from 252 to 21 sessions ago | 99.9% of training rows are below 8.4. Larger values in the data were price errors. |
| 4 | `mom63_vs_sector` | fraction | -6 to 6 | `momentum_63` minus the sector median on the same date | Difference of two values inside the limits above. |
| 5 | `mom21_vs_sector` | fraction | -6 to 6 | `momentum_21` minus the sector median on the same date | Same reason. |
| 6 | `net_margin` | fraction | -1,000,000 to 1,000,000 | Net income / revenue, latest filing | See the note below. |
| 7 | `operating_roa` | fraction | -1,000,000 to 1,000,000 | Operating income / total assets | See the note below. |
| 8 | `debt_to_assets` | fraction | 0 to 1,000,000 | Total debt / total assets | Debt is not negative. |
| 9 | `cash_to_assets` | fraction | -1,000,000 to 1,000,000 | Cash / total assets | See the note below. |
| 10 | `fcf_to_assets` | fraction | -1,000,000 to 1,000,000 | Free cash flow / total assets | See the note below. |
| 11 | `accruals_to_assets` | fraction | -1,000,000 to 1,000,000 | Accruals / total assets | See the note below. |
| 12 | `filing_age_days` | days | 0 to 1000 | Days since the latest filing that was public on the signal date | Training range: 0 to 615. |

Note on the ratio limits. Accounting ratios have very long tails when revenue or assets are near zero (training values reach about 950,000). The saved pipeline clips each feature at its 1% and 99% training quantiles, so a large value cannot move the prediction much. The limit only refuses values that cannot be a real ratio. The full training range of each feature is in `artifacts/model_metadata.json`.

The service returns **422** and does not call the model for: a missing field, an extra field (so 11 or 13 features are refused), a list in place of named fields, a numeric string, a boolean, `null`, NaN, infinity, and a value outside its range. Whole JSON numbers such as `63` are accepted. There are no categorical fields. The model does not use the sector.

One limit of this contract: in training, a missing financial ratio was imputed. The API requires all 12 values, so a stock with a missing ratio cannot be scored.

## Output contract

| Status | Body | When |
|---|---|---|
| 200 | `{"prediction": <finite float>, "model_version": "hw4-round1-ridge-fit-2016-03-31"}` | Valid input |
| 422 | `{"detail": [{"type", "loc", "msg"}, ...]}` | Invalid input. The input value is not copied into the response. |
| 500 | `{"detail": "Prediction unavailable"}` | Unexpected inference failure, or a prediction that is not finite. The full error is only in the server log. |

`GET /health` returns `{"status": "ready", "model_version": ...}` when the model is loaded.

Example input: `example_request.json` is AAPL on 2016-03-31, the first signal date of the HW4 test period. The HW4 code predicts 0.0444 for it (a 90-day return of 4.44%).

## Files

| File | Job |
|---|---|
| `app.py` | Request handling: input and output models, model loading at startup, `/predict`, `/health` |
| `inference.py` | Model logic without web code: load the artifact, predict one row |
| `preprocessing.py` | Feature list, model version, `QuantileClipper`, and the pipeline (copied from the lecture file, cited) |
| `export_model.py` | Fits the HW4 model again and saves the artifact, the metadata, and the example request |
| `artifacts/model.joblib` | Fitted pipeline, model version, feature list (about 4 KB, in Git) |
| `artifacts/model_metadata.json` | Library versions, training facts, feature ranges, and the reference input and prediction |
| `example_request.json`, `invalid_request.json` | A valid and an invalid request for the demonstration |
| `tests/test_app.py` | 25 tests |
| `demo_container.sh` | Runs the Milestone 5 demonstration and prints the transcript |
| `Dockerfile`, `Makefile`, `requirements.txt`, `pyproject.toml` | Container, commands, dependency versions, tool settings |
| `latency_report.txt`, `evidence/` | Outputs of my runs |

## Setup and commands

Use Python 3.12. From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test      # python -m pytest -q
make lint      # ruff check and ruff format --check
python -m uvicorn app:app --host 127.0.0.1 --port 8000    # local server, docs at /docs
```

The artifact is in Git, so there is nothing to download. The service loads `artifacts/model.joblib` one time at startup (the `lifespan` function), with a path relative to `inference.py`. Load only an artifact that you made or trust, because joblib can run code when it loads a file.

To make the artifact again (optional), go to the repository root, put `equities_top2000.duckdb` in `Lectures/modeling/data/`, and run:

```sh
uv sync --locked
uv run python Homeworks/HW5/Ziang_Gao/export_model.py
```

## Tests

`tests/test_app.py` has the four groups of the assignment.

1. **Valid prediction.** Uses the real saved model. Checks status 200 and the schema. Checks that the API result equals a direct call of the saved pipeline. Checks that it equals the reference prediction. The reference input and prediction are in `artifacts/model_metadata.json`. `export_model.py` recorded them from the HW4 code path (`course_data.make_pipeline`, the call in `hw4.ipynb`) before the API existed. The tolerance is 1e-9, because the last digits of a float can change between CPUs, and a real model error is much larger.
2. **Input validation.** 15 invalid bodies and 4 non-finite values (NaN, Infinity, -Infinity, 1e999). Each must return 422. A recorder replaces `predict` on the loaded model, and each test checks that the recorder was not called.
3. **Inference failure.** Valid input, and `predict` is replaced by a function that raises. The response must be 500 and must not contain the internal error text. A second test makes the model return NaN, which must also be 500.
4. **Performance.** See below.

Outputs of my run: `evidence/test_output.txt` and `evidence/lint_output.txt`.

One thing I found with the tests: the default FastAPI 422 response copies the input value, and a NaN input cannot be written as JSON. The server then answered 500 for a NaN input. `app.py` has its own handler for validation errors that does not copy the input.

## Latency

The performance test sends 10 warm-up requests and then 200 requests, and times each one with `time.perf_counter`. It is an **in-process `TestClient` measurement**. There is no network and no real HTTP socket, so a request to the container takes longer. The test writes the median, the p95, the hardware, and the method to `latency_report.txt`. My run on an Apple M4 MacBook Pro (10 logical CPUs, Python 3.12.9): median 1.16 ms, p95 3.07 ms.

The test asserts p95 < 250 ms. This budget is for my laptop only. One Ridge prediction on one row takes a few milliseconds, so 250 ms fails only for a real regression, for example if the service loaded the model again for each request.

## Container

```sh
make build     # docker build -t mfe-hw5:local .
make run       # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The image is `python:3.12-slim` with the packages of `requirements.txt`, `app.py`, `inference.py`, `preprocessing.py`, and `artifacts/`. The process listens on port 8000 in the container. `.dockerignore` keeps `.venv`, caches, databases, and `.env` files out of the build. To reproduce: clone the repository, go to this folder, start Docker, and run the two commands. If host port 8000 is in use, stop that program first or change the first `8000` in the `run` command.

With the container running, send a request:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

### Evidence of requests to the container

`demo_container.sh` starts the image, sends the valid request and an invalid request, stops the container, starts the same image again, and sends the valid request again. The transcript of my run is `evidence/container_transcript.txt`. It shows the 200 response with the prediction and the model version, the 422 response, and the 200 response after the restart. The model is a file in the image, so it is available after each restart.

```sh
make build
bash demo_container.sh 2>&1 | tee evidence/container_transcript.txt
```

## What needs attention before a real deployment

- **The model.** The signal is weak (mean rank IC 0.015 in HW4) and the strategy did not beat the S&P 500. The fit is from March 2016. A real service needs a retraining schedule and monitoring of the input distribution and of the prediction quality.
- **Feature computation.** The caller computes the 12 features. If the caller computes them differently from `hw4_features.py` (other windows, percent in place of fractions, a filing that was not yet public), the prediction is wrong and the service cannot see it. The feature code must be shared or served.
- **Missing values.** Training imputed missing ratios. The API refuses them. A real contract must decide this case.
- **Security and limits.** There is no authentication, no TLS, no rate limit, and no limit on the request size. The container runs as root.
- **Operations.** No request ID, no metrics, one worker, and no Docker health check. The latency number is in-process and not a load test with real HTTP.
- **Dependencies.** The image also installs pytest and ruff. A real image needs a separate list of runtime packages with hashes.
- **Batch requests.** The endpoint scores one stock for each request. The HW4 strategy scores about 2,000 stocks on each signal date.
- **CI.** I did not do the optional GitHub Actions extension. The checks ran on my laptop only.

## Credits and AI assistance

`QuantileClipper` and the pipeline in `preprocessing.py` are copied from the lecture file `Lectures/modeling/course_data.py`. `app.py` and the tests adapt the HW5 template and the Lecture 9 demo (`Lectures/Lecture 9/demo/`): the lifespan loading, the strict input model, the 500 handling, and the idea to replace `predict` in the tests.

Claude, an AI assistant from Anthropic, helped write the code and this README, and it found the cause of the cache error and the NaN problem. I ran the export, the tests, the lint checks, the container, and the requests on my laptop, and I checked the outputs against `hw4.ipynb`.
