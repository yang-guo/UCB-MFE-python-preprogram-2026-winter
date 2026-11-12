# HW5 - Serve and containerize the HW4 model

This API serves my final HW4 `model_full`: a fitted StandardScaler followed by Ridge regression. It predicts a stock's approximately 90-calendar-day return from two price features. The service loads the saved pipeline at startup; it does not retrain or query the HW4 databases.

## Model and input contract

Model version: `hw4-model-full-2016-04-01`. The model uses development signal dates from 2010 through 2015, with outcomes known before April 1, 2016. Ridge alpha is 100000. Both StandardScaler and Ridge were fitted with exponentially decreasing sample weights, with a half-life of 180 calendar days and average weight normalized to one.

`POST /predict` expects a JSON object containing exactly these two fields. Inference always builds a DataFrame in this order:

| Feature | Meaning and units | Accepted range |
| --- | --- | --- |
| `momentum_63` | Adjusted-price return over 63 trading sessions, as a decimal | Finite number greater than -1 and at most 10 |
| `volatility_63` | Sample standard deviation of 63 daily adjusted-price returns, multiplied by sqrt(252), as a decimal | Finite number from 0 through 10 |

The momentum lower bound follows from positive adjusted prices; its upper cap of 10 allows a +1,000% return over 63 sessions while rejecting more extreme extrapolation. Volatility must be nonnegative; its upper cap of 10 allows 1,000% annualized volatility while rejecting extreme inputs such as 1e10. These broad service limits are explicit plausibility guards, not bounds estimated from the training sample. Numeric strings, booleans, nulls, missing fields, extra fields, and non-finite values are rejected with HTTP 422. Named fields enforce the required feature count.

Valid input returns HTTP 200 with a finite numeric `prediction` and a string `model_version`. Prediction is a decimal return: -0.017168 means about -1.72%. Unexpected inference failures return HTTP 500; exception details stay in the server logs. Validation responses omit the invalid input values so non-finite values cannot break JSON error serialization.

## Saved artifacts

`model.joblib` contains both fitted preprocessing and regression. `hw4_reference.json` records one input from the first row of HW4's `predictions_full`, its direct `model_full.predict` result, and the model version. `example_request.json` contains that input alone. The reference prediction is -0.017168045543674057. The valid-input test cross-checks the API against both the direct pipeline call and this saved HW4 result, confirming the same model prediction. HW4's own output in [PR #214](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/pull/214) also shows this value: the first row of `predictions_full` (stock A, 2016-03-31) has a prediction of -0.017168. Only load trusted Joblib artifacts.

The export was reproduced from the original HW4 full-data model cells in `prepare_hw4_model.py`, using the alpha and half-life recorded in HW4's output. This preparation helper requires the original course environment, `Lectures/modeling` helpers, and HW4 database; the API and Docker image require none of those. `inspect_hw4.py` loads the saved pipeline and prints the example prediction alongside the HW4 reference for a quick manual check. `export_hw4.py` can instead be run with `%run -i` inside an existing HW4 notebook session containing `model_full`, `features_full`, and `predictions_full`.

## Local setup and checks

Run commands from `Homeworks/HW5/Theo_Basseras`. Use Python 3.12. The `python` command below must point to that version. On this Windows checkout, the course root's existing `.venv/Scripts/python.exe` is another available Python 3.12 interpreter.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q -s tests/test_app.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

No environment activation or PowerShell execution-policy change is needed with these explicit interpreter paths. Stop the server with Ctrl+C. Interactive API documentation is at http://127.0.0.1:8000/docs.

The saved pipeline and serving dependencies use numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.16.3, joblib 1.5.3, and threadpoolctl 3.6.0, matching the HW4 export environment. These versions are pinned in `requirements.txt` and installed in Docker too.

## Docker

Stop any existing service using host port 8000 before starting this container. Docker Desktop must be running.

```powershell
docker build --load -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The Dockerfile uses `python:3.12-slim` and copies only the requirements, API, fitted artifact, and reference JSON. The ignore file excludes local environments, databases, credentials, and caches. No custom transformer modules are needed for this StandardScaler/Ridge pipeline.

In another terminal in the submission folder, send valid and invalid requests:

```powershell
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@example_request.json"
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@invalid_request.json"
```

To reproduce the saved restart evidence, run `docker restart mfe-hw5` in another terminal, wait for startup to finish, and repeat the valid request. The evidence records the unchanged image ID and the new startup time. The fitted model is included in the image. The `--rm` option removes the stopped container, not the image.

## Validation and latency

All tests passed, including the two upper-limit tests at 10 and just above 10. They check real-model/direct/API/reference agreement, invalid inputs rejected before inference, a deliberately broken predictor returning 500, and latency. Ruff lint and formatting checks also passed. The test dependencies currently produce a Starlette/httpx deprecation warning; it does not fail the tests.

The saved November 12, 2026 verification run measured a median of 12.455 ms and p95 of 21.142 ms using `time.perf_counter`, seven successful warm-up requests, and thirty successful measured requests. Hardware was an Intel64 Family 6 Model 154 Stepping 4 GenuineIntel processor on Windows 11 build 26100. This is an in-process FastAPI TestClient measurement, not real localhost HTTP or Docker latency. Model loading is outside the timed section. No universal latency threshold is asserted; results vary with machine load.

The supplied Makefile retains `test`, `lint`, `format`, `build`, and `run` targets. On Windows without make, use the equivalent commands above. With make installed, select the HW5 interpreter using `make PYTHON=./.venv/Scripts/python.exe test` (or `lint`/`format`).

## Before a real deployment

Add authentication and HTTPS, request/resource limits, model monitoring and out-of-distribution checks, artifact integrity checks, and a model/version deployment policy. Validate financial usefulness separately: serving the model correctly does not establish investment performance.

## Sources

Application, test fixtures, Dockerfile, Makefile, and tooling were adapted from the teacher's [HW5 template](../template/). Model preparation and training come from my [HW4 submission (PR #214)](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/pull/214). AI assistance: I used ChatGPT to organize the submission and the saved-model link to HW4, and as a tutor for hints, explanations and examples; it also helped draft and debug helper, API and test code.

## Evidence files

- `evidence/http_requests_and_restart.txt`: real HTTP 200/422 responses and a verified same-image restart followed by another matching prediction.
- `evidence/checks_and_latency.txt`: successful Ruff checks and all tests, including the captured latency measurements and machine details.
