# HW5: Serve the submitted HW4 model

## Model and prediction contract

This service loads the final Ridge pipeline submitted for HW4 (source commit `c18b63174ebdb251568be66f06afe0ea25361824`). The artifact is fitted once using the HW4 training procedure and includes the learned feature clipping, median imputation, scaling, and Ridge estimator. The model is not fitted or updated in response to API requests.

`POST /predict` accepts one JSON object with these five features in the order used by HW4:

| Feature | Meaning and unit | Allowed values |
|---|---|---|
| `momentum_21` | Adjusted-price return over 21 market sessions, as a fraction | Greater than -1; finite number |
| `momentum_63` | Adjusted-price return over 63 market sessions, as a fraction | Greater than -1; finite number |
| `momentum_252_21` | Adjusted-price return from 252 to 21 sessions ago, as a fraction | Greater than -1; finite number |
| `volatility_63` | Annualized volatility estimated from 63-session returns, as a fraction | Zero or greater; finite number |
| `drawdown_252` | Current adjusted price relative to its trailing 252-session high | -1 through 0; finite number |

Values are fractions, not percentage points: `0.05` means 5%. The model predicts the adjusted-price return from the next market session's close through the first market session on or after 90 calendar days. A valid response contains a finite `prediction` and `model_version`. Invalid requests receive 422; inference failures receive 500 without exposing exception details.

The bounds follow the feature definitions: a simple return from positive prices is greater than -1, volatility cannot be negative, and drawdown from a trailing high lies between -1 and 0. Every field must be a finite JSON number. Missing or extra fields, strings, booleans, non-finite values, and out-of-range values are rejected with 422.

The example request is a feature row from the HW4 experiment at the model fit date, March 31, 2016. `artifacts/model_reference.json` records the direct HW4 pipeline prediction for that same row before the API wrapper; the positive API test compares against both that value and a direct pipeline call.

## Setup and checks

Use Python 3.12.9 in a separate environment. These commands run from this folder:

```sh
uv venv --python 3.12.9 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest -q -s
python -m ruff check .
python -m ruff format --check .
```

The model-library versions match the HW4 training environment because loading a scikit-learn artifact across versions is unsupported. The fitted artifact and metadata are under `artifacts/`; the HW4 database and virtual environments are not included.

To run the service locally:

```sh
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The service loads the artifact once at startup. `/health` reports whether it loaded successfully.

## Docker and request demonstration

From this folder, build and run the container:

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

In another Terminal window, send a valid request:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/example_request.json
```

Send an invalid request and confirm it is rejected:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{"momentum_21":"not-a-number"}'
```

Stop the container with Ctrl-C, start the same image again, and repeat the valid request to show the saved model is still available. Save a short terminal transcript or screenshot covering the valid request, invalid request, and request after restart.

## Performance and operational notes

The automated latency test warms up the endpoint and measures 20 successful in-process `TestClient` requests using `time.perf_counter`. On an Apple Silicon Mac (Darwin arm64) with Python 3.12.9, the measured median was 0.833 ms and p95 was 0.999 ms. This is not a real network or container latency measurement. No universal latency threshold is imposed.

Before real deployment, add authenticated access, HTTPS, monitoring and alerting, artifact integrity controls, model/data drift checks, and a tested rollback process. This homework verifies model serving and packaging; it does not repeat or make new claims about HW4 investment performance.

## Course references and assistance

The request validation, lifespan loading, error handling, tests, and container layout adapt the structure of `Lectures/Lecture 9/demo/`; that lesson's synthetic model is not used. `course_data.py` contains the `QuantileClipper` required to load the HW4 pipeline, adapted from `Lectures/modeling/course_data.py`. The prediction features, training labels, chosen alpha, and fitted pipeline come from the final HW4 submission identified above. Codex assisted with portions of implementing and debugging this service, tests, and documentation.
