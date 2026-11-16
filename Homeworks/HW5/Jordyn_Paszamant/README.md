# Homework 5 — Serve the HW4 Random Forest

`POST /predict` scores one stock as of a signal close using model version
`hw4-rf-2016-04-01`. The prediction is a decimal adjusted-price return.
The HW4 target enters at the next market session's close and exits at the
first valid close on or after 90 calendar days from entry. For example, a
prediction of `0.1314` means approximately 13.14%.

## Model and preprocessing

This is the first annual fit from
`Homeworks/HW4/Jordyn_Paszamant/Homework_4_Stock_Strategy_V_Final.ipynb`.
It used 136,687 observations, with the last training exit on March 7, 2016.
The original HW4 loop discarded each fitted model after scoring its annual
window. To save the first one, I reran the notebook's data and feature cells
and the first iteration of its annual training and scoring loop against the
same course database. The model has 200 trees, depth 6, a minimum leaf size of
100, and seed 42. HW4 clipped each input feature and the training target at its
training sample's 1st and 99th percentiles. The saved pipeline combines the
learned **input** clip thresholds with the fitted forest. Serving uses one
forest prediction thread to reduce overhead for a single row.

The API loads `artifacts/hw4_random_forest.joblib` once per worker at startup.
`model.py` contains the importable clipping function. The service needs the
checked-in artifact and matching model-library versions in `requirements.txt`.
It makes predictions without database access or fitting. Only load trusted
joblib artifacts because loading them can execute code.

`example_request.json` contains the raw feature row for EGRX on April 1, 2016.
The original notebook's stock-selection table records the prediction as
`0.1314`. `tests/hw4_reference.json` preserves the full-precision prediction
from rerunning the original notebook code before wrapping the estimator in
the API. The test checks both records and compares the API against a direct
pipeline call using an absolute tolerance of `1e-12` for direct agreement and
`1e-10` for the notebook replay. The small tolerances allow for floating-point
summation order.

## API contract

Send a JSON object with exactly these eight named fields. JSON key order is
arbitrary. The service constructs a float64 DataFrame in the training order
shown below. All inputs are finite JSON numbers in decimal units. Use `0.05`
for 5%. Quoted numbers and booleans are rejected.

| Training order | Field | HW4 definition | Accepted range |
| --- | --- | --- | --- |
| 1 | `momentum_21` | Adjusted close divided by its 21-session lag, minus 1 | ≥ −1 |
| 2 | `momentum_63` | Adjusted close divided by its 63-session lag, minus 1 | ≥ −1 |
| 3 | `momentum_126` | Adjusted close divided by its 126-session lag, minus 1 | ≥ −1 |
| 4 | `momentum_252` | Adjusted close divided by its 252-session lag, minus 1 | ≥ −1 |
| 5 | `volatility_21` | Sample standard deviation of the latest 21 one-session decimal returns | ≥ 0 |
| 6 | `volatility_63` | Sample standard deviation of the latest 63 one-session decimal returns | ≥ 0 |
| 7 | `ma_ratio_50` | Adjusted close divided by the 50-session mean, minus 1 | [−1, 49] |
| 8 | `ma_ratio_200` | Adjusted close divided by the 200-session mean, minus 1 | [−1, 199] |

The windows use the HW4 market calendar and end at the signal close. Moving
averages include the current close, so the ratio before subtracting 1 is at
most the window length for nonnegative prices. The inclusive boundaries
allow rounding at the mathematical limits. Momentum and volatility have no
fixed upper bounds, and the saved thresholds handle large finite values.
Send the original features before clipping.

- **200** returns a finite numeric `prediction` and string `model_version`.
- **422** returns validation errors for missing or extra fields, invalid types,
  incorrect shape, and values outside the accepted ranges.
- **500** returns a generic prediction error. Exception details go to server logs.

## Local setup and verification

From this folder, using Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make lint
make test
python -m pytest -q -s tests/test_app.py::test_predict_performance
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the interactive docs at `http://127.0.0.1:8000/docs`. Ruff lint and
format checks passed and pytest passed all 17 cases. The in-process
FastAPI `TestClient` run on an Apple M2 Mac with 8 GB RAM, macOS 15.7.4,
and Python 3.12.9 measured **6.09 ms median** and **7.43 ms p95**.
Those are 30 sequential successful requests measured with
`time.perf_counter` after five warm-up requests, with the artifact
already loaded. See [test output](test-output.txt) and the
[local HTTP transcript](local-http-output.txt).

## Docker reproduction and verification

On Docker Desktop for Apple silicon, the image built successfully and
returned HTTP 200 for the EGRX example, HTTP 422 for an empty object, and
HTTP 200 with the same prediction and model version from a new container
created from the same image. See the [container transcript](container-output.txt).

Run the following from this folder with Docker running and port 8000 free.

```sh
(
  set -ex
  make build
  docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
  curl -fsS --retry 15 --retry-connrefused --retry-delay 1 \
    http://127.0.0.1:8000/openapi.json -o /dev/null
  curl -i http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' --data-binary @example_request.json
  printf '\n'
  curl -i http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' --data '{}'
  printf '\n'
  docker stop mfe-hw5
  docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
  curl -fsS --retry 15 --retry-connrefused --retry-delay 1 \
    http://127.0.0.1:8000/openapi.json -o /dev/null
  curl -i http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' --data-binary @example_request.json
  printf '\n'
  docker stop mfe-hw5
) 2>&1 | tee container-output.txt
```

Verify that the transcript shows a successful build, a 200 and a 422 from the
first container, and another 200 with the same model version and prediction from
the second. Include `container-output.txt` in the PR and assign `kevinramlal`
and `raghuramkowdeed`.

## Before a real deployment

Validate live feature calculations against the HW4 implementation and monitor
freshness, missing data, extreme values, and clipping rates. Track latency and
error rates and evaluate predictive drift as realized 90-day labels arrive.
Add authentication, request limits, automated artifact verification, and a
rollback path. This is a historical 2016 fit and would need fresh out-of-sample
evaluation before any live trading use.

## Sources and assistance

I adapted the app lifespan, tests, Makefile, and tooling configuration from the
course HW5 template. The feature definitions, clipping, fit, and example come
from my HW4 notebook. Codex helped replay and export the fit, implement the
service and tests, and verify the local request and restart behavior.
