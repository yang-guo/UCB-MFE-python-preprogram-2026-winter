# HW5 — Jiajing Wang

Serves the HW4 StandardScaler + Ridge pipeline (alpha 1000), recreated with the
original HW4 training code and data. The fit used 135,394 development rows, with
the last training label dated 2016-02-29. The prediction is a simple stock return
over roughly 90 calendar days, expressed as a decimal.

## API

POST /predict accepts one JSON object with a `features` array of exactly six
finite JSON numbers in the following order:

| Position | Feature | Definition |
| --- | --- | --- |
| 1 | return_5 | 5-session adjusted-close return |
| 2 | return_10 | 10-session adjusted-close return |
| 3 | return_21 | 21-session adjusted-close return |
| 4 | return_63 | 63-session adjusted-close return |
| 5 | return_252_skip_21 | Adjusted close at session −21 divided by adjusted close at −252, minus 1 |
| 6 | volatility_63 | Sample standard deviation of 63 daily adjusted-close returns, unannualized |

Features use data through the signal close before entry and are in decimal units.
Returns must exceed −1 because the underlying prices are positive. Volatility
must be nonnegative. The API reconstructs the named training DataFrame.

A valid request returns 200 with a finite `prediction` and `model_version`.
Invalid requests return 422 and unexpected inference errors return 500.

The lifespan loads the trusted `artifacts/model.joblib` once per process. It
contains both the fitted scaler and Ridge. No pricing data is needed at runtime.
The input in `example_request.json` is ticker A at the 2016-03-31 signal close.
`artifacts/reference.json` records the direct HW4 pipeline prediction before
the API was written, plus the training environment versions. The test uses an
absolute tolerance of 1e-10 for floating-point differences.

## Run

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make lint
python -m pytest -q -s
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

`verification.txt` records lint, tests and latency. It measures sequential
in-process TestClient requests on an Apple Silicon Mac, with 5 warm-ups and
30 timed requests.

## Docker

```sh
make build
make run
```

From another terminal in this folder:

```sh
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"features":[0]}'
docker stop mfe-hw5
```

Run `make run` again and repeat the valid request. Expected statuses are 200,
422 and 200. See `docker-verification.txt` for the measured responses, including
the prediction after restarting the container.

Before deployment, add authentication, request limits and monitoring for
latency, inference failures and feature drift.

Adapted from the course HW5 starter with AI assistance. Model and preprocessing
come from the HW4 submission.
