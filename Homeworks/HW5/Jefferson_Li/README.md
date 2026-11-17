# HW5 — Jefferson Li

## Setup

From this folder, create a separate Python 3.12 environment with uv:

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

The app, tests, and Dockerfile use the [course starter](../template/).

## Model

`hw4-ridge-v1` is the fixed StandardScaler and Ridge (alpha 100) pipeline from
[HW4, commit 12947ed](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/12947ede850bfd79f2303a094911687896d5cf35/Homeworks/HW4/Jefferson_Li/run_final.py).
It was recreated using the submitted training code and the verified full course
database. The 126 monthly selections and scores matched the saved HW4 results.
The fitted pipeline is in `artifacts/model.joblib`.

Inputs are `momentum_63` (63-session adjusted-close return as a fraction) and
`volatility_63` (annualized standard deviation of the last 63 daily adjusted-close
returns as a fraction), in that column order. The output estimates the gross
adjusted-close return over 90 calendar days from the next session's close.

`example_request.json` is the SKY signal on 2016-04-01. The direct HW4
pipeline prediction was `0.5395505648644271`, matching the saved monthly
selection. Keep this value as the reference for the API test.

## API

`POST /predict` accepts exactly two named JSON numbers. Both must be finite.
`momentum_63` must be greater than `-1` because both prices in its ratio are
positive. `volatility_63` must be at least `0` because standard deviation is
nonnegative. Neither feature has an upper bound. Send values as fractions
(e.g. `0.20` for 20%). Integers and decimals are accepted.

The app loads the local pipeline at startup. A successful response is HTTP 200
with a finite `prediction` and `model_version`. Invalid input returns 422.
Unexpected inference failures return 500 with a generic message.

From this folder with the HW5 environment active, start the service:

```sh
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In another terminal in this folder, send the example request:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

## Tests and latency

Run `make test` and `make lint` with the HW5 environment active. The checks
reported `16 passed`, `All checks passed!`, and `3 files already formatted`.
The SKY prediction matches the direct pipeline and the HW4 reference
within a tolerance of `1e-12`. Invalid inputs are rejected before inference,
and an injected model failure returns HTTP 500.

On an Apple M2 Mac (8 GB RAM, Python 3.12.9), median latency was **1.123 ms**
and p95 was **1.234 ms**. The test used `time.perf_counter()` for 50 sequential
in-process `TestClient` requests after five warm-ups, with the model already
loaded. The p95 uses the nearest-rank method. The timing covers the local
request and response path without a network socket.

```sh
python -m pytest -q -s tests/test_app.py::test_predict_performance
```

## Docker

With Docker running, build and start the image from this folder:

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The image uses Python 3.12 and the pinned requirements. It contains the
application and fitted pipeline. Once it starts, run the same curl request
shown above. If port 8000 is busy, use `-p 127.0.0.1:8001:8000` and send
requests to port 8001.

The [container transcript](container-check.txt) shows a valid request, a
rejected request, and the same prediction after starting a new container
from the same image.

## Before deployment

Use authenticated HTTPS and rate limits, and monitor error rates, latency,
and input drift. Version the model and feature definitions together so the
team can reproduce predictions and roll back safely.
