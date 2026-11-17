# HW5 · Serve the HW4 stock-return model

`POST /predict` serves **hw4-ridge-20160401-v1**, the fixed Ridge model selected
during HW4 development and used in the final April 2016–September 2026 backtest.
The response is a predicted simple return in decimal units. For example,
`0.0346974744` represents about 3.47%. The target starts at the next session's
adjusted close, then runs to the first quoted close on or after entry plus 90
calendar days. The endpoint forecasts that target. It does not rank stocks or
apply the HW4 universe and portfolio rules.

## Input and output

Send a JSON object with all eight keys. The order below is the order used
by the pipeline. The server reorders fields, so JSON key order can vary.

| Position | Field | HW4 definition | Allowed value |
| ---: | --- | --- | --- |
| 1 | `ret21` | Adjusted close at signal divided by adjusted close 21 trading sessions earlier, minus one | Finite number greater than −1 |
| 2 | `ret63` | Same simple return over 63 trading sessions | Finite number greater than −1 |
| 3 | `ret126` | Same simple return over 126 trading sessions | Finite number greater than −1, or `null` |
| 4 | `ret252skip21` | Adjusted close 21 sessions before signal divided by adjusted close 252 sessions before signal, minus one | Finite number greater than −1, or `null` |
| 5 | `vol63` | Sample standard deviation of the last 63 daily simple returns, unannualized | Finite number at least zero |
| 6 | `op_margin` | Quarterly operating income divided by revenue | Any finite number, or `null` |
| 7 | `cashflow_assets` | Quarterly operating cash flow divided by total assets | Any finite number, or `null` |
| 8 | `debt_assets` | Quarterly total debt divided by total assets | Any finite number, or `null` |

Inputs are decimal fractions. The HW4 signal uses the last available quarterly
filing as of filing date plus one day. Financial ratios more than 550 calendar
days beyond fiscal end are missing. The first two returns and the volatility
are required observations because HW4 eligibility requires 63 complete daily
returns. The longer lookbacks and financials can be missing. Use explicit
JSON `null` for those fields so the trained imputer receives `NaN`. A missing
key is a validation error. The return bounds follow from positive prices and
the volatility bound follows from its definition. Financial fields have no
artificial cap to preserve training values, including negative ratios and
ratios above one.

A valid request returns HTTP 200 with a finite number and a version string.
Missing or extra keys, array payloads, numeric strings, booleans, non-finite
numbers, and out-of-range fields return 422. Unexpected inference failures
return 500 with a generic message and are logged with a traceback.

## Artifact and reference

HW4 saved the code and backtest outputs but did not persist the fitted
estimator. `export_hw4.py` replays the **original HW4 final-test code** and
captures the pipeline created by that code. The replay matched the archived
portfolio metrics and rank correlation, used 615,519 training rows, and
confirmed March 28, 2016 as the latest training exit. The exported artifact
contains the fitted median imputer with missing indicators, the fitted
standard scaler, and Ridge with alpha 100. There are no custom transformers.

The original HW4 `predict` call during replay produced the reference score
**0.034697474382189356** for AAPL at the March 31, 2016 signal close, for entry
April 1. The exporter saved that output before any inference through FastAPI.
`reference_prediction.json` preserves the score and its row provenance.
`example_request.json` is the request from that same row. Tests compare the
API against both a direct saved-pipeline call and the replay reference using
`rel=1e-9` and `abs=1e-12`, which allow small float64 differences while
detecting a changed prediction.

`artifacts/hw4_ridge.joblib` is about 1.6 KB and is included. The adjacent
manifest records package versions, feature order, and SHA-256. The lifespan
hook checks the manifest and loads the fitted pipeline once per process.
Each prediction uses a one-row float64 DataFrame with the training names.
Serving and container startup require neither the database nor a training
step. Use only the supplied trusted joblib artifact. A checksum detects a
file mismatch and does not authenticate an untrusted bundle.

The optional export can be reproduced in a separate Python 3.12 **training**
environment with the original unzipped HW4 submission, course database, and
the HW4 dependencies. Use the model stack in this submission, plus
`duckdb==2.0.0` and `matplotlib==3.10.8` for the offline replay:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python export_hw4.py \
  --hw4-dir ../equity_backtest --db ../equities_top2000.duckdb
```

## Local setup and checks

From this submission directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m pytest -q -s -k performance
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In Windows PowerShell, create the environment with `py -3.12 -m venv .venv`
and activate it with `.\.venv\Scripts\Activate.ps1`. The remaining Python
commands are the same. Try the interactive [API docs](http://127.0.0.1:8000/docs).
`make test`, `make lint`, and `make format` run the corresponding test,
lint and formatting commands.

The run had a median of **2.404 ms** and a p95 of **3.145 ms**.
The latency test times 40 serial, successful requests after five warm-ups
with `time.perf_counter`. The reported p95 uses the nearest rank. It includes
JSON encoding, validation, inference and response handling in an in-process
TestClient, without TCP or container overhead. The machine was an x86-64
Linux workspace exposing nine logical CPUs on an AMD EPYC 9V74, running
Python 3.12.14. Raw output, including the supplied TestClient stack's
deprecation warning, is in `evidence/checks.txt`.

## Docker and HTTP evidence

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

From a second terminal in this directory:

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Stop that instance before running the repeatable smoke test below. It launches
the same image twice and records a valid request, a validation failure, and
the response after restart. If port 8000 is busy, pass `--port 8001`.

```bash
docker stop mfe-hw5
python smoke_http.py | tee evidence/docker_http.txt
```

GitHub Actions built the Docker image and passed the HTTP smoke test on
`main`. The workflow built the image, checked a valid request and a rejected
request, and checked that the same prediction and model version were returned
after restarting the same image.

- [Successful GitHub Actions run](https://github.com/guptaa-s/Python-5/actions/runs/43997796555)
- [Docker build and HTTP transcripts](https://github.com/guptaa-s/Python-5/actions/runs/43997796555/artifacts/14291430284)

The transcripts are included as `evidence/docker_build.txt` and
`evidence/docker_http.txt`. The GitHub links require repository access. The local process transcript in
`evidence/localhost_http.txt` records a separate Uvicorn check.

## Before deploying beyond the assignment

Add authentication, request limits and TLS at the ingress. Monitor request
latency, failures, missing-feature rates and distribution shifts. Validate
the freshness and corporate-action treatment of upstream data, and evaluate
the model on later resolved outcomes before relying on its signals. Publish
a signed image and model manifest with a documented rollback version. Keep
the training data outside the build context and rescan the image when
dependencies change.

## Implementation sources

The application lifespan, test fixtures, Makefile targets, and container
entry point were adapted from the supplied HW5 template. The feature logic
and estimator come directly from the prior HW4 submission. See
[scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
for version and trust constraints on joblib artifacts.
