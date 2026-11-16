# HW5 - Weitao Su

FastAPI service for my HW4 Ridge model (`alpha=1000`). It loads the saved model
and learned preprocessing once at startup, without training or database access.

## Setup and checks

From `Homeworks/HW5/Weitao_Su/`, using Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Windows: use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`.
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The Makefile also provides
`test`, `lint`, `format`, `build`, and `run` targets.

## API contract

`POST /predict` accepts exactly these seven named, finite numeric fields.
The table gives the training DataFrame order; JSON key order does not matter.
All units are decimal fractions (`0.05` = 5%). `P[t]` is the adjusted closing
price on the signal date, and lookbacks count market sessions.


| Field                   | Definition                                            | Allowed range              |
| ----------------------- | ----------------------------------------------------- | -------------------------- |
| `momentum_21`           | `P[t] / P[t-21] - 1`                                  | Greater than -1            |
| `momentum_63`           | `P[t] / P[t-63] - 1`                                  | Greater than -1            |
| `momentum_252_21`       | `P[t-21] / P[t-252] - 1`                              | Greater than -1            |
| `volatility_63`         | Sample std of 63 daily returns times`sqrt(252)`       | 0 through 0.8              |
| `drawdown_252`          | `P[t] / trailing 252-session maximum - 1`             | Greater than -1, at most 0 |
| `relative_momentum_63`  | Stock`momentum_63` minus `^SP500TR` 63-session return | Any finite number          |
| `momentum_acceleration` | `momentum_21 - momentum_63 / 3`                       | Any finite number          |

The caller computes these features. Positive prices justify the return/drawdown
bounds; the volatility cap matches HW4's screen. Other returns have no upper cap.
Missing/extra fields, arrays, strings, booleans, nulls, NaN, infinity, and values
outside the ranges return **422** before inference. This API requires complete
inputs even though the saved HW4 pipeline includes imputation.

The prediction is the adjusted-price return from the next session's close to the
close on or after 90 calendar days from entry. A valid request returns **200**:

```json
{"prediction": 0.02729199745191903, "model_version": "hw4-ridge-alpha1000-1c2a664-v1"}
```

Unexpected inference errors or non-finite outputs return **500** with
`{"detail":"Prediction unavailable"}`; detailed exceptions stay in server logs.

## Model and reference

Source: branch `hw4_Weitao_Su`, commit `1c2a664f5743d17cbf233becfcc16d68b8123aa4`.
The final fit uses 65,071 signals from January 2010 through November 2015, with
targets completed before March 31, 2016; the last outcome is February 29, 2016.
`artifacts/model.joblib` includes fitted quantile clipping, median imputation,
scaling, and Ridge. `course_data.py` supplies its importable `QuantileClipper`.
Only load trusted artifacts. Model-library versions in `requirements.txt` match
HW4's `uv.lock`; normal serving needs neither DuckDB nor the database.

HW4 saved results but no estimator. `export_hw4.py` reproduces the original final
fit, verifies its cross-validation results, and records a direct pipeline prediction
before API use. To repeat this optional export with the full bCourses database:

```sh
git fetch origin hw4_Weitao_Su
python -m pip install duckdb==1.5.6
python export_hw4.py --database /path/to/equities_top2000.duckdb
```

`example_request.json` is AAPL's February 29, 2016 signal, with March 1 entry.
`artifacts/reference_prediction.json` records its input and expected result.
Tests compare the direct model call and saved reference using `rel=1e-8` and
`abs=1e-10` to allow small floating-point differences between machines.

## Docker and requests

Stop any local server using port 8000. The image uses `python:3.12-slim` and
contains the fitted artifact and custom transformer. Wait for startup in
`docker logs mfe-hw5` before requests, including after restart.

```sh
docker build -t mfe-hw5:local .
docker run -d --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data '{}'
docker stop mfe-hw5
docker start mfe-hw5
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
docker rm -f mfe-hw5
```

## Results and deployment

[evidence.txt](evidence.txt) records **96 passing tests**, passing lint/format
checks, and real container responses **200, 422, and 200 after restart**.
Tests cover direct/reference agreement, validation without inference, forced
inference failures, startup loading, and performance.

Latency: **1.018 ms median, 1.279 ms p95**, using in-process `TestClient` on an
Apple M3 Max, 36 GiB RAM, macOS arm64, Python 3.12.12. Five warm-up calls precede
50 sequential successful requests timed with `time.perf_counter`; startup is
excluded and p95 uses numpy's linear percentile. Run `python -m pytest -q -s`
to print the measurement. No timing cutoff is imposed because machine load varies.

Before deployment: add authentication, TLS, request limits, monitoring, concurrent
load tests, a non-root container user, and model release/rollback procedures.
Monitor feature drift and predictive quality; HW4's model underperformed.
