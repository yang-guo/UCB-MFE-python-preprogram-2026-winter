# Homework 5: Serve and Containerize a Model

# 1. Input & Output Contract
The `POST /predict` endpoint serves the fitted 90-day equity return Ridge regression pipeline ($\alpha = 10.0$) from HW4. Inputs are cross-sectionally winsorized (1%/99%) and z-scored (unitless, standard-deviation units) in exact training column order:

1. `ret_5d_z` (`float`, $[-10.0, 10.0]$): 5-trading-day return z-score
2. `ret_21d_z` (`float`, $[-10.0, 10.0]$): 21-trading-day return z-score
3. `ret_63d_z` (`float`, $[-10.0, 10.0]$): 63-trading-day return z-score
4. `ret_126d_z` (`float`, $[-10.0, 10.0]$): 126-trading-day return z-score
5. `mom_12_1_z` (`float`, $[-10.0, 10.0]$): 12-minus-1 month momentum z-score
6. `vol_21d_z` (`float`, $[-10.0, 10.0]$): 21-day realized volatility z-score
7. `vol_63d_z` (`float`, $[-10.0, 10.0]$): 63-day realized volatility z-score
8. `dist_ma200_z` (`float`, $[-10.0, 10.0]$): Distance from 200-day moving average z-score
9. `dd_252d_z` (`float`, $[-10.0, 10.0]$): 252-day drawdown z-score

* **Validation Rules:** `extra="forbid"`, `strict=True` (rejects numeric strings and booleans), `allow_inf_nan=False` (rejects `NaN`/`Inf`), and bounds $[-10.0, 10.0]$ justified by HW4's 1%/99% cross-sectional winsorization. Invalid requests return HTTP `422`.
* **Output Schema:** `{"prediction": <finite float>, "model_version": "hw4-ridge-alpha10-v1"}` (HTTP `200`).

# 2. Setup, Artifact Loading & Reproduction
`hw4_model.joblib` packages the fitted scikit-learn pipeline (`SimpleImputer` + `StandardScaler` + `Ridge(alpha=10.0)`), feature column order, model version, and a recorded out-of-sample reference input/prediction (`0.036645`) from HW4 (`2016-04-01`). It is loaded once at startup via FastAPI's `lifespan` handler into `app.state` without querying DuckDB or refitting.

    uv venv --python 3.12 .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    make lint
    make test

# 3. Docker Build & Run
Build and run the `python:3.12-slim` container locally on port `8000`:

    make build
    make run

# 4. Test, Lint & Latency Report
* **Lint & Format (`make lint`):** `All checks passed!`, all files formatted.
* **Test Suite (`make test`):** `10 passed` (covering valid schema, direct pipeline agreement, recorded HW4 reference value agreement, parametrized 422 validation + non-inference spy check, `NaN` 422 check, mocked 500 inference failure, and latency benchmark).
* **Latency Report (MacBook Pro, Apple Silicon, in-process `TestClient`, 5 warmup + 30 timed requests via `time.perf_counter`):**
  * **Median:** `1.28 ms`
  * **p95:** `1.67 ms` (well below a 50 ms in-process budget).

# 5. Container Request & Restart Transcript
    $ docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
    61d7f723180e58e87325b78187c45224b332d2292012589e41e28a7878936195

    $ curl -i -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
    HTTP/1.1 200 OK
    date: Mon, 16 Nov 2026 08:55:11 GMT
    server: uvicorn
    content-length: 73
    content-type: application/json

    {"prediction":0.03664499340784111,"model_version":"hw4-ridge-alpha10-v1"}

    $ curl -i -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"ret_5d_z": "invalid", "extra": 1}'
    HTTP/1.1 422 Unprocessable Entity
    date: Mon, 16 Nov 2026 08:55:11 GMT
    server: uvicorn
    content-length: 34
    content-type: application/json

    {"detail":"Invalid input payload"}

    $ docker stop mfe-hw5 && docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
    mfe-hw5
    a58741fee7ea2e6e1a5ffc1517cd795411aaaf7af42059dbe5e52dbbdf97b50a

    $ curl -i -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
    HTTP/1.1 200 OK
    date: Mon, 16 Nov 2026 08:55:13 GMT
    server: uvicorn
    content-length: 73
    content-type: application/json

    {"prediction":0.03664499340784111,"model_version":"hw4-ridge-alpha10-v1"}

# 6. Production Readiness Considerations
Before deploying to live trading infrastructure, the service would need:
1. **Authentication, Rate Limiting & TLS:** API key/OAuth authentication behind an HTTPS reverse proxy to restrict access and prevent abuse.
2. **Observability & Drift Monitoring:** Structured JSON logging, Prometheus latency/error metrics, and feature-distribution drift alerts against the 2010–2015 training baseline.
3. **Health Probes & Artifact Registry:** Dedicated `/health` and `/ready` endpoints for container orchestration, plus loading signed artifacts from a versioned object store with SHA-256 checksum verification rather than local disk.
