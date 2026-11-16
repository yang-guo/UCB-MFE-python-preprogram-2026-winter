# Homework 5: Stock Return Prediction Service & Docker Container

This service wraps the fitted HW4 equity return model (`StandardScaler` + `Ridge(alpha=10.0)`) into a tested FastAPI endpoint (`POST /predict`) packaged in a `python:3.12-slim` Docker container.

## 1. Served Model & Artifact Details

- **Model Version:** `hw4-ridge-alpha10-v1`
- **Pipeline:** `sklearn.pipeline.Pipeline` combining `StandardScaler` and `Ridge(alpha=10.0)` trained on eligible US equities (`universe_membership` top 2,000) from 2010-01-01 through 2015-12-31.
- **Prediction Horizon:** Gross 90-calendar-day forward return (63 trading sessions) entered at the next session's close (`t+1`) following a signal generated at close `t`.
- **Artifact Persistence:** Serialized in `model_artifact.joblib` (1,640 bytes, included directly in Git) using `scikit-learn==1.8.0`, `numpy==2.3.5`, `pandas==2.3.3`, and `joblib==1.5.3`.
- **Loading Strategy:** Loaded once at server startup inside FastAPI's `lifespan` handler (`app.state.pipeline`), avoiding per-request disk I/O or database access.
- **HW4 Reference Sample (`example_request.json`):** Extracted from `AAPL` on signal date `2016-03-31` (entry `2016-04-01`), producing reference prediction `0.039889326526`.

*(To regenerate `model_artifact.joblib` from the HW4 database, run `python export_hw4_model.py` from the repository root environment with `equities_top2000.duckdb` present in `Lectures/modeling/data/`.)*

---

## 2. API Input / Output Contract (`POST /predict`)

### Request Schema (`InputData`)
JSON object with `extra="forbid"` and strict numeric types (`StrictFloat | StrictInt`; strings, booleans, `NaN`, and `Infinity` are rejected with HTTP `422`):

| Field | Order | Type | Unit | Allowed Range | Description |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `mom_20d` | 1 | `float \| int` | Decimal return | `[-1.0, 50.0]` | 20-trading-day trailing return at signal close `t` |
| `mom_60d` | 2 | `float \| int` | Decimal return | `[-1.0, 50.0]` | 60-trading-day trailing return at signal close `t` |
| `vol_20d` | 3 | `float \| int` | Decimal std dev | `[0.0, 5.0]` | 20-trading-day trailing daily return volatility at close `t` |

### Response Schema (`OutputData`)
- **HTTP `200 OK`:**
  ```json
  {
    "prediction": 0.03988932652573104,
    "model_version": "hw4-ridge-alpha10-v1"
  }
  ```
- **HTTP `422 Unprocessable Entity`:** Missing fields, extra fields, non-object payloads, non-numeric or boolean values, out-of-range values, or non-finite floats (`NaN` / `Infinity`).
- **HTTP `500 Internal Server Error`:** Unexpected runtime errors during inference (details logged server-side, generic error message returned to the client).

---

## 3. Local Setup, Testing, and Linting

From `Homeworks/HW5/Elouann_Frantz/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make format
make lint
make test
```

### Verification & Latency Report
- **Linter & Formatter:** `ruff check .` and `ruff format --check .` pass with zero warnings or diffs.
- **Automated Tests (`pytest`):** All **12 tests** pass (`12 passed in 1.52s`), covering valid HW4 reference agreement, 9 parametrized `422` validation cases (verifying `pipeline.predict` is never invoked), `500` inference failure handling, and latency benchmarking.
- **Latency Measurement:** Measured in-process via Starlette `TestClient` across `n = 30` requests after 5 warm-up calls on an Apple M3 MacBook Pro (macOS, Python 3.12.10):
  - **Median latency:** `1.17 ms`
  - **p95 latency:** `1.38 ms` (well within the `250 ms` single-row inference budget)

---

## 4. Docker Build & Container Verification

### Build and Run Commands
```bash
make build
make run
```
*(Equivalent to `docker build -t mfe-hw5:local .` and `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local`.)*

### Container Request & Restart Transcript
```text
=== 1. Valid request ===
HTTP/1.1 200 OK
date: Mon, 16 Nov 2026 13:42:31 GMT
server: uvicorn
content-length: 73
content-type: application/json

{"prediction":0.03988932652573104,"model_version":"hw4-ridge-alpha10-v1"}

=== 2. Invalid request ===
HTTP/1.1 422 Unprocessable Entity
date: Mon, 16 Nov 2026 13:42:31 GMT
server: uvicorn
content-length: 326
content-type: application/json

{"detail":[{"type":"float_type","loc":["body","mom_20d","float"],"msg":"Input should be a valid number","input":"0.08"},{"type":"int_type","loc":["body","mom_20d","int"],"msg":"Input should be a valid integer","input":"0.08"},{"type":"extra_forbidden","loc":["body","extra"],"msg":"Extra inputs are not permitted","input":1}]}

=== 3. Stop and restart container ===
mfe-hw5
06d139a74836ce932e83e4954b9daadc08ce02cd37980fee8aad187bac557e75

=== 4. Valid request after restart ===
HTTP/1.1 200 OK
date: Mon, 16 Nov 2026 13:42:34 GMT
server: uvicorn
content-length: 73
content-type: application/json

{"prediction":0.03988932652573104,"model_version":"hw4-ridge-alpha10-v1"}
```

---

## 5. Production Readiness Considerations

Before deploying this service to a live trading or client-facing environment, the following areas require attention:
1. **Security & Access Control:** Add API key or OAuth2/JWT authentication, TLS termination (via a reverse proxy such as Nginx or an AWS ALB), rate limiting, and run the Docker container as a dedicated non-root user.
2. **Observability & Health Checks:** Expose a lightweight `GET /health` endpoint for container orchestrators (Kubernetes/ECS) and emit structured JSON logs with request IDs, latency histograms, and input/prediction distributions to detect feature drift or regime shifts.
3. **Upstream Feature Engineering & Batching:** Live callers typically pass `(ticker, as_of_date)` rather than precomputed momentum/volatility features; a production pipeline should fetch validated, split-adjusted market data from an online feature store and support batch predictions (`list[InputData]`) for scoring the full 2,000-stock universe at once.
4. **Artifact Integrity:** Store versioned model artifacts in an immutable object store (e.g., S3) with SHA-256 checksum verification at startup rather than bundling binaries directly inside source control.
