# HW5 – Serve and Containerize a Machine Learning Model

## 1. Overview

This project serves the fitted Random Forest regression pipeline developed in HW4 through a REST API using FastAPI and containerizes the application using Docker.

The API accepts 12 numerical input features and returns a model prediction along with the model version.

The application:
- Loads the saved model once during application startup.
- Validates incoming requests using Pydantic.
- Preserves the feature names and order used during model training.
- Handles invalid requests and inference failures.
- Runs inside a Docker container.

## 2. Project Structure

```text
Vanditha_Toleti/
├── app.py
├── course_data.py
├── model/
│   └── model_artifact.joblib
├── tests/
│   └── test_app.py
├── evidence/
│   ├── valid_request.txt
│   ├── invalid_request.txt
│   └── restart_request.txt
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Makefile
├── example_request.json
└── README.md
```

## 3. Model and Features

- **Model:** Random Forest Regressor
- **Artifact:** `model/model_artifact.joblib`
- **Model version:** `hw4_random_forest_2025-12-31`
- **Framework:** scikit-learn
- **Custom transformer:** `QuantileClipper`, defined in `course_data.py`

### Model training and provenance

The deployed model is the final model from the HW4 yearly-refit backtest.

- **Training database:** `equities_top2000.duckdb`
- **Refit schedule:** Final model from 11 yearly refits
- **Final fit date:** 2025-12-31
- **Training observations:** 372,052 monthly rows
- **Signal history begins:** 2010-01-29
- **Target availability:** Every training outcome ended before 2025-12-31.
- **Target processing:** Training targets were clipped at the 1st and 99th percentiles.
- **Prediction period:** 2025-12-31 to 2026-09-30

The fitted pipeline was serialized as a Joblib artifact for reuse by the API.

### Model artifact export

The fitted pipeline is saved using Joblib:

```python
import joblib

joblib.dump(
    model,
    "model/model_artifact.joblib",
)
```

Here, `model` refers to the fitted HW4 pipeline used for the final yearly refit.

The custom transformer implementation required by the serialized pipeline is included in `course_data.py` so that the artifact can be loaded successfully.

### Input features

The API expects the following 12 features in the original training order:

| Feature | Description / Unit |
|---|---|
| `momentum_21` | 21-day momentum, decimal return |
| `momentum_63` | 63-day momentum, decimal return |
| `momentum_252_21` | 252-day momentum excluding the latest 21 days, decimal return |
| `volatility_63` | 63-day volatility, decimal |
| `drawdown_252` | 252-day maximum drawdown, decimal |
| `net_margin` | Net income / revenue |
| `operating_roa` | Operating income / total assets |
| `debt_to_assets` | Total debt / total assets |
| `cash_to_assets` | Cash / total assets |
| `fcf_to_assets` | Free cash flow / total assets |
| `accruals_to_assets` | Accruals / total assets |
| `filing_age_days` | Age of financial filing in calendar days |

All financial ratios and return features use decimal representation rather than percentages.

### Input validation

All 12 features are required by the API, even though the underlying fitted pipeline includes an imputation step for missing feature values.

The API applies the following validation rules:

- Only numerical inputs are accepted.
- Numeric strings and booleans are rejected.
- NaN and infinity values are rejected.
- Momentum features must be at least -1.
- `volatility_63` must be non-negative.
- `drawdown_252` must be between -1 and 0.
- `debt_to_assets` must be non-negative.
- `cash_to_assets` must be between 0 and 1.
- `filing_age_days` must be between 0 and 550.
- Missing or unexpected fields are rejected.

## 4. API

### POST `/predict`

Accepts a JSON request containing all 12 model features.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

Successful response:

```json
{
  "prediction": 0.036766915694316164,
  "model_version": "hw4_random_forest_2025-12-31"
}
```

### HTTP status codes

| Status | Description |
|---|---|
| 200 | Successful prediction |
| 422 | Invalid request or input validation failure |
| 500 | Internal model inference failure |

Interactive API documentation is available at:

`http://localhost:8000/docs`

## 5. Reference Prediction

The example request uses a feature row from the HW4 monthly panel:

- **Symbol:** A
- **Date:** 2025-12-31

The reference prediction used in the API regression test is:

```text
0.036766915694316164
```

The API prediction is compared with direct model inference using a numerical tolerance to account for floating-point differences.

## 6. Local Setup

**Python version:** 3.12

Create and activate a virtual environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Run a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

## 7. Testing and Code Quality

Run the test suite:

```bash
python -m pytest -v -s
```

The test suite covers:

- Valid prediction and agreement with direct model inference.
- Reference prediction verification.
- Missing and unexpected input fields.
- Invalid types, booleans, and non-finite values.
- Feature-specific validation constraints.
- Model inference failure handling.
- Performance measurement using 25 requests after a warm-up request.

### Test results

Latest test run:

```text
4 passed, 1 warning
```

The warning is a non-fatal Starlette deprecation warning related to the current TestClient/httpx integration.

### Code quality checks

```bash
python -m ruff check .
python -m ruff format --check .
```

Both checks passed.

### Performance results

Measured using FastAPI TestClient on a MacBook Air (Apple Silicon), Python 3.12.9.

| Metric | Result |
|---|---:|
| Requests measured | 25 |
| Median latency | 14.735 ms |
| P95 latency | 15.010 ms |

A warm-up request was excluded from the measurements.

These measurements represent in-process TestClient latency, not network or production deployment latency.

## 8. Docker

The application uses the official Python 3.12 slim base image.

### Build the Docker image

```bash
docker build -t hw5-stock-api .
```

### Run the container

```bash
docker run -d \
  --name hw5-stock-container \
  -p 127.0.0.1:8000:8000 \
  hw5-stock-api
```

The port is bound to localhost.

### Check container status

```bash
docker ps
```

### View container logs

```bash
docker logs hw5-stock-container
```

### Test the API

```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

### Container restart verification

The container was stopped and restarted using:

```bash
docker stop hw5-stock-container
docker start hw5-stock-container
docker ps
```

A subsequent prediction request returned HTTP 200 with the expected prediction and model version.

The captured evidence is available in the `evidence/` directory:

| File | Verification |
|---|---|
| `valid_request.txt` | Valid request returns HTTP 200 |
| `invalid_request.txt` | Invalid request returns HTTP 422 |
| `restart_request.txt` | Prediction succeeds after container restart |

Small last-digit differences may occur between repeated predictions because the Random Forest uses parallel tree execution (`n_jobs=-1`), which can change the floating-point summation order. These differences are numerical precision effects and are handled using a tolerance in the prediction tests.

## 9. Implementation Notes

- The fitted HW4 pipeline is loaded once per server process using FastAPI lifespan management.
- The custom `QuantileClipper` module is included to support deserialization of the saved model artifact.
- Input features are passed to the model in the original training feature order.
- Pydantic performs request validation before model inference.
- Model inference failures are logged and returned as HTTP 500 errors.
- Docker packages the application, dependencies, custom transformer module, and model artifact.
- The `.dockerignore` file excludes unnecessary local files and development artifacts from the image.

## 10. Production Deployment Considerations

A production deployment would require additional safeguards and operational capabilities:

- **Authentication:** Protect the prediction endpoint using API keys or an identity provider.
- **Monitoring:** Track API availability, request volume, latency, and error rates.
- **Logging:** Maintain structured logs for validation failures and inference errors without exposing sensitive information.
- **Input drift:** Monitor incoming feature distributions against training data.
- **Model performance:** Evaluate predictions against subsequently observed outcomes.
- **Retraining:** Establish a periodic or performance-triggered retraining process.
- **Model versioning:** Maintain reproducible artifacts and rollback capabilities.
- **Container security:** Pin base-image and dependency versions and scan images for vulnerabilities.

## 11. Credits and References

- `course_data.py` is based on the course-provided modeling material in `Lectures/modeling/`.
- The fitted model artifact originates from the HW4 modeling work.
- FastAPI, Pydantic, scikit-learn, Docker, and pytest documentation were used as technical references.
- AI assistance was used during implementation, debugging, testing, and documentation.
