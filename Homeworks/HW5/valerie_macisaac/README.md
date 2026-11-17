# HW5: FastAPI Financial Prediction Service

A containerized FastAPI web service that serves cross-sectional stock return predictions from a 15-feature ensemble pipeline (`RidgeCV` + `HistGradientBoostingRegressor`), serialized as `model_artifact.joblib`.

## Files in the Folder

1. **`export_model.py`**: The training/export script from HW4. It is a much shorter version of HW4 since it does not show any of the steps leading to the model developed in HW4 and instead is just the `main` function that runs the model. Inside `main`, it imports the 15 features, `CrossSectionalRankTransformer`, and `HW4StockReturnEnsemble` from `model_utils`, generates all required features, determines the trading dates, applies the screen, and outputs the train and test data. Finally, it exports the result to `model_artifact.joblib`.
2. **`model_utils.py`**: A module containing shared helper classes and definitions—including `FEATURES_15`, `CrossSectionalRankTransformer`, and `HW4StockReturnEnsemble`—adapted from HW4.
3. **`model_artifact.joblib`**: The serialized output of the trained scikit-learn pipeline. It stores the feature list, the model version, a reference input (matching `example_request.json`), and the reference prediction.
4. **`app.py`**: The main FastAPI web service. It loads `model_artifact.joblib` at startup and enforces strict Pydantic schema and bounds validations, handling non-finite values or invalid edge cases with HTTP `422`.
5. **`example_request.json`**: A sample JSON payload containing realistic values for all 15 features that can be used for manual API testing.
6. **`tests/`**: Contains the `pytest` suite that verifies `/predict` responses, schema/bounds validation errors (`422`), and inference failure handling (`500`).
7. **`pyproject.toml`**: Configuration file for development tools (such as `ruff`) and settings for `pytest`.
8. **`Makefile`**: Developer shortcut commands such as `make format` and `make lint`.
9. **`requirements.txt`**: Pinned list of Python packages required to run and test the project.
10. **`Dockerfile`**: Step-by-step instructions for building the container image, installing dependencies from `requirements.txt`, copying the runtime files, and launching Uvicorn on port `8000`.
11. **`README.md`**: Documentation explaining the project structure, local setup, testing commands, and Docker deployment instructions.
12. Two Screenshots to show that the image can be started, can reject an invalid request and then can be restarted and succeed again. 
---

## Local Setup & Testing

### 1. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Linting, Formatting, and Tests
```bash
make format
make lint
python -m pytest -s
```

### 3. Run the Server Locally
```bash
uvicorn app:app --reload --port 8000
```

---

## Docker Containerization

### Build & Run
```bash
docker build -t hw5-stock-api .
docker run --rm -d -p 8000:8000 --name hw5-container hw5-stock-api
```

### Test the Containerized Endpoint
```bash
curl -i -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H "Content-Type: application/json" \
  --data-binary @example_request.json
```

Expected response (`200 OK`):
```json
{"prediction":-0.0013615300561944277,"model_version":"hw4-15feat-ridge-gbdt-v1.0.0"}
```

### Stop the Container
```bash
docker stop hw5-container
```


---

## API Input/Output Contract & Artifact Loading

- **Artifact Loading:** On startup (via FastAPI `lifespan`), `app.py` loads `model_artifact.joblib` using `joblib.load()`, extracting the trained scikit-learn `Pipeline`, `model_version`, and expected `feature_names` (`FEATURES_15`).
- **Endpoint:** `POST /predict` (`Content-Type: application/json`)
- **Input Schema:** A JSON object mapping all 15 required features (defined in `model_utils.FEATURES_15`, e.g., `ebitda_to_assets`, `roa`, `ret_252d`, etc.) to finite floats (`NaN`, `Inf`, missing fields, and extra fields return HTTP `422`).
- **Output Schema (`200 OK`):**
  ```json
  {
    "prediction": -0.0013615300561944277,
    "model_version": "hw4-15feat-ridge-gbdt-v1.0.0"
  }
  ```

---

## Verification & Latency Summary

- **Linting & Formatting (`make lint`):** `All checks passed!` (`ruff check` and `ruff format --check` clean).
- **Automated Tests (`python -m pytest -s`):** All 16 tests passed (valid prediction matching HW4 reference output, `422` validation errors for missing/extra/non-finite inputs, `500` error handling on inference failure, and latency benchmark).
- **Latency Benchmark:** Sub-10 ms p50/p95 single-request inference latency via FastAPI `TestClient`
- **Container Persistence:** Verified with `Terminal_screenshot_verification_part1.png` and `Terminal_screenshot_verification_part2.png` (`200 OK` valid response, `422` invalid payload rejection, and `200 OK` availability after stopping and restarting `hw5-stock-api`).

---

## Production Deployment Considerations

Before a real deployment:
1. **Security:** Add some sort of authentication to use the app. Right now, a user could easily spam the server and cause it to crash, so there needs to be some kind of rate limiter
2. **Monitoring:** More tests can be added to ensure that the features have even more checks on their range and that the prediction is scalable. 
3. As mentioned in the videos in lecture, one of the things when facing real deployment is being able to make changes. All changes would have to be made with the app code, so you would have to constantly rewrite. 
4. **Scalability:** Right now the app runs for one stock, predicting the forward return as -0.136% for that single observation.In order to truly accomplish something, the app should be edited to predict multiple stocks at once to actually inform a decision instead of giving information one stock at a time.  
