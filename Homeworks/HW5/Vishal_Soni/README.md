# HW5: HW4 prediction API

This service exposes the fitted HW4 `StandardScaler` → `Ridge(alpha=100)` pipeline as `POST /predict`. It loads `model.joblib` once at server startup; it does not train or query the database when serving requests. The training and scoring conventions are those in HW4 commit `f38e6db`, [hw4.py](../../HW4/Vishal_Soni/hw4.py) (the source is on the `hw_4_vishal_soni` branch if that submission has not yet been merged).

## Model provenance and limitations

HW4 did not save its fitted pipeline, so it was reproduced in the original Python 3.14 HW4 environment by running `load_samples` from commit `f38e6db`, then fitting the same scaler and ridge estimator on the same 13,585 rows with signal dates from 2010 and labels ending by December 31, 2015. The reproduced model's predictions agreed with `fit_and_predict` on **every** HW4 test-period row to absolute tolerance `1e-12`. The scaler and estimator were serialized together with joblib 1.5.3. The relevant training versions are NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, SciPy 1.16.3, joblib 1.5.3 and threadpoolctl 3.6.0; the separate Python 3.12 service and Dockerfile pin the same versions. The checked-in artifact's SHA-256 is `0381abe0ae209592c26200d257cb7ec5054b20fb9a8ea218bc1a123742d385d9`.

`example_request.json` is the actual AAL signal from March 31, 2016, for April 1 entry. `reference_prediction.json` records its features and the **HW4 scoring result before the API was written**; the tests compare both to that reference and a direct call on the saved model. The return of `8.540702680292116` is a decimal return prediction (about **854%**), not a percentage value. It is implausibly large and reflects the severe return outliers and data-quality issues noted in HW4; this API reproduces the model, not a claim that the forecast is credible. HW4's unresolved PCGU price gap also means its full-period backtest was only provisional. This is an educational service, not an investment recommendation.

## Input and output

Provide a JSON object containing exactly these four numeric fields. Returns are decimals, not percentages; no strings, booleans, nulls, extra fields or non-finite values are allowed. `mom21`, `mom63`, `mom252_21` are at least -1 because the underlying adjusted prices are positive; `vol63` is nonnegative. There is no arbitrary upper bound. A list of the wrong width (or any list in place of the object) is rejected.

| Field | Meaning |
| --- | --- |
| `mom21` | Signal-close adjusted price / price 21 sessions earlier − 1 |
| `mom63` | Signal-close adjusted price / price 63 sessions earlier − 1 |
| `mom252_21` | Adjusted price 21 sessions ago / price 252 sessions ago − 1 |
| `vol63` | Sample standard deviation of 63 daily adjusted-price returns × √252 |

All features must be known at the signal close, without filling missing quotes. The model was trained on monthly signal dates for the historically eligible top-200 universe. The prediction represents the next-session entry-close to the first observed close at or after 90 calendar days, with training outcomes censored when no exit quote appeared within five market sessions. The API cannot verify input provenance or perform that data preparation; callers must follow those conventions. Named fields are reordered to the original feature order in a pandas DataFrame before prediction.

Success is HTTP 200 with `{"prediction": <finite number>, "model_version": "hw4-f38e6db"}`. Invalid inputs return 422. Unexpected inference errors or non-finite model output return 500 with a generic message; details are logged server-side. Only load trusted joblib artifacts: pickle-based formats can execute code when loaded.

## Setup, tests, and local server

From this directory, use Python 3.12 and a separate virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make lint
make test
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

If `make` is unavailable, run `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest -q`. Interactive docs are at `http://127.0.0.1:8000/docs`. In another terminal:

```sh
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data '{"mom21":"0.1","mom63":0.1,"mom252_21":0.2,"vol63":0.3}'
```

17 tests passed. With five warmups and 30 successful calls timed by `time.perf_counter` using an in-process FastAPI `TestClient` on an Apple M2 Pro, Python 3.12.9, median latency was **1.269 ms** and nearest-rank p95 was **1.752 ms**. This is not network or container latency and is not a production service-level guarantee. The test emits a Starlette warning about the `httpx` TestClient integration; it does not affect the assertions. See [local HTTP transcript](local_http_transcript.txt) for valid, rejected and post-restart requests to a real local server.

## Docker

From this directory, after installing and starting Docker:

```sh
make build
make run
# In a second terminal:
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data '{"mom21":"0.1","mom63":0.1,"mom252_21":0.2,"vol63":0.3}'
# Stop the container, run `make run` again, and repeat the valid request.
```

Built and tested with Docker on Colima (Linux arm64) on November 16, 2026. The container returned 200 for the example, 422 for an invalid numeric string, and the same prediction after a container restart. See the actual [container transcript](container_transcript.txt); host port 18005 was used for the recorded run. The Dockerfile uses `python:3.12-slim`; `.dockerignore` excludes environments, databases and secrets from the build context.

## Before a real deployment

Investigate the unreliable model, extreme outputs, data lineage, corporate actions and the HW4 missing-price incident; validate against trustworthy data before use. A real service would also need authentication, TLS, rate and request-size limits, monitoring, model registry and artifact integrity checks, reproducible builds and dependency/security updates, plus production latency/load testing. The code adapts the assignment's FastAPI template; OpenAI Codex assisted with implementation and testing.
