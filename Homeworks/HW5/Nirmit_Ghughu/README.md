# HW5: Serving my HW4 model with FastAPI and Docker (Nirmit Ghughu)

This service serves the **Ridge stock-ranking model from my HW4** (branch `hw_4_Nirmit_Ghughu`, `Homeworks/HW4/Nirmit_Ghughu/`) through `POST /predict`, packaged as a Docker image.

| | |
|---|---|
| Model version | `hw4-ridge-rank-2026-07-02`: the last quarterly refit from my HW4 final test. Trained on 636,539 stock-weeks (signals 2010-01-08 to 2026-03-27) using only 90-day outcomes that finished by 2026-06-29. Details in [artifacts/model_metadata.json](artifacts/model_metadata.json). |
| Artifact | [artifacts/hw4_ridge.joblib](artifacts/hw4_ridge.joblib) (8.8 KB, committed). It holds an sklearn `Pipeline(HW4RidgeDesign → Ridge)` plus the version string. |
| Proof it is the HW4 model | Before saving, [export_model.py](export_model.py) re-ran HW4's own code and compared the result with the scores HW4 recorded during its final run: **24,742 predictions agree to within 3.7e-9**. |
| Tests | 30 passed; `ruff check` and `ruff format --check` clean ([evidence/test_and_lint_output.txt](evidence/test_and_lint_output.txt)) |
| Container demo | Valid request returns 200; invalid and NaN requests return 422; after a stop and restart the same request returns the same 200 ([evidence/container_transcript.txt](evidence/container_transcript.txt)) |

## Input/output contract

**What the model predicts.** For one stock on one weekly signal date, it predicts the stock's **within-date percentile rank of its 90-calendar-day return, minus 0.5**. Entry is at the next session's close, and exit at the first close on or after entry + 90 days. Higher means the stock is expected to beat its peers on that date. The prediction is a ranking score, not a dollar return; in HW4 it is used to pick the top 25 stocks each week.

**Why the inputs are ranks.** HW4 preprocessing has two stages:

1. **Per-date percentile ranking:** each raw feature becomes `rank(pct=True) − 0.5` across all tradable stocks on that date. This needs the whole day's cross-section, so a single request can't compute it. The HW4 data pipeline does it upstream, and the API receives the ranked values: the same representation the Ridge model was trained on. This stage has no fitted state.
2. **Ridge design matrix** (custom transformer `HW4RidgeDesign` in [hw4_model.py](hw4_model.py), which reproduces HW4's `model._design`): a missing rank becomes 0 (the median) plus a `<feature>_missing` indicator, and sectors become one-hot dummies. The **sector list is learned at fit time** and stored in the artifact.

**Request body:** a JSON object with exactly these **23 required fields**, in training order. Extra fields are rejected.

| Field | Type | Allowed values | Raw feature before ranking (units) |
|---|---|---|---|
| `mom_5`, `mom_21`, `mom_63`, `mom_126` | number or `null` | −0.5 to 0.5 | total return over 5/21/63/126 sessions (fraction) |
| `mom_252_21` | number or `null` | −0.5 to 0.5 | return from t−252 to t−21 sessions (fraction) |
| `vol_63` | number or `null` | −0.5 to 0.5 | 63-session annualized volatility (fraction) |
| `max_ret_21` | number or `null` | −0.5 to 0.5 | largest daily return in 21 sessions (fraction) |
| `drawdown_252` | number or `null` | −0.5 to 0.5 | close / 252-session high − 1 (fraction) |
| `beta_252` | number or `null` | −0.5 to 0.5 | 252-session beta to ^SP500TR (unitless) |
| `log_dollar_vol_63` | number or `null` | −0.5 to 0.5 | log of 63-session average close × volume (log $) |
| `gross_profitability`, `operating_roa`, `debt_to_assets`, `cash_to_assets`, `fcf_to_assets`, `accruals_to_assets` | number or `null` | −0.5 to 0.5 | statement item / total assets (ratio) |
| `net_margin` | number or `null` | −0.5 to 0.5 | net income / revenue (ratio) |
| `asset_growth` | number or `null` | −0.5 to 0.5 | YoY total-asset growth (fraction) |
| `earnings_yield`, `book_to_market` | number or `null` | −0.5 to 0.5 | 4 × quarterly net income or equity / capitalization proxy |
| `filing_age_days` | number or `null` | −0.5 to 0.5 | days since the latest usable filing |
| `log_mcap` | number or `null` | −0.5 to 0.5 | log capitalization proxy (log $) |
| `sector` | string | one of the 11 fitted sectors: `Basic Materials`, `Communication Services`, `Consumer Cyclical`, `Consumer Defensive`, `Energy`, `Financial Services`, `Healthcare`, `Industrials`, `Real Estate`, `Technology`, `Utilities` | snapshot sector label |

**Why these constraints:**
- **Range:** a percentile rank minus 0.5 can only fall between −0.5 and 0.5. A value outside that range (for example a raw, unranked `log_mcap` of 23.4) means the caller skipped the ranking step, so it is rejected rather than silently extrapolated.
- **`null`:** HW4 trained with missing values, for example fundamentals when no recent filing exists, or long lookbacks for recently listed stocks. So `null` is accepted *explicitly*, but the key itself is still required.
- **Strict types:** `"0.1"` and `true` are rejected. JSON integers such as `0` are accepted as numbers.
- **Non-finite values:** `NaN` and `±Infinity` are rejected.
- **Sector:** sectors the model never saw (e.g. `"Unknown"`) are rejected.

**Response 200:** `{"prediction": <finite float>, "model_version": "hw4-ridge-rank-2026-07-02"}`, plus an `X-Request-ID` header.

**Errors:**
- **422:** any contract violation. The body lists `type`, `loc` and `msg` for each error, but not the raw input. The default FastAPI handler echoes the bad value back, and since `NaN` can't be written as JSON, it turned correct rejections into 500s. My tests caught this and the custom handler fixes it.
- **500:** `{"detail": "Prediction unavailable"}` plus `X-Request-ID` when inference fails or returns a non-finite number. The traceback is logged on the server with the same request ID.

**Other endpoints:** `GET /health` returns the status and loaded model version. Interactive docs are at `/docs`.

**Example:** [example_request.json](example_request.json) is UPBD on 2026-09-25, HW4's top-ranked stock that week. HW4 recorded a score of 0.05188995, and the API returns 0.0518899493.

## Setup and reproducing the results

Python 3.12, in a separate environment from the course `.venv`. Run from this folder:

```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt   # sklearn 1.8.0 / numpy 2.3.5 / pandas 2.3.3, same as HW4
make test        # python -m pytest -q        (add -s to print the latency line)
make lint        # ruff check + ruff format --check
python -m uvicorn app:app --host 127.0.0.1 --port 8000     # local server, docs at /docs
```

**Docker** (I used Colima on macOS; Docker Desktop works the same way):

```sh
make build       # docker build -t mfe-hw5:local .
make run         # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
python latency_http.py          # real-HTTP latency against the running container
```

The image (`python:3.12-slim`, about 740 MB) contains only `requirements.txt`, `app.py`, `hw4_model.py` and `artifacts/`, and runs as a non-root user. `.dockerignore` keeps out `.venv`, tests, evidence, databases and `.env` files.

**Artifact loading:** `app.py` loads `artifacts/hw4_ridge.joblib` once per process, in the FastAPI lifespan, using a path relative to `app.py`. Unpickling needs the importable `hw4_model.py` module and the same scikit-learn version. The service never retrains and never opens the HW4 database.

**Re-creating the artifact (optional, not needed to serve).** This needs the HW4 code and `equities_top2000.duckdb`:

```sh
HW4_DIR=<repo>/Homeworks/HW4/Nirmit_Ghughu python export_model.py
```

It rebuilds the HW4 panel, refits the 2026-07-02 model exactly as HW4's walk-forward did, asserts that it reproduces HW4's recorded scores, and then writes `artifacts/` and [tests/reference_examples.json](tests/reference_examples.json).

## Tests ([tests/test_app.py](tests/test_app.py))

| Group | What it checks |
|---|---|
| 1. Valid prediction | Uses the real saved model on 3 HW4 reference stocks: the top pick, a stock with `null` features, and a typical stock. Checks status 200, that the response has exactly `prediction` and `model_version`, and that the prediction is finite. The API result must equal a direct `pipeline.predict` call (within 1e-12) **and** the score HW4 recorded before the model was wrapped (within 1e-6; HW4's CSV keeps about 8 significant digits). Also checks that ints are accepted and that the model is loaded once per process. |
| 2. Input validation | 17 parametrized cases: missing field or sector, extra field, a 23rd feature, numeric string, boolean, out-of-range value, raw unranked value, unknown sector, numeric or null sector, list instead of number, fixed-width lists of 21 and 22 values, empty object, top-level list. Plus NaN, ±Infinity and malformed JSON. **Each test replaces `predict` with a function that fails the test if called**, which proves invalid input never reaches inference. |
| 3. Inference failure | Valid input with `predict` patched to raise returns a 500 with a generic body and no leaked internal message, and the request ID appears in the server log. A non-finite model output also returns 500. |
| 4. Performance | 10 warm-up requests, then 200 timed with `time.perf_counter`. The test asserts p95 < 100 ms (justified below). |

**Latency report** (Apple M3, 8 GB, macOS, Python 3.12.6):

| Measurement | Median | p95 | Max |
|---|---:|---:|---:|
| In-process `TestClient` (no socket), 200 requests | 2.8 ms | 3.0 ms | 3.4 ms |
| Real localhost HTTP to the Docker container (`latency_http.py`, one keep-alive connection, Colima VM), 200 requests | 3.1 ms | 3.4 ms | 26 ms (one outlier) |

Almost all of the time goes to building the one-row DataFrame and the sklearn pipeline call; Ridge itself is a single dot product. The 100 ms p95 budget is about 30× the observed in-process latency. That leaves room for slower CI machines while still catching gross regressions, such as accidentally refitting on a request, querying a database, or adding a slow per-request dependency. It would *not* catch a per-request reload of this tiny artifact, which I measured at only 0.12 ms; `test_model_loaded_once_per_process` covers that instead.

## What would need attention before a real deployment

- **Feature pipeline as a service:** ranking needs the full cross-section of stocks for the date. In production the HW4 feature job should compute each day's ranks in one place and send them, rather than callers building them, to avoid training/serving skew. A batch `POST /predict` for the whole day's universe would fit the use case better than single-stock calls.
- **Model lifecycle:** HW4 refits every quarter. That needs a registry with versioned artifacts and checksums, and promotion only after the "matches research scores" check that `export_model.py` already runs. The current model was trained on outcomes up to June 2026 and will go stale.
- **Data quality:** HW4 found unadjusted splits and quotes that stop after takeovers. Upstream checks should block bad features. Input monitoring should track the share of `null`s and the distribution of each rank, and output monitoring should track the distribution of daily scores.
- **Security and operations:**
  - Never load untrusted pickles.
  - Add authentication and rate limits.
  - Pin the base image by digest and scan it.
  - Add a container health check, structured JSON logs with request IDs, metrics, and multiple workers or replicas behind a load balancer.
  - Make startup fail loudly if the artifact or library version doesn't match.
- **Size:** the image is about 740 MB because of scipy, sklearn and pandas. A multi-stage build or a slimmer runtime (e.g. exporting the Ridge coefficients) would shrink it.

## Files

| File | Purpose |
|---|---|
| [app.py](app.py) | FastAPI app: contract, lifespan loading, `/predict`, `/health`, safe 422 handler |
| [hw4_model.py](hw4_model.py) | Feature names and order, plus the `HW4RidgeDesign` custom transformer the artifact needs |
| [export_model.py](export_model.py) | One-time export from HW4 with a reproduction check |
| [artifacts/](artifacts/) | `hw4_ridge.joblib` and `model_metadata.json` |
| [tests/](tests/) | `test_app.py` and `reference_examples.json` (inputs plus HW4-recorded scores) |
| [example_request.json](example_request.json) | Valid request used in the demo |
| [latency_http.py](latency_http.py) | Real-HTTP latency measurement |
| [evidence/](evidence/) | Container transcript, real-HTTP latency, test/lint output |
| `Dockerfile`, `Makefile`, `requirements.txt`, `pyproject.toml`, `.dockerignore`, `.gitignore` | From the template, completed |

## Reused code and AI assistance

- **Adapted code:** the app structure (lifespan loading, strict Pydantic model, request ID on 500s) is adapted from the Lesson 9 demo, `Lectures/Lecture 9/demo/app.py`. Its invalid-input test pattern (patching `predict` to fail) is adapted from `Lectures/Lecture 9/demo/tests/test_app.py`. The model, features and preprocessing come from my HW4.
- **AI assistance:** developed with an AI coding assistant (Claude Code). At my request it wrote the code, ran the tests, the Docker build and the container demo, and drafted this README; I reviewed the work before submitting.
