# HW5: Serving my HW4 model (Vasileios Dimitropoulos)

This FastAPI service serves **the model from my HW4 experiment**: `Ridge(alpha=1000)` on same-date percentile ranks of 28 features plus sector, trained on HW4's rank target. It is packaged in a Docker image.

- **Model version:** `hw4-ridge-a1000-rank-fit-2026-07-06`. This is the **last quarterly walk-forward refit of my HW4 test** (fit date 2026-07-06). It was trained on all 1,410,101 rows from 2010-01-04 to 2026-03-30 whose 90-day outcome had ended before the fit date (last label end 2026-06-29). HW4 used this fit to score every weekly signal from 2026-07-06 to 2026-09-28.
- **Exact reproduction:** before saving, the export script compares the pipeline with my HW4 code. The Ridge coefficients and all 24,865 scores from this fit are identical (max |difference| = 0.0) to the HW4 code's output and to the scores my HW4 notebook recorded in `outputs/test_predictions.parquet`.
- **Artifact:** `artifacts/model.joblib`, 2 KB, committed. It is a `sklearn.pipeline.Pipeline` made of my preprocessing transformer and the fitted Ridge. Metadata, library versions and the file's SHA-256 are in `artifacts/model_metadata.json`.

## Contents

| Path | What it is |
|---|---|
| `app.py` | HTTP layer: `InputData`/`OutputData`, lifespan loading, `POST /predict`, `GET /health`, 422/500 handling |
| `model_pipeline.py` | Reusable model logic: feature contract, the custom transformer `RankFeatureDesign`, artifact loading with hash and version checks, and the predict helper |
| `artifacts/model.joblib`, `artifacts/model_metadata.json` | The fitted HW4 pipeline and its metadata |
| `scripts/export_hw4_model.py` | One-off export: rebuilds the HW4 fit with the HW4 code, checks it, and writes the artifact and reference cases |
| `tests/test_app.py` | The four required test groups (valid, validation, inference failure, performance) |
| `tests/test_model_pipeline.py` | Unit tests of the transformer and of the artifact safety checks |
| `tests/hw4_reference_predictions.json` | Three inputs with the scores recorded by my HW4 code (the known-result references) |
| `example_request.json`, `example_invalid_request.json` | Valid request (AAPL, 2026-09-28) and an invalid one |
| `scripts/measure_latency.py` | Real-HTTP latency: median and p95 over a kept-alive connection |
| `scripts/demo_requests.sh`, `scripts/container_demo.sh` | curl evidence; build, run, stop and restart of the container |
| `evidence/` | Saved test, lint, latency and request transcripts |
| `Dockerfile`, `.dockerignore`, `Makefile`, `requirements.txt`, `pyproject.toml` | Build, tooling and pinned versions |

## 1. API contract

### Why the inputs are ranks

HW4 preprocessing has two stages:

1. **Same-date cross-sectional ranking** (`hw4lib.model.add_model_columns`). Each raw feature becomes its percentile rank among the 1,100 to 1,900 eligible stocks on that signal date (`groupby("date").rank(pct=True)`). This step needs the whole cross-section and learns nothing across time: a single stock's rank cannot be computed from that stock alone. So it stays in the upstream HW4 feature job, and the API receives the ranked `r_*` values. **Those `r_*` columns are exactly what the HW4 model was trained on**, so the training representation is preserved.
2. **The model design** (`RidgeRanker._design` in HW4). A missing rank becomes 0.5, the median. Ranks are centred by subtracting 0.5, each feature gets a 0/1 missing flag, and sector becomes 12 one-hot columns. The fitted Ridge then applies. My transformer `model_pipeline.RankFeatureDesign` reproduces `_design` exactly and is serialized in the same `Pipeline` as the Ridge, so this preprocessing always travels with the model.

### Input: `InputData` (JSON object, named fields, 29 keys)

All 29 keys are **required**, in this order (the training column order). The application checks at import time that `InputData`'s field order equals the artifact's `feature_names_in_`.

| # | Field | Underlying raw HW4 feature (units of the raw value) |
|---|---|---|
| 1-4 | `r_mom_5`, `r_mom_21`, `r_mom_63`, `r_mom_126` | Adjusted-close return over the last 5/21/63/126 sessions (fraction) |
| 5 | `r_mom_252_21` | 12-1 momentum: return from t-252 to t-21 sessions |
| 6-7 | `r_vol_63`, `r_vol_252` | Annualized volatility of daily returns over 63/252 sessions |
| 8 | `r_max_ret_21` | Largest daily return over 21 sessions |
| 9 | `r_drawdown_252` | Close / 252-session high − 1 |
| 10 | `r_beta_252` | 252-session beta to the S&P 500 |
| 11 | `r_log_dollar_volume` | log of 63-session average dollar volume (USD) |
| 12-16 | `r_gp_to_assets`, `r_roa`, `r_roe`, `r_op_margin`, `r_accruals` | TTM gross profit/assets, net income/assets, net income/equity, operating margin, (net income − CFO)/assets |
| 17-19 | `r_fcf_to_assets`, `r_debt_to_assets`, `r_cash_to_assets` | Ratios to total assets |
| 20-22 | `r_asset_growth`, `r_sales_growth`, `r_net_buyback` | YoY growth (fraction); TTM (repurchases − issuance)/assets |
| 23 | `r_log_mcap` | log(point-in-time shares × close), USD |
| 24-27 | `r_earnings_yield`, `r_book_to_price`, `r_sales_to_price`, `r_fcf_yield` | TTM earnings, book equity, TTM revenue, TTM FCF, each over market cap |
| 28 | `r_filing_age_days` | Calendar days since the latest usable filing |
| 29 | `sector` | **Categorical** string, one of: `Basic Materials`, `Communication Services`, `Consumer Cyclical`, `Consumer Defensive`, `Energy`, `Financial Services`, `Healthcare`, `Industrials`, `Real Estate`, `Technology`, `Utilities`, `Unknown` |

**Constraints on the 28 rank fields, and why:**

- **Type:** a JSON number or `null`. Validation is strict (`strict=True`): a numeric string such as `"0.67"` and booleans such as `true` are rejected with 422, never coerced to numbers. A JSON integer such as `1` is the same number as `1.0` and is accepted.
- **Range (0, 1]:** `rank(pct=True)` gives k/n for the k-th smallest of n stocks. The observed minimum in HW4 was 1/1929 ≈ 0.0005 and the maximum was exactly 1. Zero, negatives, and anything above 1 are not percentiles. For example, a raw 12% return sent as `12.0` would be a units error, so it gets 422.
- **Finite:** `NaN`, `Infinity`, `-Infinity` and overflowing values such as `1e999` are rejected (`allow_inf_nan=False`). Python's JSON parser accepts those tokens, so this check matters.
- **`null` is allowed but the key is required.** `null` means the raw feature was unavailable for that stock on that date, for example no usable filing in the last 550 days. Up to 9.5% of training rows had this, and HW4 trained on it: the model imputes 0.5 and sets that feature's missing flag. That differs from sending 0.5, and a test shows the gap equals the missing-flag coefficient. Leaving a key out is a 422.
- **Sector:** must be one of the 12 exact labels. Case matters, and `null` or numbers are rejected. `Unknown` is HW4's label for a symbol without a sector; it never occurred in the training rows, so its coefficient is 0.
- **No extra keys** (`extra="forbid"`): an unknown field or a 29th feature is a 422.
- **Feature count:** with named fields, sending the wrong number of features means a missing key (422 "Field required") or an extra key (422 "Extra inputs are not permitted"). A positional list, of any length, is also rejected, so the model never sees a wrongly sized matrix.

### Output: `OutputData`

```json
{"prediction": 0.04108925851957752, "model_version": "hw4-ridge-a1000-rank-fit-2026-07-06"}
```

- `prediction` (finite float): the forecast of the stock's **same-date percentile rank of its 90-calendar-day forward return, minus 0.5**. It is dimensionless; training targets lie in −0.5 to +0.5. The scores this fit produced ranged from −0.11 to +0.07 (sd 0.022). Higher means a better expected return relative to the other eligible stocks that day. It is a ranking score, not a return in percent. HW4 bought the 10 highest scores each week.
- `model_version`: identifies the artifact; it also appears in `GET /health`.
- Each 200 or 500 response carries an `X-Request-ID` header, which is also written to the server log.

### Status codes

| Code | When | Body |
|---|---|---|
| 200 | Valid input, successful inference | `OutputData` |
| 422 | Missing or extra fields, wrong types (strings, booleans, lists), out-of-range or non-finite numbers, unknown sector, malformed JSON | `{"detail": [{"type", "loc", "msg"}, ...]}`. The rejected values are **not echoed back** (see below) |
| 500 | Any failure inside inference: the model raises, returns a non-finite value, or returns the wrong shape | `{"detail": "Prediction unavailable"}`. The full traceback is logged server-side with the request ID; the client never receives a default prediction |

**A bug found while testing:** FastAPI's default 422 handler copies each rejected value into the response. For a request containing `NaN`, building that JSON response failed, so the client received a **500 instead of a 422**. `app.py` therefore registers its own `RequestValidationError` handler that returns only each error's type, location and message. The tests send `NaN`, `Infinity`, `-Infinity` and `1e999`.

## 2. Setup and commands

Python 3.12. From this folder:

```sh
# macOS / Linux
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# or, with uv:  uv venv .venv --python 3.12; uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

`requirements.txt` pins the model stack to **my HW4 training environment**: scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, on Python 3.12.9. The service and tooling packages keep the template versions.

| Task | With `make` | Without `make` (Windows here has no `make`) |
|---|---|---|
| Tests | `make test` | `python -m pytest -q` |
| Lint | `make lint` | `python -m ruff check .` and `python -m ruff format --check .` |
| Format | `make format` | `python -m ruff format .` |
| Run locally | `make serve` | `python -m uvicorn app:app --host 127.0.0.1 --port 8000`, docs at http://127.0.0.1:8000/docs |
| HTTP latency | `make latency` | `python scripts/measure_latency.py --url http://127.0.0.1:8000` |
| Build image | `make build` | `docker build -t mfe-hw5:local .` |
| Run image | `make run` | `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` |

Example request (Git Bash, macOS or Linux; in PowerShell use `curl.exe`):

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

## 3. The artifact: how it was made and how it is loaded

**Creation (offline, once).** `scripts/export_hw4_model.py` is run with the **HW4 virtual environment**, so the artifact is saved by the same library versions it was trained with:

```sh
../../HW4/Vasileios_Dimitropoulos/HW4/Scripts/python.exe scripts/export_hw4_model.py \
    --hw4-dir <folder with my HW4 hw4lib/> --outputs-dir ../../HW4/Vasileios_Dimitropoulos/outputs
```

The HW4 code is in my HW4 submission (branch `hw_4_Vasileios_Dimitropoulos`, folder `Homeworks/HW4/Vasileios_Dimitropoulos/`). One way to get it next to this branch is `git worktree add ../../../../hw4-code hw_4_Vasileios_Dimitropoulos`, which creates a `hw4-code` folder beside the repository folder, and then pass `--hw4-dir ../../../../hw4-code/Homeworks/HW4/Vasileios_Dimitropoulos`. The script uses my HW4 code unchanged (`hw4lib.model.training_rows`, `RidgeRanker`) and HW4's cached feature panel, `outputs/panel.parquet`, which my HW4 notebook built from the bCourses database. The script does not open the database. Ridge has a closed-form solution, so fitting `Pipeline([RankFeatureDesign(), Ridge(alpha=1000)])` on the same 1,410,101 rows reproduces the HW4 fit. The script **asserts** that coefficients and scores match the HW4 code and the HW4-recorded predictions within 1e-9; the measured differences were exactly 0. It then writes the artifact, the metadata, `example_request.json` and the three reference cases. Re-running it is only needed to change the model. The panel and database are git-ignored, so a reviewer does not need them: **the service and the tests use only the committed 2 KB artifact.**

**Loading (each server process, at startup).** The lifespan function calls `model_pipeline.load_bundle()`, which:

1. **checks the file's SHA-256 against `model_metadata.json` before unpickling.** joblib/pickle can execute code while loading, so only this exact trusted file is loaded;
2. **refuses to start if the installed scikit-learn differs from the version that saved it** (loading across versions is unsupported);
3. checks that the artifact's feature order equals the API contract;
4. stores the pipeline and version on `app.state`.

The service never fits, refits or queries a database. Paths are resolved relative to `model_pipeline.py`, not the working directory. `RankFeatureDesign` lives in `model_pipeline.py`, which the pickle references by import path, so the module is copied into the Docker image.

## 4. Tests (Milestone 2)

`python -m pytest -q` → **46 passed** (full output: `evidence/test_and_lint.txt`).

1. **Valid prediction** (`test_predict_valid_input` and related tests). These use the real saved model. Each test asserts 200, an exact response schema of `{prediction, model_version}`, a finite float, the right version, an `X-Request-ID`, and exactly one model call. It then compares the API result with:
   - a **direct `pipeline.predict` call** on the same one-row DataFrame;
   - the **score recorded by my HW4 code** before HW5 existed. The source is `outputs/test_predictions.parquet`, written by cell 18 of `hw4_strategy.ipynb`, the quarterly walk-forward of `RidgeRanker("y_rank", 1000)` on the full database. There are three cases from 2026-09-28: AAPL (all features), ALMR (17 of 28 features null, testing the missing-value path) and NVDA (HW4's top score that week).

   **Tolerance: abs 1e-9.** Same model, same float64 arithmetic: the API returns 0.04108925851957752 and HW4 recorded 0.041089258519577516. They differ by one unit in the last place (~1e-17), because a 1-row and a 24,865-row matrix product sum in a different order. 1e-9 allows for other BLAS builds. It is still 45,000× smaller than the smallest gap between the 10th and 11th scores (the buy cutoff) in any week this fit scored (4.5e-5), so it cannot change which stocks HW4 would buy.

   Other tests in this group: `null` vs `0.5` differ by exactly the missing-flag coefficient; JSON `1` equals `1.0`; the contract matches the artifact; `/health` reports the loaded model.
2. **Input validation:** 27 parametrized cases. They cover:
   - missing feature (27 of 28), missing sector, extra field, extra 29th feature;
   - numeric string, `true`/`false`, list and object values;
   - ranks of 0, −0.1 and 1.5, and a raw return (12.0) sent instead of a rank;
   - unknown, wrong-case, numeric and null sector;
   - empty object, positional list bodies of the right and wrong length;
   - `NaN`, `Infinity`, `-Infinity`, `1e999`;
   - empty, non-JSON and truncated bodies.

   Each case asserts **422** and, through a spy wrapped around the loaded pipeline's `predict`, **that the model was never called**.
3. **Inference failure:** valid input, with the loaded pipeline's `predict` replaced (monkeypatch) by a function that (a) raises, (b) returns NaN, or (c) returns two values for one row. Each case asserts **500**, no `prediction` in the body, no internal error message leaked, and an `inference_failed request_id=...` record with a traceback in the log, matching the response's `X-Request-ID`. Mocking is used only here; the positive tests use the real model.
4. **Performance:** see below. The test asserts p95 < 250 ms.

`tests/test_model_pipeline.py` adds:
- a hand check of the design matrix (centring, missing flags, one-hot);
- rejection of reordered columns and of an unknown sector;
- batch independence (a stock's score is the same alone or in a batch);
- refusal of a tampered artifact (hash) and of a mismatched scikit-learn version.

## 5. Latency report

All numbers come from **sequential single-client requests**, each timed with `time.perf_counter()` after warm-up, sending `example_request.json`. Hardware: a Windows 11 laptop with an Intel Core CPU (Family 6 Model 189), 8 logical CPUs and 32 GB RAM, running Python 3.12.9.

| Measurement | Method | Requests | Median | p95 |
|---|---|---|---|---|
| In-process (`tests/test_app.py::test_predict_performance`) | `TestClient`, **no network**: ASGI + validation + inference | 50 after 5 warm-up | 4.5 ms | 5.8 ms |
| Local uvicorn (`evidence/local_uvicorn_transcript.txt`) | **Real localhost HTTP**, httpx kept-alive connection | 200 after 10 warm-up | 4.2 ms | 5.5 ms |
| Docker container | Real HTTP through Docker port forwarding | — | **not measured** (Docker not installed; see section 7) | |

Most of the ~4 ms is building the one-row pandas DataFrame and the transformer's pandas operations, not the 68-coefficient dot product. **Budget:** the test asserts p95 < 250 ms. That is more than 40× the measured value, so slower CI runners or Docker-on-VM won't fail it, but it does fail if per-request work creeps in, such as reloading or refitting the model (the training panel alone has 1.4M rows). HW4 trades once a week after the close, so a few milliseconds is far below any real requirement.

## 6. Docker (Milestone 3)

The Dockerfile uses `python:3.12-slim` with `WORKDIR /app`. It installs the pinned `requirements.txt` first, a separate layer that is cached when only code changes. It then copies only `app.py`, `model_pipeline.py` (the custom transformer module) and `artifacts/`, and runs uvicorn on port 8000 as an unprivileged user. `.dockerignore` keeps `.venv/`, `.git/`, caches, `.env*`, databases (`*.duckdb`, `*.db`), parquet files, tests, scripts and evidence out of the build context.

To reproduce, with Docker Desktop or Docker Engine running and host port 8000 free:

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
# other terminal:
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

Everything the image needs is in this folder; no database, network access or credentials are needed at build or run time beyond PyPI. If port 8000 is taken, use `-p 127.0.0.1:8001:8000` and the matching URL.

## 7. Evidence (Milestone 5)

> **Limitation: the container has not been built or run yet.** Docker is not installed on the machine I used, and I could not install it before the deadline. So there is **no container evidence** in this submission: no build log, no request to the container, and no stop/restart demonstration. The Dockerfile is complete but **untested**. Everything else, including the same requests against the same app under local uvicorn, is shown below. To produce the missing evidence on a machine with Docker running and port 8000 free, run:
>
> ```sh
> sh scripts/container_demo.sh .venv/Scripts/python.exe 2>&1 | tee evidence/container_transcript.txt   # .venv/bin/python on macOS/Linux
> ```
>
> The script builds the image, sends the valid (200) and invalid (422) requests to the container, measures latency through Docker, runs `docker stop`, **restarts the same image**, sends the valid request again, and prints the container logs.

- `evidence/test_and_lint.txt`: `pytest -q` (46 passed, including the latency line), `ruff check`, `ruff format --check`.
- `evidence/local_uvicorn_transcript.txt`: real HTTP to local uvicorn. It shows `GET /health` → 200; `POST /predict` with `example_request.json` → **200** `{"prediction":0.04108925851957752,"model_version":"hw4-ridge-a1000-rank-fit-2026-07-06"}`; `example_invalid_request.json` → **422** listing all three problems (numeric string, missing `r_roe`, extra `symbol`); and the HTTP latency.
- `evidence/container_transcript.txt`: **not yet produced** (see the limitation above).

## 8. What would need attention before a real deployment

- **The ranking step must be served too.** The API expects ranks computed against the whole eligible universe on the same date. In production, a scheduled job would compute the HW4 features and ranks for all ~1,900 stocks and call a **batch** endpoint, or the service would own that job. Sending a stock's rank computed against a different universe or date silently gives a wrong score. Version that feature job together with the model.
- **Model freshness and monitoring.** HW4 refits quarterly; this artifact is the 2026-07-06 fit. Real use needs a scheduled retrain-and-export, a registry of versions with their validation results, and monitoring of input drift (rank and missing-value rates by feature), score distribution and realized IC. The HW4 signal was weak (IC ≈ 0.02) and the strategy lagged the S&P, so a deployment decision would need much more than a working API.
- **Artifact trust and storage.** Pickle is code: keep artifacts in access-controlled storage with signed hashes, as the SHA-256 check here does in miniature, or use a non-executable format (e.g. ONNX or the coefficients as JSON) for a linear model.
- **Security and operations.** Add authentication, TLS, rate limits and request-size limits. Run several workers or replicas behind a load balancer with readiness/liveness probes (`/health`). Use structured logs with request IDs, metrics and alerting. Pin the base image by digest, use a lock file with hashes, and run dependency and image scans. Run CI on every change (tests, lint, image build).
- **Data quality upstream.** HW4 found broken split adjustments, implausible share counts and acquired stocks with no delisting proceeds. The API checks only the format of its inputs, not whether the upstream data was right.

## 9. Sources and AI assistance

- **Adapted from the Lesson 9 demo** (`Lectures/Lecture 9/demo/app.py`, `train_model.py`, `Dockerfile`, `tests/`): lifespan loading onto `app.state`, `extra="forbid"`/`strict`/`allow_inf_nan=False` validation, `X-Request-ID` plus logged 500s, the spy pattern for "invalid input never reaches the model", and the Dockerfile layout. FastAPI and Starlette docs on lifespan in tests and `raise_server_exceptions=False`.
- **My HW4 code** (`hw4lib/model.py`: `RidgeRanker`, `training_rows`, `SECTORS`; `hw4lib/features.py`: `NUMERIC_FEATURES`) is used unchanged by the export script, and `RankFeatureDesign` re-implements `RidgeRanker._design`.
- **Written for this homework:** the export with equivalence checks, the transformer, the hash and version-checked loader, the input contract, the sanitized 422 handler, the tests, the latency script and the container demo.
- **AI assistance:** developed with an AI coding assistant (Claude Code) working under my direction. I reviewed the design decisions (serving ranks rather than raw features, the null policy, the tolerance, the error contract) and can explain every part.
