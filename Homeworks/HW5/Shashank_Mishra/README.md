# HW5 — Shashank Mishra: serving my HW4 LightGBM model

This service serves the **LightGBM model from my HW4** (`lgb_excess`) behind a FastAPI `POST /predict` endpoint, tests it, and packages it in Docker.

You send it one stock's 12 HW4 feature percentiles. It returns that stock's **predicted 90-day return relative to the average stock** on the same date.

**Which HW4 model, and why this one.** My HW4 *submitted* strategy was 12-1 momentum. Momentum is a fixed rule (sort by the `mom_12_1` percentile) with nothing fitted, so there is no fitted model to serve. The fitted stock-return model in my HW4 experiment is the **LightGBM `lgb_excess`** model, which ran next to momentum in the final test under the same trading rules. HW4 refit it every spring. I serve the **last refit**:

| | |
|---|---|
| `model_version` | `hw4-lgb_excess-fit-2026-04-09` |
| HW4 code | `run_hw4.FITTED["lgb_excess"]` = `hw4.model.RankLGBM(target="excess", num_leaves=7, n_estimators=300, min_child_samples=5000)` |
| Fit date / training rows | 2026-04-09, 1,586,680 stock-weeks from 2010-01-04. Every row's 90-day label had ended before the fit date; the last one ended 2026-04-06 (`hw4.model.training_rows`) |
| HW4 used it for | the 2026-04-09 → 2026-09-30 signals of the final test |
| Same model as HW4? | Yes. Its feature importances match the 2026 column of HW4's `test_lightgbm_feature_importance` in `results/results.csv` to 4e-7, which is the rounding used in that file. |
| Artifact | [`artifacts/model.joblib`](artifacts/model.joblib), 256 KB, committed. Metadata in [`artifacts/model_metadata.json`](artifacts/model_metadata.json) (SHA-256, versions, parameters, training ranges). |

## 1. Input / output contract

`POST /predict` takes a JSON **object with exactly these 12 keys**. The API puts them into the model in this order, which is HW4's `FEATURES` order. The model sees them as columns `r_<name>` in a float64 DataFrame, just like HW4's `RankLGBM._X`.

| # | Field | Percentile of… (HW4 `features.py`) | Null allowed? | Missing in training |
|---|---|---|---|---|
| 1 | `idio_vol_252` | 252-session residual volatility after beta | yes | 2.1% |
| 2 | `earnings_yield` | TTM net income / market cap | yes | 8.8% |
| 3 | `debt_to_assets` | total debt / total assets | yes | 0.4% |
| 4 | `sales_to_price` | TTM revenue / market cap | yes | 8.8% |
| 5 | `gross_profitability` | TTM gross profit / total assets | yes | 8.8% |
| 6 | `earnings_change` | (NI_q − NI_q−4) / total assets | yes | 11.9% |
| 7 | `mom_12_1` | 12-month return skipping the last month | yes | 2.9% |
| 8 | `volume_trend` | log(21-session / 252-session average volume) | yes | 3.6% |
| 9 | `accruals` | (TTM net income − TTM operating cash flow) / total assets | yes | 9.3% |
| 10 | `asset_growth` | total assets / total assets 4 quarters earlier − 1 | yes | 12.0% |
| 11 | `rev_1m` | last 21-session return | **no** | 0% |
| 12 | `beta_252` | 252-session beta to ^SP500TR | yes | 2.1% |

**Units and constraints, and why I chose them:**
* **Every value is a percentile rank in (0, 1].** HW4 never fed raw ratios to the model. Each feature was first ranked with `rank(pct=True)` among that date's eligible top-2000 stocks. Such a rank is always between 1/n (n ≤ ~2,000) and exactly 1, and the training data range was 0.0005–1.0. So `0`, negative values, and anything above 1 cannot come from the HW4 pipeline and are rejected. For example, a raw momentum of `35.0` is rejected instead of being silently treated as a top-ranked stock.
* **The caller computes the percentiles.** A percentile needs the whole cross-section of stocks on that date, which a single request cannot provide. This is how HW4 scored stocks too. HW4 learned no preprocessing from the training data (no scaler or imputer), so the fitted `LGBMRegressor` is the whole pipeline.
* **All 12 keys are required.** `null` is the explicit "missing" marker. It becomes NaN, which LightGBM routes the way it did in training. `null` is allowed only for features that were actually missing in the training rows; `rev_1m` never was, so `null` is rejected for it.
* **Strict types.** A value must be a JSON number. The strings `"0.5"` and `true`/`false` are rejected, not converted. An integer such as `1` is accepted, because it is a valid percentile (the top stock).
* **422 cases:** NaN or ±Infinity, extra keys, a missing key, 11 or 13 features, a list instead of an object, nested objects, and malformed JSON. The 12 named fields with `extra="forbid"` fix the feature count, so a wrong width can never reach LightGBM and turn into a 500. There are no categorical inputs: HW4's `lgb_excess` did not use sector.

**Output (200):**
```json
{"prediction": 0.004700122008907656, "model_version": "hw4-lgb_excess-fit-2026-04-09"}
```
* `prediction` is always finite. It is a fraction: `0.0047` means the model expects this stock to beat the same-date average stock by 0.47 percentage points over 90 days. The 90 days run from entry at the next session's close to the first close on or after entry + 90 calendar days. The HW4 target was winsorized at 1%/99% per date and demeaned. Use it to **rank** stocks; it is not an absolute return forecast.
* Every response has an `X-Request-ID` header.
* **422** returns `{"detail": [{"loc", "msg", "type"}, ...]}`. It deliberately does **not** echo the input back (see §4).
* **500** returns `{"detail": "Prediction unavailable"}` with the `X-Request-ID`. The full traceback goes to the server log under that request ID. A model that returns a non-finite value also gives a 500, never a made-up default.
* `GET /health` → `{"status": "ready", "model_version": ...}`. Interactive docs are at `/docs`, and include an example.

[`example_request.json`](example_request.json) is a real HW4 input: stock **A** (Agilent) on signal date 2026-04-09. HW4's code predicted **0.004700122008907656** for it.

## 2. Setup and reproducing my results

From this folder (`Homeworks/HW5/Shashank_Mishra`), with Python 3.12:

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # macOS/Linux
python -m pip install -r requirements.txt
```
On Windows I had only Python 3.13 installed, so I used uv to get 3.12:

```powershell
uv venv --python 3.12 .venv; .venv\Scripts\Activate.ps1; uv pip install -r requirements.txt
```

`requirements.txt` pins the model stack to the **exact versions of my HW4 training environment**: lightgbm 4.7.0, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, joblib 1.6.0. Pickled models are only supported on the version that wrote them. So `model_service.load_bundle` **refuses to start** if the installed lightgbm or scikit-learn differs from the versions recorded in the artifact, and a test covers this.

| Step | Command (Makefile target) |
|---|---|
| Tests | `python -m pytest -q` (`make test`) |
| Lint + format check | `python -m ruff check .` and `python -m ruff format --check .` (`make lint`) |
| Run locally | `python -m uvicorn app:app --host 127.0.0.1 --port 8000`, then open http://127.0.0.1:8000/docs |
| Build image | `docker build -t mfe-hw5:local .` (`make build`) |
| Run container | `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` (`make run`) |
| Request | `curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json` |
| Real HTTP latency | `python measure_latency.py -n 200` (while the server or container is running) |
| Full container demo | `bash evidence/container_demo.sh` (needs the built image and a free port 8000) |

**Artifact loading.** The service loads only the committed `artifacts/model.joblib`, once per process, in the FastAPI lifespan. Its path is resolved relative to `model_service.py`, so it works from any working directory. The service never retrains and never opens the HW4 database. joblib can execute code, so only load this trusted artifact.

**Regenerating the artifact (optional; this is how I made it).** [`export_hw4_model.py`](export_hw4_model.py) runs **in the HW4 training environment** (the repository-root `.venv`, Python 3.13) with the HW4 code, its cached feature panel, and `equities_top2000.duckdb`:

```bash
python export_hw4_model.py --db ../../../equities_top2000.duckdb   # ~15 s with HW4's cached panel
```

It imports HW4's own functions: `run_hw4.signal_dates`, the refit schedule, `run_hw4.FITTED`, and `hw4.model.training_rows`. With these it re-runs the last refit exactly as `stage_test` did, checks the feature importances against HW4's `results.csv`, and records the reference predictions. It then writes `artifacts/`, `example_request.json`, and `tests/reference_day_2026-04-09.csv`. Re-running it reproduces `model.joblib` byte for byte (same SHA-256). It expects HW4 in `../../HW4/Shashank_Mishra` (my HW4 PR branch); otherwise, pass `--hw4-dir`.

## 3. Tests and latency

[`tests/test_app.py`](tests/test_app.py) has **41 tests, all passing**, and ruff check/format is clean. The full output is in [`evidence/checks.txt`](evidence/checks.txt).

1. **Valid prediction (real saved model).** The test asserts a 200, the exact response schema, a finite float, and the model version. It then checks the API against a **direct call** of the loaded estimator on the same DataFrame, and against **the value HW4's code produced**. The second check is parametrized over 8 real stocks. One of them, ACI, has a missing `accruals` value, so the null → NaN path is checked against HW4 too. A further test sets all 11 nullable features to `null` and checks the API against the direct estimator call.
   * **The whole HW4 cross-section.** All **1,991 eligible stocks on 2026-04-09** (`tests/reference_day_2026-04-09.csv`, with HW4's prediction for each) go through the API. Every one returns 200 and matches HW4 exactly, including the 133 stocks with real missing-value patterns. This also proves that my (0, 1] bounds and null rules accept every real HW4 input.
   * **Where the reference values came from.** `export_hw4_model.py` called HW4's `RankLGBM.predict` on the 2026-04-09 panel rows before any API code was involved. It ran in Python 3.13, and the tests run in Python 3.12, so they also prove the artifact survives that move.
   * **Tolerance.** Absolute 1e-12. LightGBM sums the same leaf values in float64, so anything larger would mean a different model, not rounding. The values matched exactly.
   * **Other checks:** the order of keys in the request does not change the prediction, a row scores the same alone or in a batch, the feature order matches the artifact, the artifact is loaded exactly once per process, `/health` reports the loaded model, and startup is refused on a library-version mismatch.
2. **Input validation → 422.** This covers 13 bad payloads: missing key, 11 or 13 features, extra key, empty object, list, numeric string, boolean, `null` for `rev_1m`, 0, 1.5, negative, raw return `35.0`, and nested object. It also covers NaN, Infinity, −Infinity, and malformed JSON. In each case the model's `predict` is replaced with a function that **fails the test if it is called**, which shows that invalid input never reaches inference. A companion test counts calls on that same patched method: a valid request calls it once, an invalid one zero times. So the guard is on the object the API really calls, and the other tests are not passing vacuously. Two positive edge cases check that `null` for a nullable feature and the integer `1` are accepted.
3. **Inference failure → 500.** The test sends **valid** input with the loaded model's `predict` monkeypatched to raise. It asserts a 500 with the generic body, that the exception text is *not* in the response, and that the server log *does* contain it under the response's `X-Request-ID`. A second test makes the model return NaN, which must give a 500, not a 200.
4. **Performance.** 10 warm-up requests, then 200 sequential requests timed with `time.perf_counter`, reporting the median and p95.

**Latency.** Measured on a 13th Gen Intel Core i7-1355U laptop (10 cores, 16 GB RAM) with Windows 11, Python 3.12.9, and Docker Engine 29.7.2. Each measurement is one stock per request:

| Measurement | Requests | Median | p95 | Max |
|---|---|---|---|---|
| In-process `TestClient` (no network; `pytest`) | 200 after 10 warm-up | 2.52 ms | 3.06 ms | 4.07 ms |
| **Real localhost HTTP to the Docker container** (`measure_latency.py`, one keep-alive httpx connection) | 200 after 10 warm-up | 2.99 ms | 4.04 ms | 5.66 ms |

The test asserts **p95 < 50 ms**. That is more than 10× my measured p95, which leaves room for a slower laptop or CI runner. It still catches gross regressions such as refitting per request (seconds). Reloading the 256 KB artifact takes only ~2 ms, too little for a latency budget to notice, so a separate test counts artifact loads and asserts exactly one per server process. I set LightGBM's prediction threads to 1 at load. This changes only how `predict` runs, not the trees or the predictions. With one row per request, extra threads only add start-up cost and oversubscribe the CPU under concurrent requests.

**Concurrency check (manual).** FastAPI runs `def` endpoints in a thread pool, so several requests can call the same loaded model at once. I ran local uvicorn and sent 800 requests from 64 parallel threads, cycling through the 8 reference stocks. All 800 returned 200 with exactly the HW4 reference prediction (maximum difference 0.0).

## 4. Container and request evidence

* **Dockerfile.** It starts from `python:3.12-slim` and installs `libgomp1`, which LightGBM's Linux wheel needs. The working directory is `/app`. It installs the pinned requirements in their own cached layer, then copies `app.py`, `model_service.py`, and `artifacts/`. The image runs as a non-root user.
* **Build context.** `.dockerignore` keeps `.venv`, databases, parquet files, `.env`, tests, and the evidence out of it.
* **No custom transformer modules are needed.** The artifact is a plain `LGBMRegressor` plus a dict, not HW4's `RankLGBM` class.
* **The image is 176 MB.**

[`evidence/container_transcript.txt`](evidence/container_transcript.txt) is the unedited terminal output of [`evidence/container_demo.sh`](evidence/container_demo.sh). It shows:
1. `docker run`, then `/health`.
2. **A valid request → 200**: `{"prediction":0.004700122008907656,"model_version":"hw4-lgb_excess-fit-2026-04-09"}`, identical to HW4.
3. **An invalid request → 422.** [`evidence/invalid_request.json`](evidence/invalid_request.json) has a numeric string, `rev_1m = 1.5`, and an extra `log_mcap` key, and all three errors are reported.
4. **A NaN request → 422.**
5. The real-HTTP latency run.
6. The container log, showing the model loaded **once** at startup.
7. **`docker stop`, then a new container from the same image**, which returns the same 200 prediction and model version.

**A bug the tests caught.** My first NaN test did not get a 422. FastAPI's default 422 handler echoes the bad input back, and `NaN` cannot be written as JSON, so the error handler itself crashed. I added a `RequestValidationError` handler (`app.py`) that returns the location, message, and type of each error without the input values. This also avoids reflecting client data back.

## 5. What would need attention before a real deployment

* **Feature parity (the biggest risk).** The service trusts the caller's percentiles. In production they must come from the same versioned code that built the HW4 panel. Otherwise training/serving skew gives confident, wrong predictions with no error.
* **Retraining and versioning.** HW4 refit every spring. A real system needs a scheduled retrain, a model registry with checksums (the SHA-256 is already in the metadata), a staged rollout, and a rollback.
* **Monitoring.** Track input drift (for example, null shares against the training shares in the metadata), the prediction distribution, realized rank IC as labels mature, latency, and 4xx/5xx rates. As the Anthropic postmortem from class showed, quality can degrade silently while every request still returns 200.
* **Batch endpoint.** HW4 scores ~2,000 stocks per date, so production would take a batch instead of 2,000 round trips.
* **Security and operations.** Add authentication, rate limits, and TLS. Pin the base image by digest. Add structured logs, worker and resource limits, and readiness probes. Prefer a non-pickle model format, such as LightGBM's text format.
* **The model itself.** Its HW4 test rank IC was only 0.018, and it lost to momentum and the index. Serving it reliably says nothing about whether it should drive trades.

## 6. Files

| File | What it is |
|---|---|
| [`app.py`](app.py) | FastAPI request handling only: the `InputData`/`OutputData` contract, lifespan loading, `/predict`, `/health`, and the 422/500 handling |
| [`model_service.py`](model_service.py) | Model logic with no HTTP: loading and version checks, the training-representation DataFrame, and `predict_one` |
| [`artifacts/`](artifacts/) | `model.joblib` (the fitted HW4 model), `model_metadata.json`, and `reference_predictions.json` (HW4 inputs and outputs) |
| [`export_hw4_model.py`](export_hw4_model.py) | The one-time export from HW4 (run in the HW4 environment) |
| [`tests/test_app.py`](tests/test_app.py), [`tests/reference_day_2026-04-09.csv`](tests/reference_day_2026-04-09.csv) | The 41 tests; the full 2026-04-09 HW4 cross-section with HW4's predictions |
| [`measure_latency.py`](measure_latency.py) | Real-HTTP latency measurement |
| [`example_request.json`](example_request.json) | A valid HW4 input (stock A, 2026-04-09) |
| [`evidence/`](evidence/) | The container transcript and its script, the invalid request, and the test/lint output |
| `Dockerfile`, `.dockerignore`, `Makefile`, `requirements.txt`, `pyproject.toml` | Build, tooling, and pinned versions |

## 7. Sources and AI use

* **Adapted code.** The structure follows the course HW5 template. The request-ID header, the logged generic 500, the `extra="forbid"`/`strict` contract, and the "invalid input never reaches the model" test pattern are adapted from the [Lesson 9 demo](../../../Lectures/Lecture%209/demo/app.py). The model and its training are my HW4 code, imported rather than copied.
* **AI assistance.** I used Claude (Anthropic) in Claude Code, which the assignment allows. It wrote the code, tests, Dockerfile, and this README, and ran the checks and the container demo at my direction. I chose which HW4 model to serve, reviewed the input contract and constraints, and checked the evidence.
