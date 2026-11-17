# HW5: Serving my HW4 model — Antonio Rivera

A FastAPI `POST /predict` service, tested and packaged with Docker, that serves the **frozen
HW4 model**: Ridge(alpha=100) on the course F1 baseline features, trained on the per-date
rank of the 90-day forward return (HW4 branch `hw_4_Antonio_Rivera`, PR #257).

```
HW4 code + HW4 panel ──export_model.py (run once)──► model/pipeline.joblib  (5.7 KB)
                                                     model/metadata.json
                                                     model/reference.json
JSON request ─► InputData (strict, 422) ─► 1-row DataFrame ─► pipeline.predict ─► OutputData
                                                               (loaded once, lifespan)
```

## 1. Which model is served

| Item | Value |
|---|---|
| `model_version` | `hw4-ridge-F1-rank-fit2026-07-01` |
| Pipeline | `cd.make_pipeline(Ridge(alpha=100), cd.NUMERIC_FEATURES)`: QuantileClipper(1%, 99%) → SimpleImputer(median, missing indicators) → StandardScaler, plus sector one-hot → Ridge |
| Target | `target_rank` = per-date percentile rank of the 90-calendar-day return, entry at the next close (t+1) |
| Fit date | 2026-07-01 = the **last quarterly refit** of the HW4 final test (2016-04-01 → 2026-09-30). It is the most recent model HW4 used, so it is the one to serve next. |
| Training rows | 8,040,321 eligible rows, signal dates 2010-01-04 onward, only labels that ended before 2026-07-01 (same rule as `hw4_walkforward.py`) |
| Check against HW4 | Re-fit predictions equal the HW4 recorded scores for all 124,716 rows of that quarter (max abs diff 0.0) |
| Versions | Python 3.12.9, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3 — same in HW4 env, HW5 venv, and Docker image |

**Custom transformer.** `QuantileClipper` comes from `Lectures/modeling/course_data.py`, which
imports duckdb. I copied the class (same logic) into `hw5_transformers.py`, and
`export_model.py` builds the pipeline with that copy, so the pickle references
`hw5_transformers.QuantileClipper`. The service and image need no duckdb, no parquet engine,
and no database.

## 2. Input / output contract

`POST /predict`, JSON object, **all 13 keys required**, no extra keys (`extra="forbid"`),
strict types (a numeric string like `"0.1"` or a boolean is rejected), finite numbers only
(NaN/±Infinity → 422). Integers are accepted for float fields (JSON has one number type).
Fields marked *nullable* accept JSON `null` = "value not available"; it becomes NaN and is
median-imputed with a missing indicator, exactly as missing values were handled in HW4
training (about 3% of training rows lack `momentum_252_21`/`drawdown_252`, 0.2–2% lack a
financial ratio). The other fields were never missing in training, so `null` is rejected.

Order below = training column order. All values use data through the signal close t.

| # | Field | Type | Unit | Allowed range | Nullable |
|---|---|---|---|---|---|
| 1 | `momentum_21` | float | 21-session return, fraction (0.05 = +5%) | [-1, 100] | no |
| 2 | `momentum_63` | float | 63-session return, fraction | [-1, 100] | no |
| 3 | `momentum_252_21` | float | return from t-252 to t-21 sessions, fraction | [-1, 100] | yes |
| 4 | `volatility_63` | float | annualized 63-session volatility, fraction | [0, 100] | no |
| 5 | `drawdown_252` | float | close / 252-session max close − 1 | [-1, 0] | yes |
| 6 | `net_margin` | float | TTM net income / revenue | [-1e4, 1e4] | yes |
| 7 | `operating_roa` | float | operating income / total assets | [-1e4, 1e4] | yes |
| 8 | `debt_to_assets` | float | total debt / total assets | [-1e4, 1e4] | yes |
| 9 | `cash_to_assets` | float | cash / total assets | [-1e4, 1e4] | yes |
| 10 | `fcf_to_assets` | float | free cash flow / total assets | [-1e4, 1e4] | yes |
| 11 | `accruals_to_assets` | float | (net income − CFO) / total assets | [-1e4, 1e4] | yes |
| 12 | `filing_age_days` | float | calendar days since latest filing | [0, 1000] | no |
| 13 | `sector` | string (categorical) | one of the 11 training sectors | `Basic Materials`, `Communication Services`, `Consumer Cyclical`, `Consumer Defensive`, `Energy`, `Financial Services`, `Healthcare`, `Industrials`, `Real Estate`, `Technology`, `Utilities` | no |

**Why these ranges.** Hard limits come from definitions: a simple return cannot be below −1;
volatility ≥ 0; a drawdown is in [−1, 0]; filing age ≥ 0 (training max 633 days). Upper limits
on returns and the ±1e4 ratio limits are sanity guards: they keep more than 99.9% of the HW4
training rows (training 0.1%/99.9% quantiles: e.g. `momentum_252_21` 9.5, `net_margin`
−1,068) and reject only values like the vendor unit errors I found in HW4
(`net_margin` = −3e14, returns of +14,000,000%). They do not change real predictions, since
the pipeline clips every numeric feature to its training 1%/99% quantiles anyway.
A sector not seen in training is rejected (the one-hot encoder would silently zero it).

**Output** (200):

```json
{"prediction": 0.512245321993285, "model_version": "hw4-ridge-F1-rank-fit2026-07-01"}
```

`prediction` = predicted per-date percentile rank of the 90-day return (about 0..1, higher =
better). It is meant for **ranking stocks on the same date** (HW4 strategy S1R), not as an
expected return. Always finite (an inference NaN/inf → 500).

**Errors:** 422 `{"detail": [{"type", "loc", "msg"}, ...]}` — I replaced FastAPI's default
422 handler because it echoes the bad input, and a `NaN` input cannot be written back to JSON
(found by a test: the default handler crashed instead of returning 422). 500 = plain
`{"detail": "Prediction unavailable"}` with an `X-Request-ID` header; the exception and
traceback go only to the server log. `GET /health` returns status and model version.

## 3. Files

| File | Purpose |
|---|---|
| `app.py` | API contract, lifespan loading, `/predict`, `/health` |
| `hw5_transformers.py` | `QuantileClipper` (copied from course_data.py) for unpickling |
| `model/pipeline.joblib` | fitted HW4 preprocessing + Ridge (5.7 KB, committed) |
| `model/metadata.json` | version, fit date, feature order, sectors, library versions |
| `model/reference.json` | one real HW4 input row (symbol A, 2026-09-29) + the score HW4 recorded |
| `example_request.json` | the same row as an API request |
| `export_model.py` | one-off export from HW4 code + data (not used by the service or image) |
| `tests/test_app.py` | 29 tests in the 4 required groups |
| `evidence/` | container transcript, invalid request body, lint/test output |

## 4. Setup and reproduce

From this folder (`Homeworks/HW5/Antonio_Rivera/`):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # or: uv pip install --python .venv/bin/python -r requirements.txt

make lint PYTHON=.venv/bin/python      # ruff check + ruff format --check
make test PYTHON=.venv/bin/python      # 29 passed
.venv/bin/python -m pytest -q -s -k performance          # prints the latency report

.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000   # docs: http://127.0.0.1:8000/docs

make build                             # docker build -t mfe-hw5:local .
make run                               # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

(I pass `PYTHON=.venv/bin/python` because my editor sets `VIRTUAL_ENV` to another course
venv; with the HW5 venv activated, plain `make test` works.) Stop anything else on host
port 8000 first.

**Artifact loading.** `lifespan` in `app.py` loads `model/pipeline.joblib` and
`model/metadata.json` once per process, with paths relative to `app.py`. No fitting, no
database. joblib uses pickle, so only load this artifact (I created it from my own HW4 code).

**Rebuild the artifact (optional; needs the gitignored HW4 parquet caches and the course
`.venv` with pyarrow/duckdb).** The exact commands are in the docstring of `export_model.py`:
extract the HW4 branch into a temp dir with `git archive`, link `Lectures/` and the HW4
`outputs/`, then run `export_model.py --hw4-dir ...` from the repo root. It refits, asserts the
match with the HW4 scores, and rewrites `model/` and `example_request.json`.

## 5. Tests (`tests/test_app.py`)

1. **Valid prediction** — real model; 200, exact response keys, finite float, model version,
   the DataFrame sent to the model has the training column order; equals a direct
   `joblib.load(...).predict(DataFrame)` (abs 1e-12) and equals the **HW4 recorded score**
   0.512245321993285 for symbol A on 2026-09-29 (abs 1e-9: same code and library versions, so
   only float noise could differ). Reference source: `outputs/scores_final_ridge_F1_baseline.parquet`
   written by `hw4_walkforward.py` during the HW4 final run, copied by `export_model.py`. A second
   test sends `null` for two nullable fields and gets 200.
2. **Input validation** — 17 parametrized bad bodies (missing field, missing sector, missing
   nullable key, extra field, numeric string, boolean, null in a non-nullable field, wrong
   sector type, unknown sector, out-of-range values, list instead of fields, 11 of 13
   features, empty body), 6 non-finite cases (NaN/±Infinity in two fields), and non-JSON
   text → all 422. A spy on the loaded model's `predict` proves it is called **0 times**.
3. **Inference failure** — `predict` monkeypatched to raise → 500, the exception text is not in
   the response, no prediction is returned, and the log has `inference_failed`. Also a model
   returning NaN → 500.
4. **Performance** — 5 warm-up requests, then 100 timed requests with `time.perf_counter`.

**Latency report:** in-process `TestClient` (no real network), MacBook Air, Apple Silicon
(arm64, macOS), Python 3.12.9: **median 1.80 ms, p95 2.07 ms** (n = 100; earlier run
1.75 / 1.97 ms). Budget asserted: p95 < 250 ms — a one-row Ridge pipeline needs a few ms, so
this leaves room for a slow CI runner but fails on gross regressions (e.g. a refit per request).

The pytest warning `StarletteDeprecationWarning ... install httpx2` comes from the template's
pinned httpx inside FastAPI's TestClient; it does not affect results.

## 6. Container evidence

`evidence/container_transcript.txt` (2026-11-16, Docker 29.9.3, arm64): `mfe-hw5:local`
image; valid curl → **200** `{"prediction":0.512245321993285,"model_version":"hw4-ridge-F1-rank-fit2026-07-01"}`;
invalid body (`evidence/invalid_request.json`: numeric string, unknown sector, extra field)
→ **422**; container stopped (gone from `docker ps -a`), same image started again, same
request → **200** with the same prediction. I started the container with `-d` (same flags as
`make run`) so all requests run in one recorded shell. `evidence/checks.txt` has the lint and
test output.

## 7. Before a real deployment

- **Model quality:** HW4's out-of-sample rank IC was only +0.038 and the strategy did not beat
  the benchmark. Do not trade on this model as is.
- **Retraining and versioning:** HW4 refits every quarter; this artifact is frozen at
  2026-07-01. Need a scheduled export, a model registry/version tag per artifact, and a
  checksum check at load (joblib is pickle: never load untrusted files).
- **Feature pipeline:** the service trusts callers to compute the 12 features exactly as
  `course_data.py` does (same units, same timing, no look-ahead). A real system should compute
  them server-side from a point-in-time feature store, and batch-score a whole date (the
  output is only meaningful as a cross-sectional rank).
- **Data quality:** my sanity bounds are wide; add monitoring of input distribution drift and
  of the share of nulls versus training.
- **Operations:** run as a non-root user, add a Docker `HEALTHCHECK`, multiple workers,
  request size limits, authentication/TLS, structured logs with request IDs, and pin the base
  image by digest. Upgrade `httpx` per the deprecation warning.
- **CI:** run lint + tests + image build on every push (optional GitHub Actions not done).

## 8. AI use and adapted code

- `hw5_transformers.QuantileClipper`: copied from `Lectures/modeling/course_data.py`.
- `app.py` structure (lifespan load, strict Pydantic input with `allow_inf_nan=False`,
  `FiniteFloat` output, logged 500 with `X-Request-ID`): adapted from
  `Lectures/Lecture 9/demo/app.py`.
- `export_model.py` imports my own HW4 code (`hw4_walkforward`, `hw4_model`,
  `hw4_build_cache`) from the HW4 branch and repeats its last refit.
- Claude Code (Anthropic) assisted in pair mode: it read the spec and my HW4 code, drafted the
  export script, app, tests, Dockerfile and README, and ran the checks; I reviewed each step,
  the contract choices (nullable fields, ranges, last-refit model), and the evidence. The
  NaN → 422 crash was found by the tests and fixed with a custom 422 handler.
