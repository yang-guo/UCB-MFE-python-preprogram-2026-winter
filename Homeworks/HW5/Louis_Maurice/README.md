# HW 5 — Louis Maurice

A FastAPI `POST /predict` service for my HW4 Ridge model of 90-day stock returns, with tests and a Docker image.

## Model served

- **Model:** the HW4 pipeline (`course_data.make_pipeline`: quantile clip → median impute
  with missing indicators → scale, one-hot sector → Ridge with alpha = 1000), from
  [HW4](../../HW4/Louis_Maurice/README.md).
- **Version `hw4-ridge-refit-2026-06-30`:** the last quarterly refit of the HW4 test run. It was
  trained on 381,778 eligible month-end rows (2010-01 to 2026-05) whose 90-day labels ended
  before 2026-06-30, with targets clipped at their 1%/99% quantiles. This is the model that
  scored the final quarter (2026-06-30 to 2026-09-30) of the HW4 backtest.
- **Artifact:** `artifacts/model.joblib` (6 KB) holds the fitted pipeline with its learned
  preprocessing, the version, the feature list, and the scikit-learn version.
  [export_model.py](export_model.py) created it by calling the HW4 functions unchanged
  (`build_panel`, `choose_alpha`, `clip_target`) in the HW4 environment. The service loads
  only this file. It does not retrain and does not read the database.
- **Custom transformer:** `QuantileClipper` is pickled as `course_data.QuantileClipper`, so an
  unmodified copy of `Lectures/modeling/course_data.py` ships with the service and the image.
  It imports `duckdb`, which is why `duckdb` is in the requirements.
- **Versions:** the requirements pin the same model stack as HW4's `uv.lock` (scikit-learn 1.8.0,
  numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3). `model.load_bundle` refuses to start
  if the installed scikit-learn version differs from the one that saved the artifact.

## Input / output contract

`POST /predict` takes one JSON object with exactly these 13 fields, in HW4 training order.
Returns, volatility, drawdown and ratios are **fractions** (0.05 = 5%).

| Field | Type | Allowed | Meaning |
|---|---|---|---|
| `momentum_21` | number | −1 to 1e6 | 21-session return |
| `momentum_63` | number | −1 to 1e6 | 63-session return |
| `momentum_252_21` | number or null | −1 to 1e6 | 12-1 month momentum |
| `volatility_63` | number | 0 to 1e6 | annualized 63-day volatility |
| `drawdown_252` | number or null | −1 to 0 | drawdown from the 252-day high |
| `net_margin` | number or null | ±1e15 | net income / revenue |
| `operating_roa` | number or null | ±1e15 | operating income / assets |
| `debt_to_assets` | number or null | 0 to 1e15 | total debt / assets |
| `cash_to_assets` | number or null | ±1e15 | cash / assets |
| `fcf_to_assets` | number or null | ±1e15 | free cash flow / assets |
| `accruals_to_assets` | number or null | ±1e15 | accruals / assets |
| `filing_age_days` | number | 0 to 1000 | days since the latest filing became usable |
| `sector` | string | one of the 11 training sectors | e.g. `"Technology"` |

- **Why these bounds:** they reject impossible values (a return below −100%, a negative volatility,
  a positive drawdown, an unknown sector). The upper limits are loose on purpose. The training
  data contains extreme raw values (for example `momentum_21` up to 140,000 and `net_margin`
  down to −3.1e14), and the saved pipeline clips every feature at its training 1%/99%
  quantiles, so a large but possible input is handled exactly as in HW4.
  `filing_age_days` reached 606 in training.
- **`null`:** allowed only for features that were sometimes missing in training. It is
  imputed with the training median, as in HW4. The key itself is still required.
- **Rejected with 422:** missing or extra fields, numeric strings, booleans, null for a required
  feature, NaN/Infinity, out-of-range values, unknown sectors, and a list instead of an object.
  The model is called with a one-row DataFrame that has the training column names.
- **Response 200:** `{"prediction": <finite float, predicted 90-day return>, "model_version": "hw4-ridge-refit-2026-06-30"}`.
- **Response 500:** `{"detail": "Prediction unavailable"}` when inference fails. The traceback goes
  to the server log only.

The example input [example_request.json](example_request.json) is AAPL on 2026-06-30, a real HW4
signal row. HW4 code predicted 0.0501 (+5.0% over 90 days)
([artifacts/reference_example.json](artifacts/reference_example.json)).

## Setup and commands

From this folder (Python 3.12):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test      # python -m pytest -q
make lint      # python -m ruff check . && python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # docs at http://127.0.0.1:8000/docs
```

Docker (the artifact is committed, so the build needs nothing else):

```sh
make build     # docker build -t mfe-hw5:local .
make run       # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Rebuilding the artifact is optional. It needs `equities_top2000.duckdb` in
`Lectures/modeling/data/`. Run `uv sync --locked`, then from the repository root:
`uv run --locked python Homeworks/HW5/Louis_Maurice/export_model.py` (about 2 minutes).

## Tests (`tests/test_app.py`)

1. **Valid prediction:** the real saved model returns 200 and the exact response schema.
   The API result equals a direct `pipeline.predict` call on the same DataFrame
   (`rel=1e-12`). It also equals the HW4 reference score for AAPL (`rel=1e-9`). That score
   came from `export_model.py` scoring the full 2026-06-30 universe in one batch with HW4
   code. The pipeline is row-independent, so the tolerance only covers floating-point noise.
   A second test sends `null` financials and checks that they are imputed.
2. **Input validation:** 16 parametrized cases, each asserting 422. The cases cover missing
   fields, extra fields, wrong types, null for a required field, bounds, unknown sector,
   wrong feature count/shape, and NaN/±Infinity. A fixture replaces `predict` with a
   function that fails the test, which shows invalid input never reaches the model.
3. **Inference failure:** valid input, with the loaded pipeline's `predict` patched to raise.
   The test asserts 500 and that the exception text is not leaked.
4. **Performance:** 5 warm-up requests, then 50 timed requests with `time.perf_counter`.
   This is an **in-process `TestClient`** measurement (no network) on an Apple M1 MacBook,
   macOS 26.5. Result: **median 2.99 ms, p95 3.92 ms**. The test asserts p95 < 250 ms. That
   budget is generous because one-row Ridge inference takes a few ms; it only catches a real
   regression, such as reloading or refitting the model on every request.
5. A contract test checks that the request schema, the artifact, and the pipeline's
   `feature_names_in_` use the same feature order.

Results: `21 passed`; `ruff check`: all checks passed; `ruff format --check`: 4 files already formatted.

While writing the tests I found a real bug. FastAPI's default 422 response echoes the
rejected input, and a NaN input cannot be serialized to JSON, so that request crashed with
a 500. `app.py` now uses a validation handler that returns only the location, message and
type of each error.

## Request evidence

- **Local server (real HTTP):** [evidence/local_uvicorn_transcript.txt](evidence/local_uvicorn_transcript.txt).
  It shows a valid request (200, prediction 0.0501, model version), an invalid request (422),
  and a valid request after stopping and restarting the server.
- **Container:** [evidence/container_transcript.txt](evidence/container_transcript.txt), produced by
  `make build && ./evidence/container_demo.sh`. It shows a valid request (200), a rejected request
  (422), then the container stopped and restarted from the same image and serving again.

## Before a real deployment

- **Model quality:** this model had no out-of-sample signal in HW4 (rank IC 0.007, CAGR −3.1%).
  Serving it is a plumbing exercise, not a recommendation to trade on it.
- **Freshness:** the model is frozen at 2026-06-30. It needs a scheduled refit, artifact
  versioning/registry, and monitoring of input drift and prediction distributions.
- **Inputs:** clients must compute the features exactly as `course_data.py` does, at the
  same point in time. A shared feature service would remove that risk.
- **Operations:** a health/readiness endpoint, authentication, rate limiting, request IDs and
  structured logs, CI running these checks, a non-root container user, and dependency scanning.
- **Artifact:** joblib uses pickle. Only trusted artifacts should be loaded, ideally with a checksum.

## Sources

- Structure of `app.py`, the Dockerfile, and the tests adapted from the
  [Lesson 9 demo](../../../Lectures/Lecture%209/demo/).
- `course_data.py` copied unchanged from `Lectures/modeling/`; model code from my HW4.
- AI-assisted work: the code and write-up were produced with an AI coding assistant. I
  reviewed the implementation and checks.
