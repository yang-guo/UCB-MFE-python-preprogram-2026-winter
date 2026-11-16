# HW5 - Anouksha Bansal

A FastAPI service, packaged in Docker, that serves the **HW4 `ridge_core_ret` model**:
`Ridge(alpha=10)` fitted on seven same-date percentile-rank signals, predicting a stock's
90-day return minus the screened-universe mean.

| | |
|---|---|
| Model version | `hw4-ridge_core_ret-fit-2026-01-02` (the final walk-forward fit of the HW4 test run) |
| Training rows | 1,196,310 (fit date 2026-01-02) |
| Artifact | `artifacts/model.joblib` (1.8 KB, committed) + `artifacts/model_metadata.json` |
| Stack | Python 3.12, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, the same versions as HW4 training |

## Files

| File | Purpose |
|---|---|
| `app.py` | HTTP layer: `InputData`/`OutputData` schemas, lifespan loading, `/predict`, `/health`, error handling |
| `hw4_model.py` | Model logic only: feature names and order, float32 DataFrame input, artifact loading and checks |
| `export_model.py` | Training side (not part of the service): re-creates and saves the HW4 fit, then checks it against HW4's outputs |
| `tests/test_app.py`, `tests/hw4_reference.json` | 28 tests, plus reference inputs and scores recorded by HW4 |
| `Dockerfile`, `.dockerignore`, `Makefile`, `requirements.txt`, `pyproject.toml` | Build, tooling, pinned versions |
| `example_request.json` | A real HW4 input (AAPL, signal date 2026-09-25) |
| `measure_latency.py` | Real-HTTP latency client |
| `milestone5_.png`, `milestone5_2.png` | Milestone 5 |

## Input/output contract

`POST /predict` takes one JSON object with **exactly these seven fields, in this order**
(the HW4 `CORE` feature order; the app asserts at import time that the schema and the model agree):

| Field | Raw signal that gets ranked |
|---|---|
| `r_gross_profitability` | gross profit / total assets |
| `r_operating_roa` | operating income / total assets |
| `r_fcf_to_assets` | free cash flow / total assets |
| `r_earnings_yield` | 4 × quarterly net income / market cap |
| `r_sales_to_price` | 4 × quarterly revenue / market cap |
| `r_sales_growth_yoy` | year-over-year quarterly revenue growth |
| `r_mom_252_21` | 12-1 month price momentum |

- **Type and units:** each value is a JSON number with no units: the stock's percentile rank on
  that signal date among the screened universe, minus 0.5. So 0 is the median stock,
  +0.5 the highest and about -0.5 the lowest.
- **Allowed range:** **[-0.5, 0.5]**. In HW4, `rank(pct=True) - 0.5` produces values in
  (-0.5, 0.5], so anything outside that range can't be a rank and probably means the
  caller sent a raw ratio by mistake.
- **Missing values:** every field is required, but `null` is allowed and means the raw signal
  was unavailable, for example no fundamentals. Like HW4's `fillna(0.0)`, the pipeline's
  `SimpleImputer` sets it to 0, the median rank.
- **Rejected with 422:** a missing field, an extra field (including `ticker`), a numeric string
  (`"0.1"`), a boolean, `NaN`/`Infinity`, out-of-range values, and a list in place of the
  object. The model has named fields, not a list, so the wrong feature count shows up as a
  missing or extra field. Pydantic runs in strict mode with `allow_inf_nan=False`, so nothing
  is coerced silently.

Response (200):

```json
{"prediction": 0.0007023527286946774, "model_version": "hw4-ridge_core_ret-fit-2026-01-02"}
```

`prediction` is a finite float: predicted 90-day return minus the universe mean, in decimal
units (0.0007 = 0.07 percentage points). HW4 used it to rank stocks, not as a calibrated
return forecast.

Errors: **422** returns `{"detail": [{"loc", "msg", "type"}, ...]}`. **500** returns
`{"detail": "Prediction unavailable"}` with an `X-Request-ID` header. The traceback is logged
on the server with that ID and is never included in the response. A non-finite model output
also gives a 500, never a fake success.

> **A bug I found and fixed:** FastAPI's default 422 handler echoes each rejected `input`
> back to the client. For a `NaN` input, that response can't be encoded as JSON, so the
> rejection itself turned into a **500**. `app.py` has a custom handler that returns only
> `loc`/`msg`/`type`, and a test covers it.

## Artifact: how it was made and how it is loaded

HW4 refit the model at each walk-forward step but never saved a fitted object.
`export_model.py` repeats the **last** HW4 fit exactly: same panel, same 1%/99% target clip,
same 1,196,310 rows, and code copied from `hw4.ipynb` cells 3, 11 and 12. It saves a
`Pipeline(SimpleImputer(constant 0) → Ridge(alpha=10))` together with its version string.
To check the refit, it compares the refit's scores with the scores the HW4 notebook itself wrote
to `HW4/.../outputs/test_predictions.parquet`. Over all 74,438 rows from that fit's
period, the maximum absolute difference is 0.0. Re-creating the artifact needs the HW4 outputs
and the repo's HW4 environment (for pyarrow): `../../../.venv/bin/python export_model.py`.

**The service never trains and never touches the HW4 database.** The `lifespan` function calls
`hw4_model.load_bundle()` **once per process**. That function checks the artifact's feature
names and order and its version string, and refuses to start if either is wrong. joblib uses
pickle, so loading a file can run code; the service only loads this self-created artifact.
The artifact is 1.8 KB, so it's committed directly, and the Docker image copies it in.

## Tests (`make test`) and lint (`make lint`)

**28 passed. `ruff check` and `ruff format --check` are clean.** The full output is in
`evidence/test_and_lint.txt`.

1. **Valid prediction:** the real model returns 200 with the exact response schema. The API
   prediction equals a direct `pipeline.predict` call, and it matches **HW4's recorded scores**
   for AAPL, JPM, XOM and ACHR on 2026-09-25 (ACHR has a `null` feature). Those scores come from
   HW4's `test_predictions.parquet`, picked out by `export_model.py`. Batch and single-row
   predictions agree, and `null` gives the same result as sending 0.
2. **Validation (17 cases):** every case above returns 422, and a spy on `predict` shows
   **the model is never called**.
3. **Inference failure:** `predict` is monkeypatched to raise, which gives a 500 with no internal
   details in the body and the traceback plus request ID in the log. A model output of `NaN` also
   gives a 500.
4. **Performance:** see below.

**Tolerance, `abs=1e-7`:** the model runs in float32, like HW4. Different BLAS libraries round
float32 sums differently. The host numpy uses Apple Accelerate and the Docker numpy uses OpenBLAS,
so single-row predictions differ from HW4's recorded scores by **up to 3.7e-9** on the host and in
the container (the container gives exactly HW4's AAPL value). With predictions around 1e-2, that is
a few float32 units in the last place. 1e-7 allows for that rounding but is still about 10⁵×
smaller than a typical prediction, so a wrong feature order, wrong scaling or wrong model would
still fail the test.

> **Lesson learned:** my first transcript looked perfect, but it came from a leftover host
> `uvicorn` still listening on 127.0.0.1:8000. Docker's port forward lost the port without any
> error, so the "container" responses actually came from the host process. I noticed because the
> container's log had no request lines and the prediction differed in the 9th decimal place
> (Accelerate on the host vs. OpenBLAS in the container). `demo_container.sh` now refuses to run if
> anything is already answering on its port, and the transcript includes `docker logs` as proof.

## Before a real deployment

- Clients currently have to compute the seven cross-sectional ranks themselves, using the HW4 
  screen and point-in-time fundamentals, before they can call the service. In production, the 
  service or a feature store should compute ranks from raw data for the whole universe on each 
  date. A rank for a single stock means nothing without its universe.
