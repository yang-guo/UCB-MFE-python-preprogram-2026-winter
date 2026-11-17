# HW5: Serving the HW4 stock-ranking model (Ian Kwon)

A FastAPI service, tested and packaged with Docker, that serves the model from my HW4
submission: the frozen strategy `P1_top50__R1` (config hash `f06421a6…`, branch
`hw_4_Ian_Kwon`).

| | |
|---|---|
| Model served | `quantlab.models.FixedComposite`, HW4 experiment `P1_top50__R1-f06421a6` |
| `model_version` | `hw4-P1_top50__R1-f06421a6` |
| Endpoint | `POST /predict` (also `GET /health`, `GET /docs`) |
| Tests | 38 passed; `ruff check` and `ruff format --check` clean ([evidence/test_and_lint.txt](evidence/test_and_lint.txt)) |
| Latency | in-process TestClient: median 2.7-3.8 ms, p95 3.4-4.8 ms; real HTTP to the container: median 2.0-2.2 ms, p95 2.4-2.5 ms |
| Container | built, valid/invalid requests, restart, request again ([evidence/container_transcript.txt](evidence/container_transcript.txt)) |
| Makefile | `test`, `lint`, `format`, `build`, `run`, `demo` all run with exit code 0 ([evidence/make_targets.txt](evidence/make_targets.txt)) |

## 1. What is served

HW4's submitted model is a **fixed composite**: the signed average of seven same-date
percentile ranks.

```
score = ( gross_profitability + operating_roa - accruals_to_assets + book_to_market
          - vol_63 + mom_252_21 - ret_21 ) / 7          (each term is a rank in [-0.5, 0.5])
```

The signs were fixed before any HW4 result.  The estimator's `fit` only converts them
into weights of ±1/7 (`coef_`).  A missing rank (NaN) counts as 0, the median.

**Learned preprocessing: none, by design.**  HW4 turns raw features into ranks with
`quantlab.features.cross_sectional_rank`, i.e. `rank(pct=True) - 0.5` within each
signal date's eligible stocks.  This transform has no fitted parameters: it depends only on
the same date's cross-section, so there is nothing to serialize.  The composite has no
imputer or scaler (HW4's Ridge candidates did; they were not selected).  The service
therefore takes the seven **ranks**: the exact representation the model saw in HW4.

### Artifact and provenance

| File | What it is |
|---|---|
| `artifacts/model.joblib` (395 bytes) | The fitted estimator, pickled with joblib |
| `artifacts/model_metadata.json` | Version, feature order, signs, `coef_`, fit date, library versions, artifact SHA-256 |
| `quantlab/models.py` | The custom estimator's module, a **byte-for-byte copy** of HW4's `quantlab/models.py` (same SHA-256 `0c417865…`); the pickle imports `quantlab.models.FixedComposite` |
| `scripts/export_hw4_model.py` | Produced the artifact, the reference cases and `example_request.json` from HW4 code |
| `example_request.json` | A valid input: AAPL on 2015-12-31, as scored by HW4 |

`scripts/export_hw4_model.py` does not re-implement the model.  It imports the HW4
package and does four things:

1. It checks `config/frozen_strategy.json` against `config/FROZEN.lock`.
2. It builds the HW4 panel in **research mode**: HW4's SQL firewall caps data at
   2016-03-31, so exporting never touched the final-test holdout.
3. It runs HW4's `walk_forward_scores` exactly as `run_research` does, and saves the
   estimator fitted at the last development refit (2015-10-02, 476,371 training rows,
   last training label 2015-09-29).
4. It records three (input, HW4 score) pairs and checks that the saved model reproduces
   them.

Because the composite's weights do not depend on the training data, this estimator is
identical to every quarterly refit in the HW4 final run.  Re-running the script produced
the same artifact SHA-256 (`80da27f3…`).

**Versions.**  HW4 trained under Python 3.12.9 with scikit-learn 1.8.0, NumPy 2.3.5 and
joblib 1.5.3 (course `uv.lock`).  `requirements.txt` pins the same versions.  The local HW5
venv uses Python 3.12.3 and the image uses Python 3.12.16, both with the same libraries.
At startup the service refuses to load the artifact in any of these cases:

* its SHA-256 differs from the metadata (checked *before* unpickling: only trusted, unchanged files are loaded);
* the installed scikit-learn version differs from the export version, or sklearn emits `InconsistentVersionWarning`;
* the class, feature order or `n_features_in_` differ from what the API expects.

## 2. Input contract (`InputData`)

A JSON object with exactly these seven fields.  Field order in the JSON does not
matter: the service always passes values to the model in the order below, which is the
HW4 training column order.

| # | Field | Raw feature (HW4 registry) | Model sign |
|---|---|---|---|
| 1 | `gross_profitability_csr` | quarterly gross profit / total assets | + |
| 2 | `operating_roa_csr` | quarterly operating income / total assets | + |
| 3 | `accruals_to_assets_csr` | (net income - operating cash flow) / total assets | - |
| 4 | `book_to_market_csr` | total equity / market-cap proxy | + |
| 5 | `vol_63_csr` | annualised volatility of the last 63 daily returns | - |
| 6 | `mom_252_21_csr` | return from 252 to 21 sessions before the signal date | + |
| 7 | `ret_21_csr` | return over the last 21 sessions | - |

**Type and units.**  Each value is a JSON number or `null`.  Each value is a
**centred same-date percentile rank** (unitless): `rank(pct=True) - 0.5` among the stocks
eligible on that signal date.  0.5 is the highest raw value and 0 the median.  The `_csr`
suffix is HW4's column name ("cross-sectional rank").  Prices are as of the signal close.
Statements count from one day after their filing date (HW4's timing rules).

**Allowed range: [-0.5, 0.5].**  The smallest possible rank is 1/n - 0.5, where n is the
number of eligible stocks that date (at least 1,356 in 2012-2015, at most 2,000).  Across all 602,903 eligible HW4 research
rows, every feature lies between -0.49945 and 0.5.  A value such as 1.7 is almost
certainly a raw ratio sent by mistake, and is rejected.

**`null` = unavailable.**  HW4 passed a missing rank to the model as NaN, which counts as
the median.  The API keeps that meaning through an explicit `null`: the field is still
required, but its value may be `null`.  Missing counts among the 602,903 eligible research
rows (2010-2015) were:

| Feature | Missing count | Typical cause |
|---|---|---|
| mom_252_21 | 14,863 (2.5%) | less than 12 months of price history |
| accruals_to_assets | 1,676 | no cash-flow statement yet |
| gross_profitability | 801 | no statement yet |
| operating_roa | 791 | no statement yet |
| book_to_market | 723 | no statement yet |
| ret_21 | 2 | — |
| vol_63 | 0 | — |

**Rejected with 422** (all before the model is called; the tests check this):

* missing field (wrong feature count, too few), extra or misspelled field (too many);
* a list instead of the named object, an empty, `null` or malformed body;
* numeric strings (`"0.1"`) and booleans: `strict=True`, so nothing is silently coerced;
* `NaN`, `Infinity`, `-Infinity` (`allow_inf_nan=False`);
* values outside [-0.5, 0.5].

There are no categorical inputs.

## 3. Output contract (`OutputData`)

```json
{"prediction": 0.10577686157588896, "model_version": "hw4-P1_top50__R1-f06421a6"}
```

* `prediction`: a finite float in [-0.5, 0.5] (`FiniteFloat`), the HW4 composite score.
  Higher means more attractive for the next 90 calendar days (the HW4 horizon: entry at
  the next session's close, exit at the first quoted close at least 90 days later).  It
  is a **cross-sectional ranking score, not a forecast return**.  HW4 bought the 50
  highest scores each week.
* `model_version`: the HW4 experiment that produced the model.
* Response header `X-Request-ID`: also written to the server log on failure.

## 4. Errors

| Status | When | Body |
|---|---|---|
| 422 | the request breaks the contract | `{"detail": [{"loc": [...], "msg": ..., "type": ...}]}` |
| 500 | the model raises, or returns NaN/Inf or the wrong shape | `{"detail": "Prediction unavailable"}` and `X-Request-ID` |

Inference errors are never turned into a default prediction.  The traceback goes to the
server log with the request ID; the client sees no internal details.

The 422 body deliberately omits FastAPI's usual `input` echo.  The tests caught a real
bug here: FastAPI's default handler echoes the offending value, JSON cannot encode NaN,
so a `NaN` request crashed the response and returned **500** instead of 422.  The custom
handler in `app.py` reports only the location, message and type.

## 5. Setup and commands

Python 3.12, in a separate environment, from this folder:

```sh
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: py -3.12 -m venv .venv
python -m pip install -r requirements.txt                    #          .venv\Scripts\Activate.ps1

python -m pytest -q                       # make test   (prints the latency report)
python -m ruff check .                    # make lint
python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000      # then open http://127.0.0.1:8000/docs
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The service needs only this folder.  It does not import HW4, open the database, or fit
anything.

### Container (reproducing the build)

The artifact is committed (395 bytes), so the build needs only this folder:

```sh
docker build -t mfe-hw5:local .                                        # make build
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local     # make run
bash scripts/container_demo.sh 2>&1 | tee evidence/container_transcript.txt   # make demo
```

The image is `python:3.12-slim` with the workdir `/app` and the pinned requirements.  It
contains only `app.py`, `model_service.py`, `quantlab/` and `artifacts/`, and runs as a
non-root user.  `.dockerignore` keeps `.venv`, databases, `.env` files, tests and
scripts out of the build context (about 60 kB).  Stop anything else on host port 8000 first, or
run with `PORT=8001 bash scripts/container_demo.sh`.

My machine runs Windows 11 without Docker Desktop.  I used Docker Engine 29.1.3 inside
WSL2 Ubuntu 22.04 (`apt-get install docker.io`, daemon started by hand), which uses
Docker's legacy builder; the deprecation notice in the transcript comes from that.  On
macOS, Docker Desktop and the commands above are enough.

All six Makefile targets were run and exited with code 0
([evidence/make_targets.txt](evidence/make_targets.txt)).  `make test`, `make lint` and
`make format` ran on Windows (GNU Make 4.4.1 from Git Bash, HW5 venv activated).  `make
build`, `make run` and `make demo` ran in WSL2, where Docker runs (GNU Make 4.3).  For
`make run`, which keeps the server in the foreground, a request returned 200 before the
container was stopped.

### Re-exporting the artifact (only if needed)

Run this with the **HW4/course environment**, which has duckdb and the HW4 data, from the
repository root:

```sh
git worktree add ../hw4-src hw_4_Ian_Kwon
.venv/Scripts/python Homeworks/HW5/Ian_Kwon/scripts/export_hw4_model.py \
    --hw4-dir ../hw4-src/Homeworks/HW4/Ian_Kwon \
    --database Lectures/modeling/data/equities_top2000.duckdb     # about 30 s
```

It rewrites `artifacts/`, `tests/data/hw4_reference.json` and `example_request.json`, and
fails if the saved model does not reproduce HW4's scores.

## 6. Tests (`tests/test_app.py`)

Reusable fixtures provide two clients: `client` and `failure_client`
(`raise_server_exceptions=False`), both used as `with TestClient(...)` so the lifespan
runs.  The other fixtures are `valid_input_data` (the AAPL reference row) and
`model_must_not_run` (a spy on the loaded model).

1. **Valid prediction (real model).**  These tests check:
   * status 200, the exact response keys, the `OutputData` schema, a finite value and the
     version;
   * agreement with a direct `estimator.predict` call on a float64 array in training
     order (HW4's representation);
   * the score HW4 itself recorded, for all three reference cases.

   Other tests in this group show that JSON key order is irrelevant, that `null` equals
   sending the median (0.0), that the boundaries ±0.5 and integer `0` are accepted, that
   the model is loaded **once** for many requests, and that `/health` works.
2. **Input validation.**  20 invalid payloads (see section 2) each return 422 with the
   model's `predict` replaced by a spy, which proves the model is never called.  NaN and
   ±Infinity are sent as raw JSON tokens.  One test checks that the error names the
   offending field.
3. **Inference failure.**  `predict` on the loaded estimator is patched to raise, and a
   valid request is sent.  The test asserts a 500 with the documented body, no internal
   message in the response, and a logged traceback carrying the same request ID.
   Further cases (NaN, Inf, two values or no value returned) must also give 500, never a
   prediction.
4. **Performance.**  See section 7.

**Reference result** (`tests/data/hw4_reference.json`).  The inputs and scores were
recorded by HW4's `walk_forward_scores` (research mode, frozen config) for signal date
2015-12-31, before any API code existed:

| Case | Symbol | HW4 score |
|---|---|---|
| all seven ranks present | AAPL | 0.10577686157588898 |
| one rank missing (`mom_252_21_csr` = null) | ACRS | -0.3818262690300268 |
| highest score on that date | DHIL | 0.31340480258777337 |

**Tolerance: absolute 1e-12.**  HW4 scored a whole date at once (an (n, 7) @ (7,)
product) while the API scores one row.  BLAS may sum the terms in a different order, and
the AAPL result differs by one unit in the last place (1.4e-17).  1e-12 is about 1e8 times smaller than
the score gap between neighbouring stocks (~1e-4), so any real change still fails.

## 7. Latency report

Hardware: AMD Ryzen 9 6900HS (8 cores / 16 threads), 16 GB RAM, Windows 11.

| Measurement | Method | Requests | Median | p95 |
|---|---|---|---|---|
| In-process | FastAPI `TestClient` (no network), `time.perf_counter` around each sequential `POST /predict`, Python 3.12.3 | 10 warm-up + 200 | 2.69-3.75 ms | 3.40-4.84 ms |
| Real HTTP, container | `curl -w %{time_total}` from WSL2 to the container on 127.0.0.1:8000, new TCP connection per request | 10 warm-up + 100 | 1.99-2.16 ms | 2.35-2.52 ms |

Ranges are from separate runs.  The test prints its own line on every `pytest` run.

**Budget: p95 < 50 ms (asserted, in-process).**  The HW4 strategy scores at most 2,000
stocks (its top-2,000 universe) once a week after the close, and trades at the next
session's close.  Even one sequential request per stock at 50 ms takes 100 seconds, far
inside that window.  The budget is at least 10 times the observed p95, so it fails only on a gross
regression and not on a slower CI machine.

## 8. Evidence

* [evidence/test_and_lint.txt](evidence/test_and_lint.txt): `ruff check`, `ruff format --check` and `pytest -v` (38 passed) with the latency line.
* [evidence/container_transcript.txt](evidence/container_transcript.txt): `docker build`, then:
  * a valid request (200, prediction 0.10577686157588896, model version);
  * a numeric string (422) and a NaN plus a missing field (422);
  * `docker stop`, a new `docker run` of the same image, `/health` and the same 200 prediction;
  * the HTTP latency and the container log.
* [evidence/local_uvicorn.txt](evidence/local_uvicorn.txt): the same request against `python -m uvicorn` on Windows, and `/docs` returning 200.
* [evidence/make_targets.txt](evidence/make_targets.txt): output and exit code of all six Makefile targets.
* Optional CI: not done.

## 9. What would need attention before a real deployment

* **Features are computed outside the service.**  The caller must compute the seven raw
  features and rank them across that date's full eligible universe.  A single-stock
  request cannot rank itself, so a real system needs a feature pipeline that reuses the
  HW4 feature code, and probably a batch endpoint that takes the whole cross-section.
  Otherwise train/serve skew (different eligibility, timing or ranking) would go
  unnoticed.  The range checks catch raw values sent by mistake, not wrong ranks.
* **The model itself.**  In HW4's final test the strategy did not beat the S&P 500
  (CAGR 13.8% vs 15.2%), and its rank IC was weak (0.036).  Before it drives money it
  needs monitoring: live IC, the share of null inputs, and score drift, plus a decision on
  whether to use it at all.
* **Artifact handling.**  joblib/pickle executes code when loaded.  The SHA-256 check
  helps, but artifacts should come from a signed model registry.  For a model this simple,
  the seven weights could be stored as JSON instead.
* **Image.**  The image is 722 MB because `requirements.txt` follows the assignment's
  single-file setup.  Before deployment I would:
  * move the dev tools (pytest, ruff, httpx) and unused libraries (pandas, scipy) into a
    separate dev requirements file;
  * pin the base image by digest and the Python patch version;
  * scan the image for vulnerabilities.
* **Operations.**  The service also needs:
  * authentication, TLS and rate limits;
  * several workers, with readiness and liveness probes for an orchestrator;
  * structured logs, metrics, alerts and request-ID propagation;
  * versioned rollouts and rollback, and CI that runs these checks on every change.
* **Data caveats carried over from HW4.**  These include vendor restatements, snapshot
  sector labels, an approximate universe, and no delisting proceeds.

## 10. Code reuse and AI assistance

* Adapted from `Lectures/Lecture 9/demo/app.py`: the strict `ConfigDict` contract, the
  `FiniteFloat` output, and the 500 handling with an `X-Request-ID` header and a logged
  traceback.  The test that checks invalid input never reaches the model follows the
  demo's monkeypatch pattern.
* `quantlab/models.py` is copied unchanged from my HW4 submission.  The export script
  calls HW4 functions (`experiment.prepare`, `walk_forward_scores`, `feature_columns`).
* Built with Claude Code (Anthropic).  I made these decisions:
  * serve the submitted composite rather than a Ridge candidate with fitted preprocessing;
  * accept `null` with HW4's "median" meaning;
  * run Docker Engine inside WSL.

  The assistant wrote the code, tests and scripts, ran them, and produced the evidence
  files.  It also found and fixed the NaN-echo 500 bug and the demo script's
  cleanup bug.
