# HW5: Serving my HW4 stock-return model

## What is being served

- **Model:** the HW4 walk-forward Ridge model (alpha 1000) with its learned preprocessing (quantile clipping, median imputation with missing-indicators, standard scaling, one-hot sector). Served version: the last annual fit, fit date 2026-01-02, trained on 7,774,241 rows with labels ending 2025-12-31. `model_version` in every response: `hw4-ridge-rank-a1000-fit-2026-01-02`.
- **What the output means:** `prediction` is a centred cross-sectional percentile rank of the 90-day forward return (roughly -0.5 to 0.5). It is a rank score, **not a return**, and one number only means something relative to other stocks scored on the same date.
- **HW4 provenance:** HW4 never saved a fitted model, so `export_model.py` refit the 2026-01-02 model exactly as HW4's walk-forward code does, then checked it against HW4's recorded `scores.parquet`: 365,938 rows compared, max absolute difference 0.0. HW4 commit: `6d363be9071ed4ec180eac26a2da29a9818d704b`. A reviewer cannot rerun `export_model.py` because it needs my HW4 data; it is included as provenance only.

## Input/output contract

`POST /predict` takes a JSON object with exactly these 13 fields (no extras). Ratios come from the latest available **quarterly** filing (single quarter, not annualised, not TTM). Order below is the training feature order.

| Field | Meaning | Allowed range |
|---|---|---|
| `momentum_21` | adjusted-close return over the last 21 trading sessions | [-1, 1000] |
| `momentum_63` | same, 63 sessions | [-1, 1000] |
| `momentum_252_21` | adj_close(t-21) / adj_close(t-252) - 1 | [-1, 1000] |
| `volatility_63` | std of last 63 daily returns x sqrt(252) | [0, 100] |
| `drawdown_252` | price / max price over last 252 sessions - 1 | [-1, 0] |
| `net_margin` | net income / revenue | [-1000, 1000] |
| `operating_roa` | operating income / total assets | [-1000, 1000] |
| `debt_to_assets` | total debt / total assets | [0, 1000] |
| `cash_to_assets` | cash and equivalents / total assets | [-1000, 1000] |
| `fcf_to_assets` | free cash flow / total assets | [-1000, 1000] |
| `accruals_to_assets` | (net income - operating cash flow) / total assets | [-1000, 1000] |
| `filing_age_days` | calendar days since the filing date, plus 1 | [0, 1095] |
| `sector` | one of the 11 sectors in `model_metadata.json` (exact spelling and case) | enum |

Response (200): `{"prediction": <finite float>, "model_version": "<string>"}`.

- **422** for a missing field, an extra field, a wrong type (strings and booleans are rejected; plain integers are accepted for numeric fields), NaN or +/-Infinity, a value outside the range above, an unknown or wrong-case sector, or a body that isn't an object. The 422 body lists `loc`, `msg` and `type` but deliberately omits the submitted `input`.
- **500** if inference fails (including a non-finite result): the body is exactly `{"detail": "Prediction failed"}`; the real error goes to the server log only.
- **The ranges are my judgment, not derived from the data**, except where a definition guarantees one (volatility >= 0, drawdown in [-1, 0], returns >= -1). The 1000, 100 and 1095 limits are deliberate guesses; the training data has some more extreme values, and I did not count how many rows they affect.

## Files

`app.py` (FastAPI app, input/output models, error handling), `model_service.py` (artifact loading and checks, no FastAPI), `quantile_clipper.py` (custom transformer the pickle imports), `model.joblib` + `model_metadata.json` (artifact and its metadata), `example_request.json` (AAPL on 2026-06-30), `hw4_reference.json` (6 HW4 rows with HW4's recorded scores), `tests/test_app.py`, `Dockerfile`, `Makefile`, `requirements.txt`, `pyproject.toml`, plus evidence files listed below.

## Setup (local)

Python 3.12. From this folder, in Git Bash on Windows:

```sh
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

On macOS/Linux use `python3.12 -m venv .venv` and `source .venv/bin/activate`. Model libraries are pinned to my HW4 training environment: scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3, threadpoolctl 3.6.0 (Python 3.12.9). No `.venv`, database or credentials are committed.

## Artifact loading

- `model.joblib` (5,498 bytes, committed) holds the fitted sklearn Pipeline; sha256 `5baddcfaee944c9b23f6fb9fdfc0193d9d94b95378037250472fbe723f8cabf0`. `model_metadata.json` records the feature order, sector categories, library versions, fit date, row counts and the sha256.
- The service loads it **once at startup** (FastAPI lifespan), never per request, and never refits or touches the HW4 database.
- At startup it refuses to run if the file's sha256 or any of the five library versions doesn't match the metadata (tested with a flipped byte and a changed version).
- The pickle references `quantile_clipper.QuantileClipper`, so that module ships with the service and image. It is a copy of `QuantileClipper` from the course's `Lectures/modeling/course_data.py`. I gave it its own name rather than `transformers.py`, because the pickle stores the module name and `transformers` collides with the HuggingFace package.
- Pickle files can run code when loaded: only load artifacts you created or trust.

## Reproduce my results

Commands (if `make` isn't available, as in my Git Bash, run the right-hand command):

| Task | make | Direct |
|---|---|---|
| Tests | `make test` | `python -m pytest -q` |
| Lint | `make lint` | `python -m ruff check .` then `python -m ruff format --check .` |
| Latency report | | `python -m pytest -q -s -k performance` |
| Build image | `make build` | `docker build -t mfe-hw5:local .` |
| Run container | `make run` | `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` |

With the container running:

```sh
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

Expected: HTTP 200, `prediction` 0.04361991939225726 (the same value HW4 recorded for this row) and the model version above. Port 8000 must be free, or change the host port. Interactive docs are at `/docs`.

## Evidence in this folder

- `test_output.txt`: 91 tests pass. `lint_output.txt`: `ruff check` and `ruff format --check` clean. `latency_report.txt`: see below. In these three files I trimmed my local folder path from one deprecation-warning line; nothing else was edited.
- `docker_build_output.txt`: the image build. `container_evidence.txt`: valid request (200), three rejected requests (422: missing field, NaN, out-of-range), container logs, then the same image stopped and started again and a valid request answered with the identical prediction. `freeze_diff.txt`: `pip freeze` in the container vs. my local venv; the only difference is `colorama` (Windows-only), so the transitive dependency versions matched on this build.
- **Tests (91):** valid inputs (6 HW4 reference rows vs. a separately loaded pipeline and vs. HW4's recorded scores, tolerance 1e-9); invalid inputs return 422 **and** a spy shows the model was never called; 500 on inference failure without leaking the error; startup rejects a tampered artifact or wrong library version.
- **Latency:** `latency_report.txt` shows median 20.3 ms and p95 23.2 ms over 100 timed requests after 5 warm-ups, using the in-process test client (includes validation, JSON and `model.predict`; excludes network and uvicorn) on Windows 11, Intel 8-cpu laptop, Python 3.12.9. It varies with machine load (16.3/19.9 ms on a quiet earlier run; 39/68 ms on a busy one). I did not measure why.
- **Mutation check:** I broke the code on purpose in a scratch copy. Allowing extra fields: caught. Widening the volatility upper bound from 100 to 1000: **not caught at first**, because my tests imported the bounds from `app.py`, so the test moved with the code. I hard-coded the expected bounds in the tests, and the same change now fails exactly one test. Turning off `allow_inf_nan=False`: still passes, because the range checks already reject NaN and infinity, so it is a redundant second layer.

## Limitations and untested assumptions

- Missing values can't be sent. HW4 imputed them (about 3% missing for `momentum_252_21` and `drawdown_252`), so stocks with under a year of history can't be scored through this API.
- The service takes precomputed features. It does not compute them from prices or filings, and does not apply HW4's data-quality flag or universe eligibility.
- The model is a fixed annual fit (2026-01-02), not refreshed.
- The image uses Python 3.12.16 while I trained on 3.12.9. The example row gave identical output to every printed digit in the container, but that is one row, not a proof for all inputs.

## What needs attention before a real deployment

- Feature parity: something has to compute these 12 features identically to HW4 at serving time.
- Monitoring and drift: input ranges and score distribution against the training data.
- Model refresh and versioning: when the model is refit and how clients see `model_version` change.
- Security: authentication and rate limits; the container runs as root; pickle artifacts are code.
- Hygiene: runtime-only requirements (the image currently includes pytest, ruff and httpx), a health endpoint, log without payloads, pin or lock transitive dependencies.

## Credits and AI assistance

- `quantile_clipper.py` is adapted from `QuantileClipper` in the course repo's `Lectures/modeling/course_data.py`. The app structure follows the Lecture 9 template and demo.
- I used Claude and Claude Code for planning, drafting code and tests, and reviewing.
