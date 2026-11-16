# HW5: Serve and containerize a model (Min Jin)

A FastAPI service, tested and packaged with Docker, that serves the model I chose in HW4: the
**fixed-weight factor score** (`FINAL_MODEL = "Factor score (fixed weights, no fitting)"` in
`Homeworks/HW4/Min_Jin/model.py`).

## Files

| File | What it does |
|---|---|
| `app.py` | FastAPI app: input/output contract, model loading in `lifespan`, `POST /predict`, `GET /health` |
| `factor_model.py` | The HW4 `FactorScore` estimator and feature order (copied from HW4 `model.py`) |
| `export_model.py` | Saves the artifact and records the HW4 reference prediction (run once, not by the API) |
| `artifacts/factor_score.joblib` | The saved HW4 model + metadata |
| `reference_example.json` | One real HW4 input and the score HW4's own code gave it |
| `example_request.json` | The same input, as a request body |
| `tests/test_app.py` | 18 tests: valid prediction, validation, inference failure, latency |
| `measure_latency.py` | Latency over real localhost HTTP against the running server/container |
| `evidence/` | `test_and_lint.txt`; `container_demo.txt`, my terminal transcript of the Milestone 5 requests run by hand |
| `Dockerfile`, `Makefile`, `requirements.txt`, `pyproject.toml` | Container, commands, pinned versions, tool config |

## Which model, and what "preprocessing" means here

HW4 compared seven candidates and only the factor score had a positive validation rank IC, so it was chosen for the test. It averages five same-date percentile ranks:

```
score = mean(rank_operating_roa, rank_fcf_to_assets, rank_drawdown_252,
             rank_momentum_252_21, 1 - rank_volatility_63)
```

The model has no fitted parameters or learned preprocessing. Its inputs are same-date ranks
across that month's stocks, so the API takes the five ranks directly, exactly the columns the HW4
model used. `FactorScore` is saved as an sklearn estimator with its feature order and library
versions. It lives in `factor_model.py`, which is copied into the image, because HW4's `model.py`
also imports the database code.

## Input / output contract

`POST /predict`, JSON body with exactly these five fields:

| # | Field | Type | Range | Meaning (rank among that month-end's eligible top-2000 stocks) |
|---|---|---|---|---|
| 1 | `rank_operating_roa` | number | 0 – 1 | operating income / total assets |
| 2 | `rank_fcf_to_assets` | number | 0 – 1 | free cash flow / total assets |
| 3 | `rank_drawdown_252` | number | 0 – 1 | close / 252-session high − 1 (closeness to the 1-year high) |
| 4 | `rank_momentum_252_21` | number | 0 – 1 | 12-month return skipping the last month |
| 5 | `rank_volatility_63` | number | 0 – 1 | 63-session daily return volatility (send the plain rank; the model flips it) |

Ranks are unitless (0 = lowest that month, 1 = highest). Why these constraints:

- **Range 0 to 1:** ranks always fall in this range. A value outside it means a raw feature was sent instead of its rank, which would give a silently wrong score.
- **Strict, finite numbers:** numeric strings, booleans, `NaN` and `Infinity` are rejected rather than converted to numbers.
- **Exactly these five fields:** missing, extra or misspelled fields are rejected, which also covers a wrong feature count. HW4 skipped a missing rank when averaging; the API requires all five.

Responses:

| Status | When | Body |
|---|---|---|
| 200 | valid input | `{"prediction": 0.8191608756158957, "model_version": "hw4-factor-score-v1"}` — `prediction` is finite and in [0, 1]; higher ranks the stock higher that month |
| 422 | any invalid input or malformed JSON | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| 500 | the model raised or returned a non-finite score | `{"detail": "Inference failed"}`; traceback only in the server log |

The 422 body leaves out the rejected value. FastAPI's default handler echoes it, and for `NaN` it cannot be written as JSON, so the default turned a NaN request into a 500 (found while testing. The NaN request in the container check below now gets a 422). `GET /health` returns `{"status": "ok", "model_version": ...}`.

Interactive docs: `http://127.0.0.1:8000/docs`.

## Setup and running

Python 3.12. Versions in `requirements.txt` match my HW4 environment:
scikit-learn 1.8.0, pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, joblib 1.5.3. The template versions already matched, so they are unchanged. The server refuses to start if the artifact's sklearn version differs from the installed one.

```sh
cd Homeworks/HW5/Min_Jin
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test    # python -m pytest -q
make lint    # ruff check + ruff format --check
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

**Artifact.** `artifacts/factor_score.joblib` is committed, so nothing else is needed to run the service. it never retrains or reads the HW4 database. To regenerate it and the reference (needs the HW4 folder, its `outputs/panel.parquet`, and the repo-root course environment from HW4):

```sh
../../../.venv/bin/python export_model.py
```

## Container

```sh
make build   # docker build -t mfe-hw5:local .
make run     # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

The image is `python:3.12-slim` with `/app` as the working directory; it installs
`requirements.txt` and copies `app.py`, `factor_model.py` and `artifacts/`. `.dockerignore` keeps
`.venv`, `.git`, caches, `.env` files, `*.duckdb` and `outputs/` out of the build context.

### Milestone 5: requests to the container, by hand

Host port 8000 must be free. In terminal 1, build and start the container:

```sh
make build
make run
```

In terminal 2, from this folder:

```sh
# 1. Valid request -> 200 with prediction and model_version
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json

# 2. Invalid request (numeric string, raw value > 1, missing field) -> 422
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  --data '{"rank_operating_roa": 0.97, "rank_fcf_to_assets": "0.96", "rank_drawdown_252": 0.77, "rank_momentum_252_21": 1.5}'

# 3. Non-finite value -> 422
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  --data '{"rank_operating_roa": NaN, "rank_fcf_to_assets": 0.96, "rank_drawdown_252": 0.77, "rank_momentum_252_21": 0.71, "rank_volatility_63": 0.33}'

# 4. Real-HTTP latency (uses this folder's .venv for httpx)
python measure_latency.py
```

Then restart the same image: press Ctrl+C in terminal 1 (or `docker stop mfe-hw5`), run
`make run` again, and in terminal 2 repeat:

```sh
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

I copied the commands and their output from both terminals into `evidence/container_demo.txt`.
Terminal 1 also shows the server log, including `Loaded hw4-factor-score-v1` once at each startup.

## Results

**Tests and lint** (`evidence/test_and_lint.txt`): `ruff check` and `ruff format --check` pass;
**18 passed**. The only warning is a Starlette deprecation notice inside FastAPI's `TestClient`
about `httpx`, which the template pins.

| Test group | What it checks |
|---|---|
| Valid prediction | 200, exact response keys, `model_version`; API = direct `model.predict` on the training DataFrame; API = HW4 reference; API = formula by hand. Also key order and the 0/1 bounds |
| Input validation | 11 parametrized bad bodies (missing, empty, extra, misspelled, numeric string, boolean, null, list value, < 0, > 1, list body) → 422. NaN is checked against the running container (Milestone 5). A counting wrapper on the loaded model's `predict` shows none reached inference; a control test shows a valid request does |
| Inference failure | `predict` replaced with one that raises → 500 and the message is not leaked; a NaN score → 500 |
| Performance | 5 warm-up + 50 timed requests |

**Reference result.** `reference_example.json` was written by `export_model.py`, which imports HW4's own `model.py`, takes the real AAPL row on the last signal date (2026-08-31) from the HW4 panel, and scores it with `candidates[FINAL_MODEL]`: **0.8191608756158957**. The test tolerance is 1e-12: the arithmetic is identical.

**Latency** (Apple M1, macOS, Python 3.12.9; in-process `TestClient`, 50 requests after 5 warm-up, timed with `time.perf_counter`): median **1.10 ms**, p95 **1.24 ms**. Over real localhost HTTP to the container (`measure_latency.py`, 200 requests): median 2.18 ms, p95 3.76 ms.

The test asserts p95 < 100 ms: loose enough for slower machines, but it would catch a regression such as reloading the model on every request.

**Container evidence** (`evidence/container_demo.txt`, recorded by hand with the commands above): the example request returns 200 with `prediction 0.8191608756158957` and `model_version hw4-factor-score-v1`; a body with a numeric string, a raw (> 1) value and a missing field returns 422 listing all three problems; a `NaN` body returns 422; after `docker stop` and a new `docker run` of the same image, `/health` and the same request return the same result. The server log shows the model loaded once at startup.

## Before a real deployment

- **The real input is the hard part.** Callers must compute the ranks against the same universe and date as HW4 (eligible top-2000 stocks at the month-end close, filings available by then). A better service would take raw features for the whole month's cross-section, or a ticker and date, and rank them itself, so a caller cannot send ranks from a different universe.
- **Batch endpoint.** The strategy scores ~2,000 stocks per month; one request per stock is wasteful.
- **Missing ranks.** Decide whether to accept `null` and reproduce HW4's "average of the available ranks" behavior, instead of rejecting.
- **Security and operations:** authentication, rate limits, HTTPS, request size limits, a non-root user in the image, structured logs, metrics and alerts on 422/500 rates and on the input distribution (ranks should be roughly uniform each month).
- **Artifacts:** a model registry or checksum instead of a file in the image; pickles run code when loaded, so only trusted, signed files.
- **Image:** split test/lint tools out of the runtime requirements (the image is 540 MB), pin the base image by digest, and run the tests in CI on every change.
- **The model itself** did not beat the S&P 500 TR in HW4; serving it does not make it worth trading.

## Reused code and AI assistance

- `FactorScore` and `FACTOR_SIGNS` are copied from my HW4 `model.py`. The app structure (lifespan
  loading, `app.state`, `TestClient` fixtures) follows the HW5 template and the Lecture 9 demo.
- **AI assistance:** I used Claude Code to help generate directions for completing this project, help write the tests, Dockerfile, measure_latency.py and this README. I wrote the code and used Claude to help identify troubleshooting and checked that the served score matches my HW4 code exactly.
