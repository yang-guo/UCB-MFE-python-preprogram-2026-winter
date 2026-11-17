# HW5 — Serving my HW4 stock-return model

Guillaume Estepa

A FastAPI service that serves the Ridge model of my HW4 experiment, with tests, a
Docker image, and evidence of requests to the running container.

| Checklist item | Where |
|---|---|
| Application, tests, Dockerfile, Makefile, dependency versions | `app.py`, `inference.py`, `tests/test_app.py`, `Dockerfile`, `Makefile`, `requirements.txt` |
| Contract, setup, artifact loading, commands | this README |
| Saved HW4 model and example input | `artifacts/`, `example_request.json` |
| Test and lint output, latency report | `evidence/checks.txt`, [Tests](#tests) |
| Valid and invalid requests to the container, request after restart | `evidence/container_demo.txt`, [Container](#container) |
| What needs attention before a real deployment | [Before a real deployment](#before-a-real-deployment) |

My HW4 is not merged into `main` yet, so the links to HW4 files below point to
its commit.

## What is served

**Model version `hw4-ridge_rank-2026-01-05`**: the `ridge_rank` model chosen in my
HW4 ([PR #248][hw4], commit `68d7d59`), as fitted on 2026-01-05. HW4 refits every
January; this is the last of its 11 walk-forward fits ([`fit_log.csv`][fit_log]),
the one that scored every signal date from 2026-01-05 to 2026-09-28.

| | |
|---|---|
| Estimator | `Ridge(alpha=1000.0)`, scikit-learn 1.8.0 |
| Training rows | 1,656,617 (signal dates 2010-01-04 → 2025-10-01, outcomes ended by 2025-12-31) |
| Features | 24, each a same-date percentile rank (table below) |
| Target | 90-calendar-day forward return as a same-date percentile rank, clipped at its training 1%/99% quantiles |
| Preprocessing | missing rank → 0.0, the middle rank; nothing is fitted |

**Preprocessing.** HW4 has no fitted scaler or imputer: every feature is ranked
within its own date, and the only step before the Ridge is `design_matrix` in
[`model.py`][model] (training column order, float32, missing rank → 0.0). The
artifact is a scikit-learn `Pipeline` with that rule as a constant
`SimpleImputer` in front of the Ridge object HW4 fitted. There is no custom
transformer, so no extra module to ship.

**What the service does not do.** Ranking a feature needs every eligible stock of
that day (about 1,900), which a one-stock request cannot carry. The request
therefore holds the ranks, which is exactly what the model was trained on, and
computing them stays in the batch job that builds the panel (`build_panel` in
[`features.py`][features]). A service taking raw values would have to transform
them differently from training.

## Contract

### Request: `POST /predict`

A JSON object with exactly these 24 keys, sent with
`Content-Type: application/json`. Key order is free: the service puts the values
in the training order below.

| # | Field | Quantity that is ranked | `null` allowed | Missing in training |
|---:|---|---|:---:|---:|
| 1 | `mom_21` | Return over the last 21 sessions | no | 0% |
| 2 | `mom_63` | Return over the last 63 sessions | yes | <0.1% |
| 3 | `mom_126` | Return over the last 126 sessions | yes | <0.1% |
| 4 | `mom_252_21` | Return from 252 to 21 sessions ago | no | 0% |
| 5 | `vol_63` | Standard deviation of daily returns, 63 sessions | no | 0% |
| 6 | `beta_252` | Beta to the S&P 500 total return, 252 sessions | yes | <0.1% |
| 7 | `max_ret_21` | Largest daily return in the last 21 sessions | yes | <0.1% |
| 8 | `drawdown_252` | Price / 252-session high − 1 | yes | <0.1% |
| 9 | `ma_gap_200` | Price / 200-session average − 1 | yes | <0.1% |
| 10 | `dollar_vol_63` | Median daily dollar volume, 63 sessions | no | 0% |
| 11 | `volume_trend` | Mean dollar volume, 21 / 252 sessions | yes | <0.1% |
| 12 | `net_margin` | Net income / revenue | yes | 3.1% |
| 13 | `operating_roa` | Operating income / total assets | yes | 1.7% |
| 14 | `gross_profitability` | Gross profit / total assets | yes | 1.7% |
| 15 | `debt_to_assets` | Total debt / total assets | yes | 1.7% |
| 16 | `cash_to_assets` | Cash and equivalents / total assets | yes | 1.7% |
| 17 | `fcf_to_assets` | Free cash flow / total assets | yes | 1.9% |
| 18 | `accruals_to_assets` | (Net income − operating cash flow) / total assets | yes | 1.9% |
| 19 | `revenue_growth` | Revenue / revenue four quarters ago − 1 | yes | 13.1% |
| 20 | `asset_growth` | Total assets / four quarters ago − 1 | yes | 11.3% |
| 21 | `earnings_yield` | Trailing-12-month net income / market value | yes | 8.3% |
| 22 | `fcf_yield` | Trailing-12-month free cash flow / market value | yes | 8.8% |
| 23 | `book_to_market` | Total equity / market value | yes | 1.7% |
| 24 | `filing_age_days` | Days since the latest statement became usable | yes | 1.5% |

Returns and price levels use adjusted closes; dollar volume and market value use
the unadjusted close. Fundamentals come from the latest quarterly statement,
usable from its filing date + 1 day. Full definitions are in HW4
[`features.py`][features].

- **Type.** JSON number. Strings (`"0.1"`) and booleans are rejected, not converted.
- **Unit.** Each value is `rank(pct=True) - 0.5` of the quantity among the stocks
  eligible at the close of the signal date: from just above −0.5 for the lowest
  stock to +0.5 for the highest, no unit.
- **Range.** [−0.5, 0.5] for every field, because a centred percentile rank
  cannot fall outside it. This catches gross mistakes only, such as a dollar
  volume (`2.5e9`), a number of days or a percentage. A raw return of 0.03 looks
  like a valid rank, and a one-stock service cannot tell the difference.
- **`null`.** Allowed for the 20 features HW4 could not always compute. The
  pipeline scores it as the middle rank, as in training. The key must still be
  present, so a forgotten field is never imputed silently. The other 4 features
  are never missing in HW4, because its eligibility screen requires them; `null`
  there means the stock was not eligible, and is rejected.
- Non-finite numbers (`NaN`, `Infinity`, `1e999`), missing keys, extra keys, and a
  list in place of the object are rejected.

**Example input** (`example_request.json`): NVDA at the 2026-09-28 close, the last
signal date of HW4. The backtest bought it the next day (lot 6299 in HW4
[`results/lots.csv`][lots]).

### Response

```json
{"prediction": 0.04386884346604347, "model_version": "hw4-ridge_rank-2026-01-05"}
```

`prediction` is the predicted 90-calendar-day return of a purchase at the next
close, as a centred percentile rank among that day's stocks (−0.5 worst, +0.5
best). It is not a return in percent, and only orders stocks of the same date.
Over the 79,925 rows this fit scored in HW4 it has mean 0.000, standard deviation
0.020 and range −0.104 to +0.060. The value above is the container's answer to
the example; my laptop returns `0.04386885091662407` (see Tolerance below).

| Status | When | Body |
|---|---|---|
| 200 | Valid input | `prediction` (finite) and `model_version`; header `X-Request-ID` |
| 422 | Missing or extra field, wrong type, out of range, non-finite, wrong shape, malformed JSON, wrong content type | `{"detail": [{"loc", "msg", "type"}]}`: the field and the reason, not the value |
| 400 | Body that cannot be decoded at all (invalid UTF-8, a number thousands of digits long) | `{"detail": "There was an error parsing the body"}` (FastAPI's own answer) |
| 500 | Inference raised, or the model returned a non-finite number | `{"detail": "Prediction unavailable"}` and `X-Request-ID`; the traceback is only in the server log, under that id |

`GET /health` returns `{"status": "ready", "model_version": ...}`. The Docker
health check and the restart demo use it.

## Setup and run

```bash
cd Homeworks/HW5/Guillaume_Estepa
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test     # python -m pytest -q
make lint     # python -m ruff check .  and  python -m ruff format --check .
make serve    # python -m uvicorn app:app --host 127.0.0.1 --port 8000

curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

Interactive documentation: <http://127.0.0.1:8000/docs>.

**Dependency versions.** `requirements.in` lists the direct dependencies.
`requirements.txt` is compiled from it and also pins everything they pull in. The
model libraries are the ones HW4 was trained with (the template's defaults already
matched the course `uv.lock`):

| | HW4 training | Service | Docker image |
|---|---|---|---|
| Python | 3.12.9 | 3.12.9 | 3.12.16 |
| scikit-learn | 1.8.0 | 1.8.0 | 1.8.0 |
| numpy | 2.3.5 | 2.3.5 | 2.3.5 |
| scipy | 1.16.3 | 1.16.3 | 1.16.3 |
| joblib | 1.5.3 | 1.5.3 | 1.5.3 |
| threadpoolctl | 3.6.0 | 3.6.0 | 3.6.0 |

I removed pandas from the template list: HW4 used it to build the features, but
the saved pipeline only needs numpy and scikit-learn.

## The artifact

`artifacts/hw4_ridge_rank.joblib` (1.9 KB, in Git) holds the pipeline, its version
and its feature order. `artifacts/model_metadata.json` describes it in plain text:
training window, library versions, checksum, and the checks below.

**How it was made.** `export_hw4_model.py`, run once in the HW4 environment:

1. checks that the database is the course file, then imports my HW4 modules
   unchanged and rebuilds the panel and the 11 fits;
2. checks that this re-run is the HW4 experiment: fit log identical to HW4
   [`fit_log.csv`][fit_log], coefficients equal to
   [`ridge_coefficients.csv`][coefficients] to its five decimals;
3. takes the Ridge fitted on 2026-01-05 as it is, without refitting, and puts
   the missing-rank rule in front of it;
4. checks that the pipeline returns exactly the predictions HW4 made with that
   fit, on all 79,925 rows;
5. saves three real rows with HW4's prediction for each
   (`tests/data/hw4_reference.json`), the first also as `example_request.json`.

The script stops before writing anything if one of these checks fails. To redo
it, with my HW4 folder (PR #248) and the bCourses database in
`Lectures/modeling/data/`, from the repository root:

```bash
uv sync --locked
uv run python Homeworks/HW5/Guillaume_Estepa/export_hw4_model.py
```

Repeated runs gave a byte-identical artifact.

**How it is loaded.** Once per server process, in the FastAPI lifespan (`app.py`
calls `inference.load_model`). The path is resolved from `inference.py`, not from
the working directory. Nothing is fitted at startup or on a request, and the HW4
database is not needed. Unpickling can run code, so before it the loader checks
that the file's SHA-256 is the one in the metadata and that scikit-learn is the
version that fitted the model. Startup also fails if the request fields are not
the features of the saved model, in the same order.

## Tests

`make test`: **63 passed**. `make lint`: clean. Output in `evidence/checks.txt`.

| Group | What is checked |
|---|---|
| Valid prediction | 200 and response schema; API equal to a direct call of the saved pipeline; API equal to HW4's own prediction on three real rows; `null` scored as the middle rank; `null` accepted on exactly the 20 fields where HW4 had missing values; key order free; bounds inclusive |
| Input validation | 12 invalid bodies, 4 non-finite numbers and malformed JSON give 422; two undecodable bodies give 400. A spy on the loaded model shows that none of them reached inference (the same spy counts one call for a valid request) |
| Inference failure | `predict` replaced by a function that raises: 500, generic body, nothing internal in the response, traceback in the log under the request id, and the next request works. A NaN from the model is also a 500 |
| Performance | 200 timed requests after 20 warm-up; median and p95 printed in the test output |
| Model loading | Loaded once per process. After startup, loading or fitting anything is made to raise, and requests still succeed. Startup fails if schema and model disagree; a file with another checksum or another scikit-learn version is refused |

**Reference result.** `tests/data/hw4_reference.json` was written by the export
script from `panel["prediction"]`, the output of HW4's own
`walk_forward_predict`. These are the numbers the HW4 backtest traded on,
computed before the pipeline object or the API existed. The rows are NVDA
2026-09-28 (complete, bought), HAS 2026-01-20 (four fundamentals missing, bought)
and AMLX 2026-09-28 (lowest prediction that day, two missing).

**Tolerance.** HW4 scored the 79,925 rows of this fit as one float32 batch; the
service scores one float32 row at a time. The two differ by at most 1.5e-8 over
those rows. I compare with an absolute tolerance of 1e-6: room for a different
BLAS in the container, and still far below a real mistake (the 2025 fit moves the
NVDA prediction by 1.3e-3, swapping two features by 4e-4). For NVDA the container
returns HW4's stored value to the last digit and my laptop is 7e-9 away. The
API-versus-pipeline comparison is exact, because the API adds no arithmetic.

**A bug found while testing.** A `NaN` in the body must be a 422. With FastAPI's
default handler it was a 500: Python's JSON parser accepts `NaN`, validation
rejects it, and the default error response then tries to echo the value back,
which cannot be written as JSON. `app.py` replaces that handler; four tests send
`NaN`, `Infinity`, `-Infinity` and `1e999` as raw bodies.

**Do the tests fail when they should?** `python mutation_check.py` copies the
service to a temporary folder, breaks it in one way, and runs the tests there,
for 18 different bugs: reversed feature order, missing rank → 0.5, float64 row,
strict mode off, extra fields allowed, no bounds, an error answered with a 0.0
prediction, exception text sent to the client, a model loaded or fitted on each
request, and others. All 18 are caught (`evidence/mutation_check.txt`). The
first versions of the tests missed three of them, and each got a test.

### Latency report

| Measurement | Median | p95 |
|---|---:|---:|
| In-process `TestClient`, no socket (`test_predict_performance`, on the laptop) | 0.95 ms | 1.04 ms |
| The same test inside the Linux image | 1.15 ms | 1.30 ms |
| Real localhost HTTP into the container (`curl`, new connection per request) | 1.4 ms | 1.7 ms |

Each line is 200 sequential requests after 20 warm-up requests, one client. The
first two are timed with `time.perf_counter` around each request, the third with
curl's `time_total`. Hardware: MacBook with Apple A18 Pro (6 cores), 8 GB RAM,
macOS 26.6, Python 3.12.9; the container runs in a Colima VM (2 vCPUs, 2 GB,
linux/arm64, Docker Engine 29.5.2). Runs vary: the in-process p95 was between 1.0
and 2.7 ms across my runs.

The test asserts p95 < 50 ms. This is a tripwire, not a service target: HW4
decides after the close and trades at the next close, so nobody waits on one
request. It is loose enough for a slow CI machine and fails if a request starts
taking tens of milliseconds. It cannot see cheap extra work, which is why loading
or fitting during a request has its own test.

## Container

```bash
make build    # docker build -t mfe-hw5:local .
make run      # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

Then send the same `curl` request as above from another terminal. Stop with
Ctrl-C or `docker stop mfe-hw5`. If host port 8000 is taken, change the left
side of `-p`.

The image is `python:3.12-slim` plus `requirements.txt`, `app.py`,
`inference.py` and `artifacts/`. It runs as a non-root user and has a health
check on `/health`. `.dockerignore` is an allow-list, so `.venv`, `.git`, `.env`
files, databases and tests never enter the build context.

`./container_demo.sh` runs the whole Milestone 5 sequence and stops if a status
code is not the expected one. Its transcript is `evidence/container_demo.txt`;
the main lines:

```text
$ docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
$ docker inspect mfe-hw5 --format '{{.Image}}'
sha256:9f400ea2d311babdcab22d7cecf561d488af9133874dff5409a5170baded2888

$ curl -si http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
x-request-id: 8874ecf7431a
{"prediction":0.04386884346604347,"model_version":"hw4-ridge_rank-2026-01-05"}

$ sed 's/"dollar_vol_63": 0.49947286/"dollar_vol_63": 2500000000/' example_request.json | curl -si http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @-
HTTP/1.1 422 Unprocessable Entity
{"detail":[{"loc":["body","dollar_vol_63"],"msg":"Input should be less than or equal to 0.5","type":"less_than_equal"}]}

$ docker stop mfe-hw5
$ docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
$ docker inspect mfe-hw5 --format '{{.Image}}'
sha256:9f400ea2d311babdcab22d7cecf561d488af9133874dff5409a5170baded2888

$ curl -si http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
x-request-id: cf3483eabf43
{"prediction":0.04386884346604347,"model_version":"hw4-ridge_rank-2026-01-05"}
```

The full transcript also has the other response headers, two more rejected
requests (a number sent as a string, a `NaN`) and the container log.

The 63 tests also pass inside the image, with `tests/` mounted read-only (last
command of `evidence/checks.txt`). That is the check that the Linux build, with
a different BLAS from my laptop, reproduces the HW4 numbers.

## Before a real deployment

- **The features are computed outside the service.** It cannot tell whether the
  ranks came from the right day, the full cross-section, or an eligible stock,
  and its range check only catches gross mistakes. Next step: an endpoint that
  takes the whole day's raw values and ranks them itself with the HW4 code, plus
  the data checks HW4 lacked (frozen prices, unadjusted splits).
- **The model.** It did not beat the S&P 500 in HW4, so this is an engineering
  exercise and not a signal to trade. It is also frozen at the January 2026 fit;
  HW4 refits yearly, so a release process is needed: new artifact, new version,
  new reference rows, and a way to roll back.
- **Artifact trust.** The checksum sits next to the artifact, so it catches a
  mix-up but not someone who can change both files. A real setup would pull a
  signed artifact from a model registry.
- **Exposure.** No authentication, TLS, rate limit or body-size limit; a large
  invalid body also gets a large error response, one entry per bad field. The
  container is published on 127.0.0.1 only and should sit behind a gateway.
- **JSON details.** If a key appears twice, Python's parser silently keeps the
  last one. A strict parser should reject that.
- **Operations.** One uvicorn worker; no metrics, alerts or load test. I would
  log latency and model version per request and monitor the share of `null`
  inputs and the distribution of predictions against the HW4 range above.
- **Image.** One requirements file puts pytest and ruff in the runtime image
  (631 MB). Split runtime and development dependencies, pin the base image by
  digest, and scan it.
- **Dependencies.** Starlette announces that its test client will need `httpx2`
  instead of the template's `httpx`; I silenced that one warning in
  `pyproject.toml` and left the pin.
- **CI.** I ran the checks on my laptop and inside the image, not in GitHub
  Actions (I did not do the optional extension). A workflow should run lint,
  tests, the image build and `container_demo.sh` on every pull request.

## Sources and AI assistance

- Started from the HW5 template. Adapted from the Lesson 9 demo
  (`Lectures/Lecture 9/demo/`): the lifespan loader, the strict and finite model
  settings, `/health`, the request id, the 500 handling, the artifact layout
  (`pipeline`, `model_version`, `features`) with its metadata file, and the idea
  of patching `predict` in tests.
- `export_hw4_model.py` imports my HW4 code (PR #248) and reuses the first steps
  of `main` in its [`run_final.py`][run_final].
- I used an AI coding assistant (Claude Code) to help write and test the code and
  draft this write-up; I reviewed the design, results and checks.

[hw4]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/pull/248
[features]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/68d7d59aa3645c9e575afc3452a96a955aef0315/Homeworks/HW4/Guillaume_Estepa/features.py
[model]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/68d7d59aa3645c9e575afc3452a96a955aef0315/Homeworks/HW4/Guillaume_Estepa/model.py
[run_final]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/68d7d59aa3645c9e575afc3452a96a955aef0315/Homeworks/HW4/Guillaume_Estepa/run_final.py
[fit_log]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/68d7d59aa3645c9e575afc3452a96a955aef0315/Homeworks/HW4/Guillaume_Estepa/results/fit_log.csv
[coefficients]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/68d7d59aa3645c9e575afc3452a96a955aef0315/Homeworks/HW4/Guillaume_Estepa/results/ridge_coefficients.csv
[lots]: https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/68d7d59aa3645c9e575afc3452a96a955aef0315/Homeworks/HW4/Guillaume_Estepa/results/lots.csv
