# hw5 - dev rao

FastAPI service for the model i froze in hw4, with tests, a docker image and request evidence.

## which hw4 model this serves

my hw4 final experiment used the **fixed-sign multi-factor composite**, not ridge or HGB (those lost in validation, see the hw4 README). so that's what is served here.

- model version: `hw4-composite-frozen-2026-11-13`
- source: `Homeworks/HW4/Dev_Rao`, branch `hw_4_dev_rao`, commit `10d27ee`, `frozen_config.json` sha256 `140d39c7ebbb...7487` (recorded in `artifacts/model_metadata.json`)
- what it does: 5 themes (value, quality, investment, momentum, low risk). each theme = mean of `sign * (rank - 0.5)` over its available features, score = mean of the available themes. 0.0 if everything is missing.
- **nothing is fitted.** the weights and signs are fixed (from the literature), so there was no training cutoff and no quarterly refit in hw4. the same model scored every test signal from 2016 to 2026. `fit()` only records the column contract.
- **preprocessing:** the only preprocessing in hw4 is the same-date percentile rank among that date's eligible stocks. that's cross-sectional (it needs the whole universe on that date), not learned from training data, so it can't run on a single-row request. the api takes the already-ranked features, same as the model saw in hw4. in a real system that ranking step would sit in the feature pipeline (see the last section).
- prediction horizon: the score ranks stocks by expected **90-calendar-day forward return** (signal after the close, entry at the next session's close, 90-day hold).

### how the artifact was made

1. `make_reference.py` (repo env, needs duckdb): pulls 4 real rows from the hw4 feature cache and the score **hw4 itself saved** for each row in its final test (`outputs/final_scores.csv`). writes `reference.json` and `example_request.json`. it doesn't compute any scores itself.
2. `export_model.py` (this folder's python 3.12 venv): wraps the hw4 math in `composite.CompositeScorer` (ported from hw4 `hw4/model.py`), puts it in an sklearn `Pipeline`, checks it reproduces all 4 hw4 scores, then saves `artifacts/model.joblib` + `artifacts/model_metadata.json` (version, features in order, versions, sha256).

the artifact is **763 bytes**, so it's committed. i built it under python 3.12 with the pinned versions, so the container never unpickles something made on another python. hw4 itself ran on python 3.13.15 with the same library versions. that doesn't matter for this model because nothing fitted needed to be carried over. the service checks the artifact's sha256 against the metadata before `joblib.load` and refuses to start if they differ.

| case (hw4 final test) | missing features | hw4 score | served | diff |
|---|---:|---:|---:|---:|
| KLAC 2016-03-31 | 0 | 0.273243070 | 0.273243075 | 5.1e-09 |
| AAP 2016-03-31 | 7 | -0.033842556 | -0.033842554 | 2.5e-09 |
| VIAC 2020-02-21 | 2 | 0.026943540 | 0.026943544 | 3.4e-09 |
| A 2026-09-25 | 0 | 0.003452483 | 0.003452483 | 3.0e-10 |

the tiny diffs are because hw4 stored and scored the ranks as float32 and the service uses float64. the tests use a tolerance of 1e-7, about 2 float32 steps near 0.5. as a sanity check i also recomputed the composite for all 918,892 hw4 test rows and it matched hw4's saved scores to 1e-16.

## input / output contract

`POST /predict`, JSON object, **15 required keys**. the order below is the training column order (json key order doesn't matter, the service rebuilds a one-row DataFrame with these exact column names in this order).

every value is a **same-date percentile rank among that date's eligible stocks**, so it's a number in **(0, 1]** (pandas `rank(pct=True)`, the lowest of n stocks gets 1/n > 0). raw values (like a 35% return) are rejected by the range check.

| # | field | theme / sign | underlying hw4 feature | null allowed |
|---:|---|---|---|---|
| 1 | `r_ey` | value + | ttm net income / market cap | yes |
| 2 | `r_bm` | value + | book equity / market cap | yes |
| 3 | `r_sp` | value + | ttm revenue / market cap | yes |
| 4 | `r_fcfy` | value + | ttm free cash flow / market cap | yes |
| 5 | `r_gp_a` | quality + | ttm gross profit / assets | yes |
| 6 | `r_roe` | quality + | ttm net income / equity (equity > 0) | yes |
| 7 | `r_roa` | quality + | ttm net income / assets | yes |
| 8 | `r_accruals_a` | quality - | (ttm NI - ttm operating cash flow) / assets | yes |
| 9 | `r_asset_growth` | investment - | yoy asset growth | yes |
| 10 | `r_share_iss` | investment - | yoy change in shares | yes |
| 11 | `r_mom_12_1` | momentum + | return t-252 to t-21 | yes |
| 12 | `r_sector_mom_126` | momentum + | sector median 126d return | **no** |
| 13 | `r_ni_chg_a` | momentum + | (NI - same quarter last year) / assets | yes |
| 14 | `r_ivol_63` | low risk - | 63d idiosyncratic vol vs ^SP500TR | yes |
| 15 | `r_max_ret_21` | low risk - | max daily return over 21d | yes |

rules and why:
- **all 15 keys are required**, even the nullable ones. leaving a key out is a 422. sending `null` explicitly is how you say "missing".
- **null:** in hw4 these features really were missing sometimes (roe 10.7% of eligible rows, the fundamentals ~7%, and even ivol 62 rows / max_ret 32 rows because of price gaps). the hw4 model handled that by averaging whatever was present, so `null` -> NaN and the model does the same thing. `r_sector_mom_126` was never missing in 1,276,254 eligible rows (a sector median always exists), so a null there means the upstream pipeline broke, and it's rejected.
- strict numbers only: `"0.5"` (string) and `true`/`false` are 422, not silently converted. an integer like `1` is fine (the top-ranked stock has rank exactly 1).
- NaN / Infinity / -Infinity -> 422. extra keys (e.g. `sector`) -> 422. a list instead of an object -> 422.
- no categorical fields: sector isn't a model input (it only enters through `r_sector_mom_126`).

response (200):

```json
{"prediction": 0.27324307470892867, "model_version": "hw4-composite-frozen-2026-11-13"}
```

- `prediction`: unitless composite score in **[-0.5, 0.5]**. higher = higher expected 90-day forward return rank. **it's not a return forecast.** in hw4 the strategy bought the top 25 scores each month. observed range in the hw4 test: -0.47 to 0.40.
- 422: validation errors as `{"detail": [{"type", "loc", "msg"}]}`. i override fastapi's default handler so it doesn't echo the rejected values back. the default body includes each bad `input`, and a NaN input can't be json-encoded, so without this a NaN request crashed into a 500 (the tests caught it).
- 500: `{"detail": "Prediction unavailable"}` plus an `X-Request-ID` header for any unexpected inference error, or if the model returns something non-finite or outside [-0.5, 0.5]. the full traceback is only in the server log (`inference_failed request_id=...`).
- `GET /health` returns `{"status": "ready", "model_version": ...}` (also used by the docker HEALTHCHECK).

## files

```
app.py               request handling: pydantic contract, lifespan, /predict, /health, error handlers
model_service.py     model logic: load + hash-check the artifact, build the one-row frame, predict
composite.py         the hw4 model as an sklearn estimator (needed to unpickle, copied into the image)
artifacts/           model.joblib (763 B) + model_metadata.json
export_model.py      builds the artifact and checks it against reference.json
make_reference.py    pulls reference rows + hw4 scores from the hw4 outputs
reference.json       4 hw4 cases with expected scores
example_request.json KLAC 2016-03-31 request body
tests/test_app.py    33 tests
measure_http_latency.py, container_demo.sh, evidence/
```

## setup and run

python 3.12 venv, from this folder. requirements pin scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3, threadpoolctl 3.6.0, the same as the repo `uv.lock` my hw4 ran with.

windows (powershell):

```powershell
py -3.12 -m venv .venv          # i didn't have 3.12 so i used: uv python install 3.12, then that python.exe -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000     # docs at http://127.0.0.1:8000/docs
```

mac/linux: `python3.12 -m venv .venv && source .venv/bin/activate`, then the same.

`make` isn't installed on my windows machine, so here are the equivalents of each target:

| make | command |
|---|---|
| `make test` | `python -m pytest -q` |
| `make lint` | `python -m ruff check .` then `python -m ruff format --check .` |
| `make format` | `python -m ruff format .` |
| `make build` | `docker build -t mfe-hw5:local .` |
| `make run` | `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` |

reproduce the reference + artifact (optional, the outputs are committed):

```
uv run python Homeworks/HW5/Dev_Rao/make_reference.py      # from the repo root; needs the hw4 outputs/ cache (gitignored, local)
.venv\Scripts\python export_model.py                       # from this folder; asserts all 4 hw4 scores match
```

the service itself never touches the hw4 database. it only loads `artifacts/`.

## tests and lint

`python -m ruff check .` -> All checks passed. `python -m ruff format --check .` -> 8 files already formatted. `python -m pytest -q` -> **33 passed**. full output in `evidence/checks.txt`. the one warning is starlette saying the template's pinned `httpx` is deprecated for TestClient. harmless, and i kept the template pin.

what the tests cover (`tests/test_app.py`):
1. **valid:** 200, exact schema, finite, in [-0.5, 0.5], right model_version. api prediction == a direct `joblib.load` + `pipeline.predict` call (abs 1e-12, same float64 path). the model received one row with the 15 training columns in order. plus all 4 hw4 reference cases (abs 1e-7, see above), including rows with 7 and 2 nulls, null handling, json key order not mattering, and `/health`.
2. **validation -> 422** (15 cases + 3 non-finite + malformed json): missing field, missing non-null field, extra field, 16 features, empty body, list instead of object, numeric string, boolean, text, nested list, null where not allowed, rank > 1, rank = 0, negative, raw value. NaN/Infinity/-Infinity are sent as raw json text (json.dumps can't write them). every one of these uses a spy on the loaded pipeline's `predict` and asserts it was **never called**.
3. **failure -> 500:** valid input with the loaded model's `predict` monkeypatched to raise. asserts 500, the error text isn't leaked, an X-Request-ID is present, and `inference_failed` was logged. also: a model returning NaN, inf or 0.9 gets a 500, not a 200.
4. **performance:** below.

## ci (optional extension)

`.github/workflows/hw5-dev-rao.yml` (repo root, since that's where github looks), adapted from the lesson workflow `.github/workflows/lesson-9.yml`. it runs on pushes/PRs that touch `Homeworks/HW5/Dev_Rao/**` or the workflow itself, and can also be started by hand.

- `checks`: python 3.12, `pip install -r requirements.txt` (same pins as here), `ruff check`, `ruff format --check`, `pytest`.
- `container` (after checks pass): `docker build`, start the container, wait for `/health`, valid request must be 200 with the hw4 model version, a bad request must be 422, and docker has to report the container `healthy`. container logs get printed either way.

run on my fork (both jobs green, commit `2c87fff`): https://github.com/DNR07/UCB-MFE-python-preprogram-2026-winter/actions/runs/44025903481

## latency

my machine: Intel Core Ultra 7 256V (8 cores), 16 GB RAM, Windows 11, Python 3.12.9, Docker Desktop on WSL 2. sequential requests with `time.perf_counter`, after warm-up.

| measurement | n | median | p95 | max |
|---|---:|---:|---:|---:|
| in-process `TestClient` (no network, no uvicorn), `test_predict_performance` | 200 | 4.2 ms | 5.1 ms | 35.9 ms |
| **real localhost HTTP** to the docker container, httpx keep-alive, `measure_http_latency.py` | 200 | 5.3 ms | 6.5 ms | 11.4 ms |

files: `evidence/latency_testclient.json`, `evidence/latency_http_container.json`. in-process numbers move between runs (5.6 ms median / 9.0 ms p95 on another run with `-v`). most of the time is pandas building a one-row DataFrame plus FastAPI/pydantic overhead, not the model. the test asserts p95 < 100 ms. that's ~10-20x what i measure, and it's only there to catch something pathological like reloading or refitting the model on every request. it's not a real SLA.

## docker

```
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

the image is `python:3.12-slim` with workdir `/app` and the pinned requirements. it copies only `app.py`, `model_service.py`, `composite.py` and `artifacts/`, runs as a non-root user, and has a HEALTHCHECK on `/health`. `.dockerignore` keeps out `.venv`, `.git`, caches, `*.duckdb`/`*.db`, `.env*`, keys, tests, evidence and the export scripts. the image is 722 MB, mostly numpy/pandas/scipy/sklearn. pytest/ruff/httpx are in there too because the template has one requirements file. a separate runtime requirements file would trim it.

to reproduce on another machine: clone, `cd Homeworks/HW5/Dev_Rao`, `docker build -t mfe-hw5:local .`. nothing outside this folder is needed, since the artifact is committed.

## container evidence

`evidence/container_transcript.txt`, made with `bash container_demo.sh > evidence/container_transcript.txt 2>&1` (git bash). same commands by hand:

```
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

what's in it:
1. valid request (KLAC 2016-03-31) -> `HTTP/1.1 200 OK`, `{"prediction":0.27324307470892867,"model_version":"hw4-composite-frozen-2026-11-13"}` (hw4 had 0.2732431)
2. invalid (`"r_ey": "0.5"` + extra `sector`) -> `422`, lists the type error, the missing fields and the extra field
3. NaN in an otherwise valid body -> `422` "Input should be a finite number"
4. `docker stop`, then `docker run` the same image again -> `/health` 200
5. same valid request after the restart -> 200, identical prediction and model_version
6. container log showing `loaded model hw4-composite-frozen-2026-11-13` at startup

## before a real deployment

- **feature pipeline:** the model needs same-date cross-sectional ranks, so something upstream has to compute the raw features from filed financials and prices (with the hw4 timing rules: filing + 1 day, next-session entry), rank them across the whole eligible universe for that date, and only then call this api. honestly a batch endpoint that takes the whole universe and does the ranking would fit this model better than one row at a time. that step would also need its own tests for training/serving skew.
- **model registry / versioning:** keep the artifact + metadata (sha256, versions, hw4 config hash) in a registry, put the version in the image tag, and keep a way to roll back.
- **monitoring and drift:** log the input distributions (each rank should look ~uniform across the universe, so a shift means a pipeline bug), null rates per feature, score distribution, latency, 4xx/5xx rates. and track realized 90-day IC / top-25 vs universe once the labels mature, since that's the only real check on the model.
- **auth and rate limiting:** right now anyone who can reach the port can call it. add api keys/oauth behind a gateway, TLS, and rate limits.
- **input logging and privacy:** log request ids and features for debugging/reproducibility, but the inputs here are derived from public data, so the main concern is retention policy and not logging secrets.
- **retraining cadence:** this model has no learned parameters, but its signs and themes were chosen once on 2010-2015. review it on a schedule (e.g. yearly) with the hw4 walk-forward process, and version any change.
- **health/readiness:** `/health` exists. split it into liveness vs readiness for kubernetes, and fail readiness if the artifact didn't load.
- **resources and scaling:** set cpu/memory limits (it's small: `docker stats` showed ~105 MiB after a request), run multiple uvicorn workers or replicas behind a load balancer, add request timeouts, and use a smaller runtime image without the dev tools.
- **dependency pinning:** unpickling sklearn objects is only supported on the same versions, so the image pins them and the metadata records them. a startup check that compares installed versions with the metadata would be a good next step.

## citations

- `app.py` structure (strict pydantic contract, lifespan load, request id + logged exception on failure, `/health`) and the Dockerfile are adapted from `Lectures/Lecture 9/demo/app.py` and `demo/Dockerfile`.
- `composite.py` ports `COMPOSITE_THEMES` and `composite_score` from my hw4 `Homeworks/HW4/Dev_Rao/hw4/model.py` (branch `hw_4_dev_rao`).
- the starter files (Makefile, pyproject, test fixtures) come from `Homeworks/HW5/template/`.
