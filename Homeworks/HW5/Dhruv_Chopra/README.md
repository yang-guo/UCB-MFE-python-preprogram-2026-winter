# HW5: Serving my HW4 stock-ranking model (Dhruv Chopra)

A FastAPI `POST /predict` service for the **HW4 v1 model**, which is the model behind my submitted HW4 strategy. It is tested with pytest, packaged with Docker, and checked with real requests to the running container.

**Where HW4 is:** HW4 isn't merged into `main` yet. Its code and results are on the [`hw_4_Dhruv_Chopra`](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/hw_4_Dhruv_Chopra/Homeworks/HW4/Dhruv_Chopra) branch (commit `81713cf`). That is where the `Homeworks/HW4/Dhruv_Chopra/...` paths below point.

## What is served

| | |
|---|---|
| Model | HW4 v1 `RidgeOnRanks` (`Homeworks/HW4/Dhruv_Chopra/model.py`): Ridge, α = 10, on 20 centred percentile-rank features plus 20 missing-value indicators |
| Version | `hw4-v1-ridge-a10-fit-2026-07-01`: the last quarterly refit of the submitted strategy (`pipeline.CHOSEN`), fitted at the first session of 2026-Q3 on 1,531,222 matured weekly rows (matches `results/final_model_refits.csv`) |
| Target | Percentile rank, within the signal date, of the stock's 90-calendar-day forward return. A higher prediction means the stock is expected to rank higher. HW4 bought the top 30. |
| Artifact | `artifacts/hw4_ridge_v1.joblib` (1.6 KB, committed): `{"pipeline", "model_version", "features", "fit_date", "n_train", "sklearn_version"}` |
| Versions | scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, joblib 1.5.3: the same as the HW4 training environment (repo `uv.lock`), pinned in `requirements.txt` |

### Preprocessing and the custom module

HW4 ranks each raw feature **across all eligible stocks on the same date** (`pandas.rank(pct=True)`). This step can't run on a single request, because one stock's rank depends on the other ~1,900 stocks that day. It is also not learned: there is no scaler or other fitted statistic. So the API contract is the **ranked** feature vector. This is the representation the Ridge was trained on, and the representation HW4's `ModelSchedule.scores` passes to `predict`.

The fixed design step that follows (rank − 0.5; a missing value becomes 0 plus an indicator) lives in `ranks_model.design`. The artifact is an sklearn `Pipeline([FunctionTransformer(design), fitted Ridge])`. The pickle refers to `ranks_model.design` by name, so `ranks_model.py` is copied into the Docker image.

### How the artifact was produced

`export_hw4_model.py` runs **once**, in the HW4 environment and against the HW4 database. The service never imports it and never touches the database.

```sh
# From this folder, with the HW4 code checked out (branch hw_4_Dhruv_Chopra) and
# equities_top2000.duckdb in Lectures/modeling/data/ (~12 min, mostly feature building):
../../../.venv/bin/python export_hw4_model.py --hw4 ../../HW4/Dhruv_Chopra
```

The export script:

1. calls HW4's own `model.fit("ridge", panel, fit_t, FEATURES)` at 2026-07-01, which is the same call `ModelSchedule` makes;
2. scores HW4's last signal date (2026-09-28) with HW4's `cross_section` + `RidgeOnRanks.predict`, and checks that the new sklearn pipeline gives the same predictions for all 1,944 eligible stocks (atol 1e-6);
3. writes the artifact, `tests/hw4_reference.json` (4 real rows with HW4's predictions) and `example_request.json` (AAPL).

As a check that this is the model that actually traded: the highest score on 2026-09-28 is **0.546341**, which equals `score_best` for that date in HW4's `results/final_cohorts.csv`.

## API contract

`POST /predict` with a JSON object of **exactly these 20 fields, in this order** (the HW4 `features.FEATURES` order, which the code checks):

`mom_5, mom_21, mom_63, mom_126, mom_252_21, vol_63, max_ret_21, drawdown_252, log_dollar_vol_63, gross_profitability, operating_roa, roe, net_margin, debt_to_assets, cash_to_assets, fcf_to_assets, accruals_to_assets, revenue_growth_yoy, asset_growth_yoy, filing_age_days`

Definitions are in `ranks_model.py` and HW4 README §2. Rules for each field:

| Rule | Why |
|---|---|
| A number with 0 < value ≤ 1, or `null` | Each field is a percentile rank: `rank(pct=True)` gives values in (1/n, 1]. Values outside that range, such as a raw `filing_age_days` of 45, are a sign that unranked features were sent, so they are rejected. |
| `null` is allowed, but the field must be present | Fundamentals are often missing (no recent filing). HW4 trained with a missing indicator for this case, so `null` is a valid value for the model. Leaving the field out is treated as a client error. |
| Strict types: no `"0.5"` strings, no `true`/`false` | A numeric string or a boolean should never silently become a feature value. JSON integers are accepted (`1` means the top rank). |
| No NaN/±Infinity | Missing values must be sent as an explicit `null`. |
| Extra fields are rejected | Together with "all 20 required", this enforces exactly 20 features. |

Response: `{"prediction": <finite float>, "model_version": "hw4-v1-ridge-a10-fit-2026-07-01"}`.

| Status | When |
|---|---|
| **200** | Valid input |
| **422** | Any rule above is broken. The model is not called; tests confirm this with a spy. |
| **500** | Inference raises an error or returns a non-finite value. The body is the generic `{"detail": "Prediction unavailable"}`. The traceback goes to the server log, tagged with an `X-Request-ID`. |

`GET /health` returns `{"status": "ready", "model_version": ...}`. The docs page is at `http://127.0.0.1:8000/docs`.

Example (`example_request.json`): AAPL's ranks on 2026-09-28. Expected response: `0.532340808776408`, which HW4 computed as 0.532341.

## Reproduce

Python 3.12, from this folder:

```sh
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

make test        # python -m pytest -q
make lint        # ruff check + ruff format --check
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # local server

make build       # docker build -t mfe-hw5:local .
make run         # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  --data-binary @example_request.json

./demo_container.sh | tee evidence/container_transcript.txt   # full Milestone 5 demo
```

The image is built from `python:3.12-slim` and contains `requirements.txt`, `app.py`, `ranks_model.py` and the artifact only. It runs as a non-root user. `.dockerignore` keeps `.venv`, `.git`, tests, `.env*`, databases and logs out of the build context. The artifact is in Git, so anyone with the repo can build the image without the database.

## Tests (`tests/test_app.py`, 25 cases)

1. **Valid prediction** (4 real HW4 rows: AAPL, MSFT, JPM, and AAL, which has a missing feature):
   - checks the response is 200 with exactly `{prediction, model_version}` and a finite float;
   - checks the API matches a direct `pipeline.predict` on the same DataFrame (1e-12);
   - checks the API matches the prediction **recorded from HW4's own `RidgeOnRanks.predict`** (atol 1e-6). The only expected difference is float32 (HW4) vs float64 (JSON) rounding in `rank − 0.5`, which is about 1e-8.
2. **Input validation** (16 cases): a missing field, an extra field, 19 features, a fixed-length list, an empty body, a numeric string, a boolean, text, a list value, ranks of 0, −0.2, 1.5 and 45, and NaN/Infinity/−Infinity. Each must return 422, and a spy on the loaded pipeline's `predict` must record **zero calls**. A control test shows the spy does record a valid call.
3. **Inference failure**: `monkeypatch` replaces the loaded pipeline's `predict` with one that raises. A valid request must return 500, and the exception text must not leak into the response. A second case returns NaN from the model, which must also be a 500 and not a "successful" prediction.
4. **Performance**: 5 warm-up requests, then 100 timed requests with `time.perf_counter`.

**Bug found by the tests:** FastAPI's default 422 handler echoes the bad input back in the response. With `NaN` this fails JSON encoding, so a correctly rejected request turned into a server error. `app.py` now has a 422 handler that turns non-finite inputs into strings.

### Latency

**In-process `TestClient`** (no network, no uvicorn), measured on an Intel Core i5-1030NG7 @ 1.10 GHz, 8 GB RAM, macOS 15.7, Python 3.12.9:

| Run | Median | p95 |
|---|---:|---:|
| `make test` run (`evidence/test_and_lint.txt`) | 7.20 ms | 11.76 ms |
| earlier run | 5.05 ms | 7.54 ms |

The test asserts **median < 50 ms**. The model itself is one 1×40 dot product, so nearly all of the time is FastAPI/Pydantic overhead. 50 ms leaves room for a slower CI machine but would still catch a regression such as reloading the artifact on each request. Real localhost HTTP through Docker would add network and serialization time and was not benchmarked.

## Evidence

- `evidence/test_and_lint.txt`: `make test` (25 passed), latency output, `make lint` (clean).
- `evidence/container_transcript.txt`: the container returns 200 with prediction 0.532341 and the model version; a numeric string gets 422; 2 of 20 features gets 422; `docker stop`, then a new container from the same image gets 200 with the same prediction; the container log shows the model loaded once at startup.

## Before a real deployment

- **Feature pipeline in production.** Clients have to send same-day cross-sectional ranks. In practice the service, or a batch job upstream of it, should compute raw features and ranks from a point-in-time data store, using the same code as training. Otherwise training and serving can drift apart. A batch endpoint that scores the whole universe at once also fits the use case better than one stock per request.
- **Model lifecycle.** HW4 refits every quarter. That needs a model registry, a record of the training data and code version for each artifact, a shadow/compare step before switching versions, and rollback. Signed or hash-checked artifacts matter too, because loading a pickle runs code.
- **Monitoring.** Track the distribution of input ranks and missing values, the distribution of predictions, and latency and error rates. Most importantly, track realised rank IC once 90-day outcomes are known. HW4 showed that the edge decayed out of sample.
- **Operations.** Authentication and rate limiting, TLS, structured logs, a pinned base-image digest and vulnerability scanning, several workers or replicas behind a health check, and resource limits.

## Code reuse and AI assistance

- The structure of `app.py` (lifespan loading, `FiniteFloat` output, logged generic 500 with a request ID) is adapted from the Lesson 9 demo (`Lectures/Lecture 9/demo/app.py`).
- `ranks_model.design` is copied from my HW4 `RidgeOnRanks._design`. The export calls HW4's code directly.
- I used Claude Code (an AI assistant) to help write the service, tests, export script and this README. I checked the results against HW4's own outputs (refit row count, best score on 2026-09-28, the 4 reference predictions) and against the container's responses.
