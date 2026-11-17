# HW5: HW4 90-day return model

The service serves one frozen HW4 fit. It does not retrain, and it does not open the HW4 database.

## Model version

`hw4-hgb-deep-static-2016-04-01`

This is the static deep `HistGradientBoostingRegressor` recorded in `Homeworks/HW4/Dhruv_Gupta/results/chosen_model.json` (`config_name: deep`, `retrain_rule: static`, `score_sign: 1`). HW4 fit that pipeline once, on the first market session on or after 2016-04-01, which is 2016-04-01. The original experiment did not pickle the estimator. `artifacts/model.joblib` is that same pipeline, refit from `Homeworks/HW4/Dhruv_Gupta/outputs/panel.parquet` and saved with its learned preprocessing.

| Item | Value |
|---|---|
| Training rows | 141,435 month-start signals |
| Signal dates | 2010-01-04 through 2015-12-01 (72 dates) |
| Latest label used | 2016-03-01 |
| Target clip at fit | training 1st/99th percentiles, -0.454545 to 0.654765 |
| Prediction | 90-calendar-day adjusted-close holding return, as a fraction |

Month-start peer features are computed on the full eligible cross-section of each training date before rows with unfinished holds are dropped. The numeric step is `QuantileClipper` (training 1st/99th percentiles). `sector` is one-hot encoded with `handle_unknown="ignore"`. Both are inside the saved pipeline. `course_data.py` is the importable copy of `QuantileClipper` that unpickling requires.

The artifact is about 0.44 MB, so it is stored at `artifacts/model.joblib`. `artifacts/model_metadata.json` records the fit. Library versions match the HW4 training environment: Python 3.12.9, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, joblib 1.6.0. The process refuses to start when the running versions differ. Loading a scikit-learn artifact across versions is unsupported.

## Example from the experiment

`example_request.json` is symbol `A` on 2016-04-01, the first symbol that day whose features are all finite and inside the contract below. Peer features for that row use the eligible names on that date. Before the API existed, `pipeline.predict` on that row returned **-0.09155196684529192** (about -9.16 percent over the 90-day hold). That pair is stored in `artifacts/model_metadata.json`. The valid-prediction test compares the endpoint both with a fresh direct call and with this recorded value, within `1e-12`.

## Input contract

`POST /predict` takes one JSON object. All 22 fields are required, and extra fields are rejected. Returns and ratios are fractions: `0.01` means 1 percent. Numeric fields must be JSON numbers. `"0.01"` and `true` are rejected. JSON integers are accepted because JSON has one number type. `null`, `NaN`, and infinities are rejected.

Price-ratio and volatility caps sit above the HW4 panel's 99.9th percentile. Drawdown, volume, size, and rank use the bounds implied by their formulas. Accounting ratios use wide finite caps that cover the ordinary panel and reject the 1e14 filing errors. The saved `QuantileClipper` still winsorizes numeric inputs at the training 1st and 99th percentiles after this check. Training could leave a financial field missing; this API does not, because a non-finite feature must be a 422 rather than a silent imputed or tree-missing value.

| # | Field | Type | Unit and range |
|---|---|---|---|
| 1 | `momentum_5` | number | 5-session adjusted return, fraction. `(-1, 5]` |
| 2 | `momentum_21` | number | 21-session adjusted return, fraction. `(-1, 10]` |
| 3 | `momentum_63` | number | 63-session adjusted return, fraction. `(-1, 20]` |
| 4 | `momentum_126` | number | 126-session adjusted return, fraction. `(-1, 30]` |
| 5 | `momentum_252_21` | number | Return from 252 sessions ago to 21 sessions ago, fraction. `(-1, 50]` |
| 6 | `volatility_21` | number | Annualized 21-session volatility, fraction. `[0, 10]` |
| 7 | `volatility_63` | number | Annualized 63-session volatility, fraction. `[0, 10]` |
| 8 | `drawdown_252` | number | Close / 252-session maximum minus 1. `(-1, 0]` |
| 9 | `volume_ratio_21` | number | Volume / 21-session average minus 1. `[-1, 20]` |
| 10 | `large_cap_score` | number | Quarterly size score, 1 = largest. `(0, 1]` |
| 11 | `net_margin` | number | Net income / revenue, fraction. `[-2000, 100]` |
| 12 | `operating_roa` | number | Operating income / assets, fraction. `[-50, 50]` |
| 13 | `debt_to_assets` | number | Total debt / assets, fraction. `[0, 100]` |
| 14 | `cash_to_assets` | number | Cash / assets, fraction. `[-1, 20]` |
| 15 | `fcf_to_assets` | number | Free cash flow / assets, fraction. `[-50, 50]` |
| 16 | `accruals_to_assets` | number | (Net income − operating cash flow) / assets, fraction. `[-50, 50]` |
| 17 | `filing_age_days` | number | Days from the day after filing to the signal. `[0, 800]` |
| 18 | `market_momentum_63` | number | `^SP500TR` 63-session return, fraction. `(-1, 1]` |
| 19 | `market_volatility_63` | number | `^SP500TR` annualized 63-session volatility, fraction. `[0, 2]` |
| 20 | `sector_relative_momentum` | number | `momentum_63` minus the same-day sector median, fraction. `[-25, 25]` |
| 21 | `sector_profitability_rank` | number | Percentile rank of `operating_roa` in the date and sector. `(0, 1]` |
| 22 | `sector` | string | One of Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities |

Column order above is the order the model was trained on. The request uses names. The handler builds a one-row DataFrame with those names before `predict`, which is what the fitted `ColumnTransformer` expects.

## Output

`200` with `prediction` (a finite float, the 90-day return as a fraction) and `model_version` (`hw4-hgb-deep-static-2016-04-01`).

`422` for a missing field, an extra field, a wrong type, a non-finite number, a value outside the ranges above, an unknown sector, or a body that is not this 22-field object.

`500` with `{"detail": "Prediction unavailable"}` when inference fails. The traceback stays in the server log.

## Latency

`test_predict_performance` warms up once, then times 30 successful `POST /predict` calls with `time.perf_counter`. This is an in-process Starlette `TestClient` measurement, not a real request to localhost. The p95 uses nearest rank. On an Intel Core i9-9980HK at 2.40 GHz with 32 GB RAM, macOS 15.6, the median was 12.313 ms and the p95 was 13.592 ms. The test asserts p95 under 250 ms: a 22-feature histogram-boost prediction on this machine is about 13 ms through the test client, and 250 ms still fails if a request starts refitting. `test_lint.txt` is the terminal record: `python -m pytest -q -s` (16 passed; `-s` keeps the latency line), then `python -m ruff check .` and `python -m ruff format --check .`. `make test` and `make lint` run the same checks. Pytest also prints a Starlette warning that `httpx` with `TestClient` is deprecated.

## Run

From this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

With that environment active, the Makefile targets use the same interpreter:

```sh
make test
make lint
make format
```

Interactive docs: <http://127.0.0.1:8000/docs>.

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

## Container

Another person needs Docker and this directory, `Homeworks/HW5/Dhruv_Gupta`. The image is `python:3.12-slim`. The build installs `requirements.txt` (the same scikit-learn 1.9.1 stack used to fit the model), then copies `app.py`, `scoring.py`, `course_data.py`, `example_request.json`, and `artifacts/model.joblib`. `app.py` handles the request. `scoring.py` loads the artifact and builds the feature row. `.dockerignore` keeps `.venv`, caches, `.env` files, and `*.duckdb` out of the build context. Nothing is downloaded or refit at runtime.

From this directory, with host port 8000 free:

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The process listens on port 8000 inside the container. `make build` and `make run` are the same two commands. Stop a local server already bound to 8000 first, or map a different host port, for example `-p 127.0.0.1:8001:8000`. Check the image with the saved example:

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

`container_request.txt` is the terminal record of that request against `mfe-hw5:local`: HTTP 200 with prediction `-0.09155196684529192` and model version `hw4-hgb-deep-static-2016-04-01`, a 422 for `{"momentum_5":"0.02"}`, then the same image stopped and started again. The restarted container returned the same prediction.

## Before a real deployment

This image is a single-process homework service. A few things would need attention before it took live traffic.

`POST /predict` has no authentication, rate limit, or caller identity. Anyone who can open port 8000 can score a row. There is no health route. `make run` starts one uvicorn process with `docker run --rm`, so a crash removes the container and the model stays down until someone starts it again.

The fitted pipeline is copied into the image. A new fit means a new build. Startup refuses to load the artifact when scikit-learn, numpy, pandas, scipy, or joblib differ from the HW4 versions, so the previous image is the rollback. There is no separate model store and no way to swap the artifact without a rebuild.

Callers must send features already built the HW4 way, including the same-day sector median and profitability rank. A number inside the ranges above can still be the wrong feature, and the service will score it. Successful predictions are not logged. The latency above is an in-process `TestClient` measurement, so a production budget would be timed against the running container, with a request timeout and a count of 500 responses.
