# HW5: serving the HW4 return model

Quentin Benchetrit

A FastAPI service that loads the model from my HW4 experiment and returns its forecast for one stock. It is tested with pytest and packaged with Docker.

## Model

The service serves **`hw4-signed-ridge-2026-06-30`**: the sign-constrained ridge of HW4, as fitted at its last quarterly refit, after the close of 30 June 2026. It forecasts a stock's 90-calendar-day return minus a shrunk beta times the S&P 500 TR return over the same days.

`export_model.py` refits it from the HW4 feature panel with the HW4 code in `hw4/` (copied from my HW4 submission, only reformatted by Ruff). It checks that the refit reproduces exactly the 24,624 forecasts HW4 recorded for this fit, then writes:

- `artifacts/model.joblib`: the fitted model with its preprocessing (centring, signing and imputation of the ranks), its version and its feature order (2 KB);
- `artifacts/model_metadata.json`: fit date, training rows, feature order, library versions;
- `artifacts/reference_example.json`: the inputs and the HW4 forecast (0.01313428) of the stock HW4 ranked first on its last signal date, PARR on 24 September 2026;
- `example_request.json`: the same inputs, as a request body.

The service only loads the artifact: it never fits a model or reads the HW4 database. The artifact is a pickle, so only load one you created yourself.

To rebuild the artifact, run the HW4 notebook (it writes `outputs/panel.parquet` and `outputs/predictions.parquet`), then, in the HW4 environment:

```bash
python export_model.py --hw4-outputs ../../HW4/Quentin_Benchetrit/outputs
```

## Input and output

`POST /predict` takes a JSON object with exactly the 26 fields below, in any key order. Each value is the stock's percentile rank among the stocks eligible on the signal date, as computed by `hw4/features.py`. It is a float in (0, 1] (`rank(pct=True)` gives 1/n to n/n), or `null` when the characteristic is unavailable. The model treats `null` as the middle rank, as HW4 did. There are no categorical inputs.

| # | Field | Characteristic ranked |
|---|---|---|
| 1 | `mom_12_1` | 12-month return, skipping the latest month |
| 2 | `mom_6_1` | 6-month return, skipping the latest month |
| 3 | `high_252` | price relative to its 252-session high |
| 4 | `ret_21` | return over the last 21 sessions |
| 5 | `vol_63` | volatility of daily returns over 63 sessions |
| 6 | `vol_252` | volatility of daily returns over 252 sessions |
| 7 | `idio_vol` | volatility not explained by the S&P 500 TR, 252 sessions |
| 8 | `beta` | beta to the S&P 500 TR over 252 sessions |
| 9 | `max_ret_21` | largest daily return over the last 21 sessions |
| 10 | `roa` | trailing-year net income / total assets |
| 11 | `roe` | trailing-year net income / equity |
| 12 | `gross_profitability` | trailing-year gross profit / total assets |
| 13 | `operating_margin` | trailing-year operating income / revenue |
| 14 | `asset_turnover` | trailing-year revenue / total assets |
| 15 | `accruals` | trailing-year net income minus operating cash flow, / total assets |
| 16 | `revenue_acceleration` | change in year-on-year revenue growth |
| 17 | `income_change` | year-on-year change in quarterly net income / total assets |
| 18 | `margin_change` | year-on-year change in quarterly gross margin |
| 19 | `asset_growth` | year-on-year growth of total assets |
| 20 | `share_growth` | year-on-year growth of shares outstanding |
| 21 | `earnings_yield` | trailing-year net income / market value |
| 22 | `book_to_price` | equity / market value |
| 23 | `sales_to_price` | trailing-year revenue / market value |
| 24 | `cash_flow_yield` | trailing-year operating cash flow / market value |
| 25 | `fcf_yield` | trailing-year free cash flow / market value |
| 26 | `ebitda_to_ev` | trailing-year EBITDA / enterprise value |

The service returns **422** when:

- a field is missing or unknown (so any count other than 26);
- a value is a string (even a numeric one), a boolean, NaN, an infinity, or outside (0, 1];
- the body is not a JSON object.

The 422 body gives the field and the reason, not the rejected value: FastAPI's default handler echoes the value, and a NaN cannot be written back as JSON, which would turn the 422 into a 500.

A valid request returns **200** with `{"prediction": ..., "model_version": "hw4-signed-ridge-2026-06-30"}`. `prediction` is the forecast as a fraction (0.0131 is 1.31%) and is always finite. If inference fails or the forecast is not finite, the service returns **500** with `{"detail": "Prediction unavailable"}` and logs the traceback.

## Setup and commands

Python 3.12, from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test     # python -m pytest -q
make lint     # python -m ruff check . and python -m ruff format --check .
python -m uvicorn app:app --host 127.0.0.1 --port 8000   # docs at http://127.0.0.1:8000/docs
```

The model-library versions in `requirements.txt` are those of my HW4 training environment.

`app.py` handles HTTP and validation; `serving.py` loads the artifact and computes the forecast.

## Tests

| Group | What is checked |
|---|---|
| Valid prediction | 200 and the response schema; same forecast as a direct call of the saved model; the forecast HW4 recorded for PARR, within 1e-8; `null` gives the middle-rank forecast; key order does not matter |
| Input validation | 13 invalid bodies return 422, with the model's `predict` replaced by a function that fails the test if called |
| Inference failure | `predict` raises, then returns NaN: 500 each time, with the details in the log and not in the response |
| Performance | 5 warm-up requests, then 50 timed with `time.perf_counter` through the in-process `TestClient` (no network) |

The 1e-8 tolerance covers float32 rounding: the model computes in float32, as in training, so scoring one row instead of a batch can change the 9th decimal (2e-9 for PARR on my Mac, 0 in the Linux container).

```
$ make test
python -m pytest -q
..................
latency, in-process TestClient, 50 requests: median 1.69 ms, p95 1.92 ms
.                                                      [100%]
19 passed, 1 warning in 5.07s

$ make lint
python -m ruff check .
All checks passed!
python -m ruff format --check .
8 files already formatted
```

Measured on an Apple M4 laptop (16 GB, macOS). The test fails if p95 exceeds 50 ms. A ridge on 26 inputs is far faster, so this budget only catches a gross regression, such as reloading the model on every request, and leaves room for slower machines. The warning is Starlette's deprecation notice for `httpx` in its test client.

## Container

```bash
make build    # docker build -t mfe-hw5:local .
make run      # docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

The image is `python:3.12-slim` with the pinned requirements, `app.py`, `serving.py`, `hw4/` (needed to load the model) and `artifacts/`. `.dockerignore` keeps local environments, databases and secrets out of the build context. Anyone with Docker can rebuild the image from this folder with `make build`.

`container_demo.txt` is the transcript of a run (with `docker run -d`, so that the same terminal can send the requests):

- a valid request returns 200 with the forecast HW4 recorded for PARR and the model version;
- a request with `beta` set to 1.5 returns 422;
- after `docker stop` and a new `docker run` of the same image, the same request returns the same 200.

## Before a real deployment

- **Inputs.** The caller must compute the ranks exactly as HW4 did, over the same universe and on the same day. A real service would compute them itself from a data feed and monitor their distribution.
- **Model updates.** Each quarterly refit needs a new versioned artifact, a checksum verified at load (pickle runs code) and a way to roll back.
- **Operations.** Authentication, rate limiting, a health endpoint, several workers, and monitoring with alerts on latency and 500s.
- **Image.** No test tools, a non-root user, and a base image pinned by digest.

## Use of AI

I used an AI assistant (Claude Code) to proofread the documentation and comments, and for the final commit and push of this submission.
