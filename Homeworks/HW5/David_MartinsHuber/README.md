# HW5 — David Martins Huber: serve the HW4 FF5 return model

This service serves **all 1,399 eligible, fitted stock regressions** from the
FF5-inspired component of my HW4 portfolio at the **October 30, 2015 refit**.
Each stock has its own intercept and coefficients. Predictions are adjusted
stock returns over a **90-calendar-day holding period**, before fees.

This is one fixed historical snapshot, not today's universe and not a model
for every ticker that has ever existed. Its 2,000 quarterly universe members
include 1,466 eligible names: 1,399 fitted models and 67 eligible names without
a valid regression. Another 534 members fail the strategy's eligibility screens.
The saved membership and model availability are in `artifacts/membership.json`.

## Model and artifact provenance

HW4 `FactorTiltStrategy.refit` estimates, separately for each stock:

```text
prediction = intercept + beta_MKT*MKT + beta_SMB*SMB
                       + beta_HML*HML + beta_RMW*RMW + beta_CMA*CMA
```

The coefficients are the actual `numpy.linalg.lstsq` result produced by the
original HW4 refit code, captured without modifying HW4 source. This exporter
reconstructs the dated fit offline, rather than recovering an object from the
original notebook kernel. Factor construction uses the full homework universe;
all stock regressions use the original full price panel. Every exported score
is checked against the original HW4 score before writing the artifact. The configuration is
HW4's 60-month rolling window, at least 24 completed observations, and a
12-observation trailing mean for expected factor returns.

`artifacts/ff5.json` stores a symbol-keyed collection of fitted intercepts and
five coefficients, feature order, target horizon, version, and preprocessing
description. `artifacts/membership.json` records each dated universe member,
selection date, eligibility, model availability and unavailability reason. HW4 fits raw
factor returns with an intercept; **no scaler, imputer or other learned
preprocessing was fitted** for this regression. The intercept is saved with the
coefficients. The importable `model_service.py` applies this same calculation.
JSON avoids dependencies on pickled custom classes; the API checks the artifact
checksum and exact training-library versions on startup.

`artifacts/metadata.json` records coverage counts, dates, source hashes, source
commit, export versions, and reference input. Each stock model additionally
records its observation count and training formation dates. `artifacts/training_fixture.csv`
is a small AAPL fixture of the actual fitted design rows and 90-day targets. A test
independently recomputes their least-squares coefficients to check the extraction.

The reference predictions were obtained directly from HW4's full
`strategy.scores` **before API wrapping**, not from this service.
`reference_prediction.json` preserves scores for every fitted symbol and the
shared expected factor vector. AAPL is retained as the example request.
The API, direct model call and original HW4 score must agree within
`rtol=1e-9, atol=1e-12` to allow floating-point/platform differences.

### Target and feature timing

The stock target is `adjusted_exit / adjusted_entry - 1`. Entry is the market
session after the formation signal; exit is the first benchmark session on or
after entry + 90 calendar days. Missing endpoint stock quotes exclude the
observation from fitting. Only targets whose exit is strictly **before** the
refit signal may enter its training sample.

HW4 builds approximate FF5 factors from the supplied universe and financials;
these are not downloaded official Fama-French factors. Market is cap-weighted
without subtracting a risk-free rate; the remaining factors are size, value,
profitability and investment spreads. HW4 compounds three monthly returns to
create the factor inputs; their alignment to the stock's exact 90-calendar-day
target is approximate and is preserved here.

At a prediction date the model uses the trailing mean of up to 12 known,
three-month compounded factor vectors. The original HW4 code excludes unfinished
factor windows using its three-month cutoff and fills missing expected factor
components with zero. The saved example is this **expected factor vector**;
it does not contain future realized factor returns. Clients must construct
inputs using information known at their prediction date.

The offline exporter loads development data through December 31, 2015 so the
course helpers have the following market session and month-end calendar.
The actual regression price context stops on October 30, and the original HW4
maturity and factor cutoffs restrict the fitted/expected inputs. No final-test
2016–2026 prices are loaded to build this snapshot.

## API contract

`POST /predict`: JSON object with **six required named fields**: categorical
`symbol` plus the five numeric factors below. `symbol` selects that stock's
frozen regression; it is not a sixth numerical model feature. The service
constructs the numerical vector in training order regardless of JSON key order.

`symbol` must be an exact uppercase HW4 ticker, 1–32 characters, starting with
a letter/digit and containing only letters/digits or `.`, `-`, `^`, `/`.
No case conversion or ticker substitution is performed. A new/reassigned ticker
must not be interpreted as the historical entity automatically.

| Order | Name | Meaning |
|---|---|---|
| 1 | `MKT` | Expected compounded market-factor return |
| 2 | `SMB` | Expected compounded small-minus-big return |
| 3 | `HML` | Expected compounded high-minus-low return |
| 4 | `RMW` | Expected compounded robust-minus-weak return |
| 5 | `CMA` | Expected compounded conservative-minus-aggressive return |

All five factor fields are JSON numbers in **decimal return units**, finite and within inclusive
`[-1, 1]`. For example, `0.02` means 2%, not 0.02%. This range is an operational
input policy for expected three-month factor returns. Factor spreads can in
principle exceed it; such extreme scenarios are rejected for investigation,
not clipped. The bounds are not learned preprocessing.

Numeric strings, booleans, nulls, arrays, missing fields, extra fields and
nonfinite values return **422 before inference**. Supplying only four factors or adding
a sixth factor is rejected. Missing or malformed symbols also return 422. A syntactically valid symbol is then checked against the saved snapshot:

| Situation | HTTP status | Machine-readable `detail.code` |
|---|---|---|
| Member with a fitted model | 200 | Prediction response |
| Not a member of the snapshot universe | 404 | `SYMBOL_OUTSIDE_UNIVERSE` |
| Member without a fitted model | 404 | `MODEL_UNAVAILABLE` |

`MODEL_UNAVAILABLE` adds `detail.reason`: `STRATEGY_INELIGIBLE` or
`NO_VALID_REGRESSION`. The latter includes insufficient completed history and
rank-deficient training designs; it does not claim which caused a specific skip.
Both 404 responses include the requested symbol, snapshot date and model version.
No other stock's coefficients are substituted. Unexpected inference/output
failures return **500** with a generic error; details are logged on the server.

Example outside-universe response:

```json
{
  "detail": {
    "code": "SYMBOL_OUTSIDE_UNIVERSE",
    "symbol": "ZZZZZZZ",
    "snapshot_date": "2015-10-30",
    "model_version": "hw4-ff5-universe-2015-10-30-v1"
  }
}
```

`example_request.json` is an actual input from the reconstructed HW4 experiment:

```json
{
  "symbol": "AAPL",
  "MKT": 0.007690397402656007,
  "SMB": 0.0008682295776333929,
  "HML": -0.04270218133233253,
  "RMW": -0.0008423709103398125,
  "CMA": -0.032673950126811846
}
```

Expected HTTP 200 response:

```json
{
  "prediction": 0.039849224628863006,
  "model_version": "hw4-ff5-universe-2015-10-30-v1",
  "symbol": "AAPL",
  "prediction_horizon_calendar_days": 90
}
```

The example predicts approximately **3.98%** return, before transaction costs.
This is an arithmetic return, not `1 + return`, a probability or an annualized
forecast. Predictions are not clipped: ordinary least squares is unconstrained
and can produce economically implausible values outside the training domain.
The model was used as a ranking signal in HW4; API correctness does not imply
forecast accuracy or profitability.

## Setup and local use

From `Homeworks/HW5/David_MartinsHuber/`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The separate environment and Docker image use pinned HW4 numpy/pandas/scipy/
sklearn versions. `/docs` exposes the interactive input/output schemas. The
artifact loads once per process using FastAPI lifespan. Requests never train,
query the database, or modify coefficients. The included artifact and custom
module are sufficient for inference without HW4 files.

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data-binary @example_request.json
```

### Offline export (optional)

Ordinary use does **not** require re-exporting. With the original HW4 environment
and database available, reconstruct this model using:

```sh
../../HW4/.venv/bin/python export_hw4_snapshot.py \
  --hw4 ../../HW4/David_MartinsHuber \
  --database ../../HW4/David_MartinsHuber/data/equities_top2000.duckdb \
  --output .
```

This refits the dated model offline and replaces the saved artifact, fixture
and all-stock references. The exporter observes the original regression, rather than
training a replacement sklearn estimator. Source data revisions may change the
reconstruction. The JSON checksum detects accidental corruption; it does not
provide authenticity if an attacker can replace both artifact and metadata.

## Tests, lint and latency

```sh
make test
make lint
make format
```

Equivalent commands: `python -m pytest -q`, `python -m ruff check .`,
`python -m ruff format --check .`, and `python -m ruff format .`.
The real saved model is used for positive tests. Tests also cover independent
training-fixture reconstruction, each input's effect and order, invalid inputs
with inference spies, intentional failures, nonfinite model output, independent
concurrent requests, load-once startup, checksum failures and version mismatch. The snapshot integration test sends
requests for **all 2,000 members**, compares every fitted stock with its original
HW4 score, and verifies unavailable-member errors. It separately checks a symbol
outside the universe. Learned parameters and membership stay frozen between
requests; symbol routing never triggers training.

`evidence/tests.txt` and `evidence/lint.txt` contain the latest captured results.
`evidence/latency.json` reports five warm-ups and thirty successful requests
measured by `time.perf_counter`, with median/p95 latency, CPU, OS and Python.
This is in-process TestClient latency (JSON, validation and inference included),
not real network latency. No universal latency budget is asserted.

## Docker and real HTTP checks

```sh
make build
make run
```

The Dockerfile uses `python:3.12-slim`, working directory `/app`, the same pinned
requirements, port 8000, serving modules and the saved FF5 artifact. The Makefile
binds to localhost only. Stop any other service on host port 8000 first.
`.dockerignore` excludes environments, databases and local secrets; the
Dockerfile explicitly copies the necessary serving files.

```sh
python check_container.py
```

This script runs the image on independent localhost port 18005, demonstrates a
valid 200 requests for two different stocks, an invalid 422 request, outside-
universe and unavailable-model 404 responses, stops and starts the same
container/image, and verifies the prediction again. It removes only its own named container.
`evidence/build.txt` and `evidence/container.txt` capture the build and request/
restart evidence.

## Before a real deployment

Validate forecast calibration and out-of-sample stability, monitor drift and
feature availability, and define offline retraining/promotion/rollback rules.
Add authenticated access, request limits, structured monitoring and artifact
provenance controls. Extending support beyond this snapshot requires a new export with explicit
membership and per-stock fits. The service never reuses one stock's coefficients
for another stock or interprets a historical ticker as a current security.
The expected-factor inputs require a separately governed causal data pipeline.

## Attribution

API lifespan, test fixtures, Makefile and tooling adapt `Homeworks/HW5/template`.
Model fitting and feature definitions come from my HW4
`strategies/factor.py` and `backtesting/famafrench.py`, with data loading attributed
in HW4 to the course helpers. The API preserves that implementation's
approximations. Export, service, tests and documentation used Codex assistance.

## Snapshot membership semantics

The saved base universe is the execution quarter's `universe_membership` rows
whose selection date was known by the signal. The next session is November 2,
2015; the quarter starts October 1. Eligibility then follows HW4's separate
known-currency and FF5 financial/price/sector screens. A known non-USD or otherwise
ineligible member remains listed as a universe member without a served model.
This preserves the difference between course-universe membership and strategy
eligibility. Eligibility alone does not guarantee sufficient observations or a
full-rank design. The model loader verifies coverage and both artifact checksums.

Inspect `artifacts/membership.json` for supported tickers and availability;
`artifacts/ff5.json` contains only the fitted subset. Nothing is inferred from
future membership changes or a present-day active-stock list.
