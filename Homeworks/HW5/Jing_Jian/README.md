# HW5: Jing Jian — serving the frozen HW4 stock-return model

This service exposes the final HW4 `price_top20` Ridge model (`alpha=100`),
version `hw4-price-top20-ridge100-20160331-v1`. It loads the included 1.8 KB
fitted pipeline once per server process. Requests use saved feature clipping,
median imputation, and scaling; they never train or query a database.

Verified on November 18, 2026: **69 tests passed**, Ruff lint/format checks passed,
and a real Docker container returned 200/422 and the same prediction after restart.
See [check output](evidence/checks.txt), [latency results](evidence/latency.json),
and [container transcript](evidence/container_requests.txt).

## Setup and local use

Use Python 3.12. From this submission directory:

```sh
python3.12 -m venv hw_5_Jing_Jian
source hw_5_Jing_Jian/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On Windows use `py -3.12 -m venv hw_5_Jing_Jian` and `hw_5_Jing_Jian\Scripts\Activate.ps1`.
The artifact is included; the setup and service need no HW4 database.
Open <http://127.0.0.1:8000/docs> for the interactive API contract.

```sh
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json
```

The included AAL example uses the March 31, 2016 signal and predicts
approximately `0.046509127389602` (4.6509%). A successful response contains
exactly `prediction` (finite number) and `model_version` (string).

## Input and output contract

`POST /predict` accepts a JSON object with exactly five required numeric fields.
The following is the internal DataFrame column order; JSON key order is irrelevant.
All values are fractions, so 0.10 means 10%, not ten percentage points encoded as 10.

1. `momentum_21`: adjusted close / adjusted close 21 sessions earlier − 1;
   strictly greater than −1, with no artificial upper bound.
2. `momentum_63`: the corresponding 63-session return; strictly greater than −1.
3. `momentum_252_21`: adjusted close 21 sessions ago / adjusted close 252
   sessions ago − 1; strictly greater than −1.
4. `volatility_63`: sample standard deviation of 63 daily adjusted-price returns,
   multiplied by sqrt(252); between 0 and 0.6 inclusive.
5. `drawdown_252`: adjusted close / maximum adjusted close over the last 252
   sessions − 1; strictly greater than −1 and at most 0.

Positive adjusted prices imply the return/drawdown lower bounds. The volatility
ceiling preserves HW4's eligibility screen. Upward momentum is not arbitrarily
capped; the saved training-only clipper handles tails. All inputs must be finite
JSON numbers; integers are valid numbers. Numeric strings, booleans, nulls,
missing/extra fields, arrays, invalid bounds, and NaN/Infinity receive **422**.
Named required fields enforce exactly five features. The validation handler
omits raw error inputs so non-finite values cannot break error serialization.

The prediction is an unannualized adjusted-price return from next-session close
to 90 calendar days after entry (next market session when maturity is closed).
It is a model score before portfolio transaction costs, not a trade order.
Callers must construct features from the signal close with the original session
calendar and adjusted-price conventions. The API does not reconstruct or certify
HW4's other universe, liquidity, history-completeness, or daily-outlier screens.

Unexpected inference failures, including non-finite predictions, return **500**
with `{"detail": "Prediction unavailable"}`. Detailed exceptions stay in server logs.
Startup fails if the artifact cannot load; there is no synthetic fallback.

## Artifact provenance and reproducibility

The original HW4 scoring function discarded its fitted estimator. `export_hw4.py`
reconstructs the frozen final fit using the original database and code, with
training signals in 2010–2015 and outcomes completed before March 31, 2016.
It verifies the database and source hashes, all six saved fit-audit fields,
and all **186** original March 31 predictions before serializing anything.
The recovered fit has **13,334** rows, with latest outcome February 29, 2016.
The maximum prediction difference from the old CSV is below 1e-16.

`artifacts/metadata.json` records source hashes, training versions, audit, and
comparison results. `artifacts/reference_prediction.json` retains the AAL input
and expected prediction from HW4's pre-existing `outputs/full/final_scores.csv`;
its hash also matches HW4's recorded pre-amendment score hash. The reference
is independent of the API wrapper. Tests compare both direct pipeline inference
and this old result, with relative tolerance 1e-10 and absolute tolerance 1e-12
for serialization/platform floating-point variation.

`model_tools.QuantileClipper` copies the original transformer behavior into an
importable module without the research module's database dependency. The artifact
preserves learned 1%/99% feature clipping, median imputation, scaling, and Ridge
coefficients. Training-only target winsorization is recorded in metadata; no
additional target clipping is applied at inference. Model-library versions in
`requirements.txt` match the verified HW4 environment exactly. Load only trusted
artifacts: joblib uses pickle semantics.

Optional artifact regeneration, from the repository root with the HW4 environment
and bCourses database already installed:

```sh
.venv/bin/python Homeworks/HW5/Jing_Jian/export_hw4.py
```

This reads HW4 and overwrites only this submission's exported artifact, metadata,
and example. It does not rerun strategy selection or portfolio backtesting.
The development feature cache omits some final-fit labels and cannot substitute
for the full database. Regeneration also needs HW4's ignored original score CSV;
the distributed artifact and saved reference let reviewers run HW5 without it.

## Checks and latency

From this directory, activate the HW5 environment, then run:

```sh
make test
make lint
make format
# Equivalent checks when make is unavailable:
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The tests use the real artifact for successful predictions. Invalid inputs are
checked with an inference spy; exceptions and non-finite outputs are injected
only in failure tests. Additional checks cover named feature order, no request
refitting, and once-per-lifespan loading/cleanup.

`evidence/latency.json` records an in-process TestClient measurement: five warmup
requests and 50 sequential successful requests, timed with `time.perf_counter`.
It measures validation, DataFrame construction, inference, and response handling,
including TestClient overhead; it excludes a real network and cold startup.
The machine is an Apple M1 Max with 32 GiB RAM on macOS 15.6.1.
The recorded run measured **2.69 ms median** and **3.53 ms p95**.
There is no universal latency budget. See the evidence file for observed median,
p95, Python version, and hardware. The supplied dependency pins emit a Starlette
warning that httpx-based TestClient is deprecated; the tests still execute.

## Docker and real requests

With Docker running, from this directory:

On this Mac, Docker Desktop installed its CLI in `$HOME/.docker/bin`. Add that
directory to the current terminal session before using the commands below:

```sh
export PATH="$HOME/.docker/bin:$PATH"
```

```sh
make build
make run
# Equivalently:
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The image uses `python:3.12-slim`, `/app`, and an unprivileged numeric user.
The build context is allowlisted to application files, requirements, the small
artifact directory, and example input; environments, databases, and credentials
are excluded. Stop another service on port 8000 before running this command.

For repeatable build, real HTTP, invalid-request, and restart evidence:

```sh
python container_evidence.py
# Choose another available host port if needed:
python container_evidence.py --port 8001
```

This builds the image, records its ID, runs a dedicated container, uses curl to
check **200** and **422**, stops/starts the same container, and verifies the
original prediction again. It removes only its own container and saves output
to `evidence/container_requests.txt`. The included transcript records a successful
Linux/arm64 build and all three request checks; the temporary test container was removed.

## Before a real deployment

Add authentication, TLS, request size/rate limits, operational health/readiness,
structured monitoring, and resource/concurrency limits. Pin and maintain the
base-image digest and scan dependencies. Define an artifact approval and rollback
process, production latency/load tests, and feature freshness/drift monitoring.
Review corporate actions, identifier mapping, calendar alignment, and HW4's
point-in-time data limitations before using these scores for trading decisions.

## Attribution

Adapted from `Homeworks/HW5/template/`, the Lesson 9 FastAPI demo, HW4
`research.py`, and `Lectures/modeling/course_data.py`. Codex assisted with model
recovery, service implementation, tests, documentation, and verification under
Jing Jian's direction. Jing Jian should review and be able to explain the work.
Submit this folder via a PR and assign `kevinramlal` and `raghuramkowdeed`.
