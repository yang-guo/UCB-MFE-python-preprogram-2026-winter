# HW5 - Yixiao Yang

This project serves the Ridge model from my completed HW4 through FastAPI.
It uses the first final-evaluation snapshot, fitted on **March 31, 2016**
for entries in **2016Q2**, with `alpha=1000`.
The model version is `hw4-yixiao-ridge-2016q2-v1`.

**Verification status:** the saved model, 106 API tests, Ruff checks, and real
native localhost HTTP requests have passed.

<!-- container-status:start -->
**Docker verification: passed.** The actual image build, valid request (200), invalid request (422), and prediction after restarting the same image passed. See `evidence/container_transcript.txt` for the commands, image ID, responses and execution time.
<!-- container-status:end -->

## 1. HW4 model and provenance

The source is my [HW4 PR #227](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/pull/227),
commit `80c17f2fd6217926c18f7548d5d03dc4cfbf2c53`.
HW4 saved forecasts and training audits, but did not serialize its fitted
objects. For HW5, `export_hw4_model.py` recreated one original quarterly fit
using the unchanged HW4 code and the official `equities_top2000.duckdb`.
It did not select a new alpha or change the research design.

The fitted sequence is:

1. Training-only 1st/99th percentile feature clipping.
2. Median imputation, with the learned missing-value indicators and
   `keep_empty_features=True`.
3. `StandardScaler`, using the saved training means and scales.
4. `Ridge(alpha=1000)`.

The restored fit used **141,407 training observations**. All **5,979 forecasts**
for the quarter matched the previously saved HW4 forecasts exactly when
compared as a batch: maximum absolute difference **0.0**.
The custom clipper's class definition was checked against HW4, moved into an
importable module without changing its fitted state, and checked again after
joblib serialization and reloading. See [Appendix A4](#a4-model-export-verification).

`artifacts/model.joblib` is **3,310 bytes** and contains the whole fitted
pipeline. Its SHA-256 is:

```text
b0011872418e857c65e07c3f8cfcb5b4711695c93f24b0e3dea58626fb544952
```

`artifacts/model_metadata.json` records the source/database/forecast hashes,
training cutoff, package versions and feature order. Python 3.12.14,
NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0 match the recorded HW4
environment. SciPy 1.17.0 and joblib 1.5.3 are explicitly recorded as the
export environment; the old HW4 log did not record their versions.
The service requirements pin the validated environment, including transitive
dependencies.

The server loads the trusted artifact once during lifespan startup. It does
not query DuckDB, read price histories, or fit any preprocessing/model at
startup or during requests. Its path is relative to `app.py`.

## 2. Input contract

`POST /predict` accepts one JSON object with **exactly these 12 required keys**.
All numbers must be finite JSON numbers. Numeric strings and booleans are
rejected. A field marked nullable must still be present; use JSON `null`
when that HW4 feature is unavailable.

The table gives the internal DataFrame column order. JSON key order does not
matter because the service explicitly reconstructs the training order.
Send raw HW4 features, without manually standardizing or imputing them.

| # | Field | Type and range | Definition / units |
|---|---|---|---|
| 1 | `momentum_21` | Number > -1; non-null | Adjusted close `P[t] / P[t-21] - 1`; return fraction |
| 2 | `momentum_63` | Number > -1; non-null | `P[t] / P[t-63] - 1`; return fraction |
| 3 | `momentum_252_21` | Number > -1 or null | `P[t-21] / P[t-252] - 1`; return fraction |
| 4 | `volatility_63` | Number >= 0; non-null | Sample standard deviation of 63 daily returns, multiplied by sqrt(252); annualized fraction |
| 5 | `drawdown_252` | Number > -1 and <= 0, or null | `P[t] / max(P[t-251], ..., P[t]) - 1` |
| 6 | `net_margin` | Finite number or null | Quarterly net income / revenue |
| 7 | `operating_roa` | Finite number or null | Quarterly operating income / total assets |
| 8 | `debt_to_assets` | Finite number or null | Total debt / total assets |
| 9 | `cash_to_assets` | Finite number or null | Cash and equivalents / total assets |
| 10 | `fcf_to_assets` | Finite number or null | Quarterly free cash flow / total assets |
| 11 | `accruals_to_assets` | Finite number or null | (Quarterly net income - operating cash flow) / total assets |
| 12 | `filing_age_days` | Integer from 0 through 2**53 - 1, or null | Calendar days since the filing became available: filing date + 1 calendar day |

Returns use fractions: `0.05` means 5%. Financial flow ratios are quarterly,
not annualized or TTM. The lower bounds on returns/drawdown follow from the
positive adjusted prices used in HW4. Volatility cannot be negative.
I do not impose arbitrary upper bounds on volatility or force financial
ratios into [0, 1]: losses, leverage and the original observations can fall
outside that interval. The fitted clipper handles extreme observed values.

HW4 permits missing long-history and financial inputs, so explicit nulls
become NaN only inside the feature matrix and are handled by the saved
imputer. The 21/63-session momentum and volatility fields are required to
be observed. `filing_age_days` is not capped at 550: HW4's 550-day stale
report rule applies to the financial ratios and uses fiscal-period age.
The upper age limit is a technical exact-integer bound for the float64 feature
matrix. It rejects oversized JSON integers during validation, before they can
overflow conversion and turn an invalid request into an inference error.

## 3. Output and example

A valid request returns HTTP **200**:

```json
{
  "prediction": -0.10751624117051217,
  "model_version": "hw4-yixiao-ridge-2016q2-v1"
}
```

This is the saved **historical AAPL example**, with signal date 2016-03-31
and entry date 2016-04-01. Its full input is in `example_request.json`.
The output is a gross adjusted-price return fraction from the entry close
to the first market session at or after **entry + 90 calendar days**.
It is not an annual return, a 90-trading-day return, or a current forecast.

`reference_prediction.json` retains the independent original HW4 forecast
`-0.1075162411705124`, taken from HW4's frozen CSV before the API existed.
The single-row result differs by about `2.22e-16` from the original batch
result. Tests use both absolute and relative tolerance `1e-10` to allow
ordinary floating-point differences.

Missing/extra keys, wrong shapes or types, non-finite values and invalid
ranges return **422** before inference. Validation error responses omit
the offending raw value so a NaN/Infinity input cannot break error JSON
serialization. An unexpected prediction failure or non-finite model result
returns **500**, with the public message `Prediction unavailable`.
The full exception stays in the server log; `X-Request-ID` links an inference
response to that log. `GET /health` reports readiness and model version.

## 4. Setup and local checks

Use Python **3.12** in a separate environment. From this folder on Windows
PowerShell, activation is optional; these commands call the environment
directly:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe benchmark.py
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pytest -q
.venv/bin/python benchmark.py
```

With an activated environment, `make test`, `make lint`, `make format`,
`make build` and `make run` provide the template's shortcuts.
Windows users without make can use the equivalent commands above/below.

Run the native API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. In another PowerShell terminal:

```powershell
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@example_request.json"
```

Stop this local server before using port 8000 for Docker.

## 5. Checks and measured latency

The fresh Python 3.12 environment passed:

- `python -m ruff check .`: all checks passed.
- `python -m ruff format --check .`: formatting check passed.
- `python -m pytest -q`: **106 passed**, with two dependency deprecation
  warnings from the pinned Starlette/httpx/AnyIO test-client stack.
- Real native Uvicorn HTTP: valid request **200**, invalid request **422**,
  then a fresh process returned the same saved-model prediction.

The tests cover the four required groups, plus request-time no-refit checks,
one artifact load per lifespan, feature ordering, explicit missing values,
and row-alone versus row-in-batch consistency. Positive tests use the actual
fitted artifact. Failure tests replace only the loaded model's `predict`.
For invalid inputs, a spy verifies that `predict` is never called.
There are no skipped or placeholder tests.

`benchmark.py` uses **10 warm-up requests** and **100 measured successful
requests**, timed with `time.perf_counter`. Results from this run:

| Measurement | Value |
|---|---:|
| Median | 1.601 ms |
| p95, nearest-rank | 5.961 ms |
| CPU reported by the execution host | AMD EPYC 9V74 80-Core Processor |
| Logical CPUs visible / in affinity | 9 / 9 |
| OS | Linux 6.18.57, x86-64 |
| Python | 3.12.14 |

This is **in-process FastAPI TestClient timing**, including response checking.
It excludes real network transport and Docker. It is a serial microbenchmark,
not a concurrency or deployment capacity claim. No hardware-independent
500 ms threshold is asserted.

The original test/lint output, full latency record, native HTTP transcript,
model-export checks and requirement audit are consolidated in
[Appendix A](#appendix-a-original-verification-records). The real Docker
transcript is kept separately at `evidence/container_transcript.txt`.

Running `benchmark.py` again writes a fresh `evidence/latency.json`; the
recorded submission result is already included in Appendix A2.

## 6. Docker build, requests and restart evidence

The supplied Dockerfile uses `python:3.12-slim`, working directory `/app`,
the pinned requirements and the saved model. The build context allowlist
includes only application modules, requirements and artifacts. It excludes
virtual environments, databases, credentials, tests and generated reports.

With Docker Desktop/Engine running, use this command from the folder on
macOS/Linux:

```sh
python3 verify_container.py
```

On Windows PowerShell:

```powershell
py -3.12 verify_container.py
```

This helper uses only Python's standard library on the host; the image
installs its own application dependencies. It:

1. Builds `mfe-hw5:local` and records the actual image ID.
2. Starts a uniquely named container bound to `127.0.0.1:8000`.
3. Sends the saved example and verifies HTTP 200, a finite prediction,
   the model version and agreement with the independent HW4 reference.
4. Sends an extra-field invalid request and verifies HTTP 422.
5. Stops and starts the same container, checks that its image ID is
   unchanged, and repeats the valid prediction.
6. Saves actual commands and responses to `evidence/container_transcript.txt`
   and removes only the temporary container it created.

If port 8000 is occupied, append `--port 8001` to the command above.
A successful run ends with `FINAL RESULT: PASS` and updates the Docker status
at the top of this README. Keep the generated transcript in the submission.
If it is missing or ends with FAIL, the required container demonstration is
still incomplete. The native localhost transcript does not satisfy that item.

The equivalent manual build/run commands are:

```sh
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

## 7. Optional: reproduce the model export

Normal API use needs neither the database nor this step. To independently
recreate the artifact, use the exact linked HW4 source commit and its
`supporting_evidence.zip`. In the HW5 environment install the original
data-reader dependency, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install duckdb==1.5.6
.\.venv\Scripts\python.exe export_hw4_model.py --hw4-dir "PATH_TO_HW4_Yixiao_Yang" --database "PATH_TO_equities_top2000.duckdb"
```

The command checks the original source and database hashes, uses the original
training dates and selected alpha, compares all quarterly predictions, and
writes the artifact/reference/evidence. It can recreate
`evidence/model_export.json`; the recorded export result is already included
in Appendix A4. It does not run during serving.
Load only the supplied trusted artifact or one reproduced by this command.

## 8. Before a real deployment

I would add authenticated access and TLS, request/rate limits and operational
timeouts; run the container as a non-root user; and measure concurrent traffic,
tail latency and memory before choosing a worker count. Each process loads
its own model copy.

I would record the model version with an immutable image digest, deploy the
same tested artifact, retain a previous image for rollback, and check the
reference request after each deployment. Monitoring needs three parts:
service errors/latency/throughput/memory, input missingness/ranges/distribution
changes, and forecast quality once the 90-day outcomes become available.
A 200 response alone does not establish model usefulness. These follow
Lesson 9's deployment and monitoring discussion.

The Anthropic postmortem shows how infrastructure changes can harm model
outputs even when existing evaluations miss the problem. My application of
that lesson is to keep the fixed prediction regression check and to observe
the deployed service itself. The Netflix presentation's public description
also emphasizes the operational challenges of distributed services; for
this small model I would keep one clear service boundary unless measured
needs justify more components. The linked video's full playback was not
available during preparation; I used the course slides and the original
InfoQ presentation description.

## 9. Files and submission

This compact submission contains **19 files**, including the two dotfiles.
Five recorded check/report files have been consolidated into the foldable
appendix below. The application, tests, fitted model, dependency pins,
Dockerfile and every file included in the verified Docker build are unchanged.

| File(s) | Purpose |
|---|---|
| `app.py`, `model_contract.py` | Endpoint, lifecycle, errors and explicit input/output contract |
| `model_components.py` | Importable original HW4 clipper and feature order |
| `artifacts/` | Small fitted pipeline and provenance metadata |
| `example_request.json`, `reference_prediction.json` | Historical input and independent HW4 expected value |
| `tests/test_app.py` | Required behavior, failure, validation and performance tests |
| `benchmark.py` | Repeatable latency measurement |
| `Dockerfile`, `Makefile`, `requirements.txt`, `pyproject.toml` | Build, environment and checks |
| `verify_container.py` | Actual Docker request/restart evidence capture |
| `export_hw4_model.py` | Optional faithful offline artifact reconstruction |
| `README.md`, Appendix A | Original checks, latency, native HTTP, model export and audit records |
| `evidence/container_transcript.txt` | Unmodified actual Docker build, valid/invalid requests and restart record |

Put this whole `Yixiao_Yang` folder in `Homeworks/HW5/`, including the
supplied dotfiles. Keep the shared template unchanged. The assignment asks
for a pull request assigned to `kevinramlal` and `raghuramkowdeed`.
Do not commit the outer download ZIP, local virtual environments or the
DuckDB database. CI is optional and was not claimed as completed.

## Sources and assistance

- [HW5 assignment and template](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/77022f1b5c5307d3d28f492102ddb3bc90b04665/Homeworks/HW5/hw_5.md): milestone structure, validation and evidence requirements.
- [My HW4 source](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/80c17f2fd6217926c18f7548d5d03dc4cfbf2c53/Homeworks/HW4/Yixiao_Yang): actual features, fitted design, training rules and frozen reference forecasts.
- [Lesson 9 slides, notebook and saved demo reader](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/77022f1b5c5307d3d28f492102ddb3bc90b04665/Lectures/Lecture%209): API testing, containers, deployment and monitoring.
- [Lesson 9 demo code](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/77022f1b5c5307d3d28f492102ddb3bc90b04665/Lectures/Lecture%209/demo): adapted lifespan loading, error logging and TestClient fixture patterns.
- [FastAPI lifespan tests](https://fastapi.tiangolo.com/advanced/testing-events/) and [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html): lifecycle and version compatibility.
- [Anthropic postmortem](https://www.anthropic.com/research/a-postmortem-of-three-recent-issues) and [Netflix presentation description on InfoQ](https://www.infoq.com/presentations/netflix-chaos-microservices/): deployment observation and distributed-service operations.

ChatGPT assisted with implementation, model export, tests and documentation.
The numerical comparisons and Python/HTTP checks above were actually run.
The Docker status and transcript distinguish executed checks from pending work.

## Appendix A. Original verification records

The sections below preserve the full contents of five earlier standalone
records. Expand a section to read its original output or JSON. These records
are documentation; neither the service nor its tests read them at runtime.
Helper commands may regenerate standalone result files when rerun.

The original Docker transcript remains a separate file because
`verify_container.py` records and refreshes it directly.

### A1. Test and lint output

This is the original local-check output followed by the later Mac Docker verification note. The older pending statement is preserved as part of that historical output; the actual Docker run subsequently passed.

<details>
<summary>Expand the original checks.txt record</summary>

Original record name: `evidence/checks.txt`. Its full content is retained below.

<!-- evidence-record:start:evidence/checks.txt -->
```text
HW5 re-audit after the filing-age validation correction
Python 3.12.14; isolated environment, dependencies from requirements.txt.

COMMAND: python -m ruff check .
All checks passed!

COMMAND: python -m ruff format --check .
8 files already formatted

COMMAND: python -m pytest -q
........................................................................ [ 67%]
..................................                                       [100%]
=============================== warnings summary ===============================
../../../../verify_env/lib/python3.12/site-packages/fastapi/testclient.py:1
  /workspace/scratch/0e8fca807e6f/verify_env/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

../../../../verify_env/lib/python3.12/site-packages/starlette/testclient.py:53
  /workspace/scratch/0e8fca807e6f/verify_env/lib/python3.12/site-packages/starlette/testclient.py:53: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
106 passed, 2 warnings in 2.00s

The 2 new parametrized cases reject filing_age_days=2**53 and 10**400
with HTTP 422 and verify that prediction is not called.

Real native HTTP and native process restart: passed again.
Docker container execution: pending; no Docker runtime is available.


Later Docker verification on the user Mac
The pending Docker statement above describes the original local-check environment.
Actual Docker execution: 2026-11-16T12:54:18.258775+00:00 to 2026-11-16T12:59:24.290401+00:00.
Docker build, valid prediction (200), invalid extra field (422), and same-image restart: PASS.
Full unmodified execution evidence: evidence/container_transcript.txt.
```
<!-- evidence-record:end:evidence/checks.txt -->

</details>

### A2. Latency measurement

This records the 10 warm-ups and 100 measured successful in-process TestClient requests, their timings, the hardware and the measurement method.

<details>
<summary>Expand the original latency.json record</summary>

Original record name: `evidence/latency.json`. Its full content is retained below.

<!-- evidence-record:start:evidence/latency.json -->
```json
{
  "measurement": "in-process FastAPI TestClient; no TCP or Docker",
  "timing": "time.perf_counter around synchronous post plus response validation",
  "warmup_requests": 10,
  "successful_requests": 100,
  "median_ms": 1.601161500047965,
  "p95_ms": 5.960515999959171,
  "p95_method": "nearest rank",
  "min_ms": 1.2235410000585034,
  "max_ms": 15.102974000001268,
  "model_version": "hw4-yixiao-ridge-2016q2-v1",
  "hardware": {
    "platform": "Linux-6.18.57-x86_64-with-glibc2.41",
    "python": "3.12.14",
    "cpu_model": "AMD EPYC 9V74 80-Core Processor",
    "logical_cpus_visible": 9,
    "logical_cpus_in_affinity": 9
  },
  "budget": "No hardware-independent latency cutoff asserted",
  "samples_ms": [
    1.4549100000067483,
    1.3989560000027268,
    1.417083999967872,
    2.47405200002504,
    2.280474000031063,
    1.532198000063545,
    1.4792380000017147,
    1.3562420000425846,
    1.2254640000719519,
    1.2987640000119427,
    1.33116400002109,
    2.059385000052316,
    1.8419809999841164,
    1.5901850000545892,
    1.4909960000295541,
    1.6494750000219938,
    1.4424820000158434,
    1.6774270000041724,
    1.7164569999295054,
    1.490965000016331,
    1.7707690000179355,
    1.70407799998884,
    1.5357820000190259,
    1.464870999939194,
    1.6559239999196507,
    1.8904889999475927,
    1.764699000091241,
    1.988031999985651,
    1.6478730000244468,
    5.158141000038086,
    1.7137019999609038,
    1.3966720000553323,
    1.547179999988657,
    1.5130279999766572,
    1.4450649999844245,
    1.7578490000005331,
    1.2877870000238545,
    1.578036999944743,
    1.7711990000179867,
    1.3320049999947514,
    1.389240999969843,
    1.5493829999968511,
    1.5764339999577714,
    1.2991249999458887,
    1.2555989999327721,
    1.2235410000585034,
    1.8594330000496484,
    1.7376879999346784,
    1.4630229999283983,
    1.3218999999935477,
    1.6967870000144103,
    1.612138000041341,
    1.8199819999153988,
    1.8039389999557898,
    1.4241339999898628,
    2.1293959999866274,
    1.6962049999165174,
    1.5689029999066406,
    1.3925260000178241,
    1.506668999923022,
    1.7059909999943557,
    1.5473010000732756,
    1.727182999957222,
    1.4943559999665013,
    1.731339000002663,
    2.0856249999496868,
    15.102974000001268,
    2.458337999996729,
    1.5763850000212187,
    2.9412489999458558,
    5.613359999983913,
    1.755706000039936,
    1.995919000023605,
    1.2709119999954055,
    6.796449999910692,
    1.5684519999012991,
    1.2791049999805182,
    8.918719999996938,
    5.156639999995605,
    1.5527089999523014,
    1.3346279999950639,
    3.92706000002363,
    1.8163280000180748,
    1.2556390000781903,
    9.226144999956887,
    1.6245670000216705,
    1.2736859999904482,
    2.2275890000855725,
    6.545360000018263,
    1.6431949999287099,
    1.4521669999112419,
    2.058933999933288,
    2.6094670000702536,
    5.296161000046595,
    5.960515999959171,
    1.5388069999744403,
    1.3121440000531948,
    1.565022000022509,
    1.4717560000008234,
    1.3285690000657269
  ],
  "measured_at_utc": "2026-11-16T08:41:22.850697+00:00"
}
```
<!-- evidence-record:end:evidence/latency.json -->

</details>

### A3. Native HTTP requests

This earlier native Uvicorn run is supplementary evidence. The actual Docker valid/invalid/restart demonstration is retained separately in evidence/container_transcript.txt.

<details>
<summary>Expand the original local_http_transcript.txt record</summary>

Original record name: `evidence/local_http_transcript.txt`. Its full content is retained below.

<!-- evidence-record:start:evidence/local_http_transcript.txt -->
```text
HW5 native localhost HTTP verification
METHOD: real Uvicorn process over TCP on localhost. This is NOT Docker evidence.
UTC: 2026-11-16T08:40:54.913097+00:00

START COMMAND JSON: ["/workspace/scratch/0e8fca807e6f/verify_env/bin/python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "47861"]

GET http://127.0.0.1:47861/health
HTTP STATUS: 200
RESPONSE HEADERS: {"date": "Mon, 16 Nov 2026 08:40:56 GMT", "server": "uvicorn", "content-length": "63", "content-type": "application/json", "Connection": "close"}
RESPONSE BODY: {"status":"ready","model_version":"hw4-yixiao-ridge-2016q2-v1"}

POST http://127.0.0.1:47861/predict
REQUEST JSON: {"momentum_21": 0.08439560439560445, "momentum_63": 0.0077614379084967045, "momentum_252_21": -0.17721518987341767, "volatility_63": 0.3014368272098843, "drawdown_252": -0.16570848833276963, "net_margin": 0.24199968367777308, "operating_roa": 0.08241499706768865, "debt_to_assets": 0.21468269663534323, "cash_to_assets": 0.05690388838122775, "fcf_to_assets": 0.0799804967199029, "accruals_to_assets": -0.031034764937739528, "filing_age_days": 63}
HTTP STATUS: 200
RESPONSE HEADERS: {"date": "Mon, 16 Nov 2026 08:40:56 GMT", "server": "uvicorn", "content-length": "80", "content-type": "application/json", "x-request-id": "e8e390f6c51b489497fa9badf912709e", "Connection": "close"}
RESPONSE BODY: {"prediction":-0.10751624117051217,"model_version":"hw4-yixiao-ridge-2016q2-v1"}

POST http://127.0.0.1:47861/predict
REQUEST JSON: {"momentum_21": 0.08439560439560445, "momentum_63": 0.0077614379084967045, "momentum_252_21": -0.17721518987341767, "volatility_63": 0.3014368272098843, "drawdown_252": -0.16570848833276963, "net_margin": 0.24199968367777308, "operating_roa": 0.08241499706768865, "debt_to_assets": 0.21468269663534323, "cash_to_assets": 0.05690388838122775, "fcf_to_assets": 0.0799804967199029, "accruals_to_assets": -0.031034764937739528, "filing_age_days": 63, "unexpected_field": 1}
HTTP STATUS: 422
RESPONSE HEADERS: {"date": "Mon, 16 Nov 2026 08:40:56 GMT", "server": "uvicorn", "content-length": "112", "content-type": "application/json", "Connection": "close"}
RESPONSE BODY: {"detail":[{"type":"extra_forbidden","loc":["body","unexpected_field"],"msg":"Extra inputs are not permitted"}]}

GET /docs returned HTTP 200 (HTML omitted).

STOPPED native server; starting a fresh process with the same artifact.

START COMMAND JSON: ["/workspace/scratch/0e8fca807e6f/verify_env/bin/python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "47861"]

GET http://127.0.0.1:47861/health
HTTP STATUS: 200
RESPONSE HEADERS: {"date": "Mon, 16 Nov 2026 08:40:57 GMT", "server": "uvicorn", "content-length": "63", "content-type": "application/json", "Connection": "close"}
RESPONSE BODY: {"status":"ready","model_version":"hw4-yixiao-ridge-2016q2-v1"}

POST http://127.0.0.1:47861/predict
REQUEST JSON: {"momentum_21": 0.08439560439560445, "momentum_63": 0.0077614379084967045, "momentum_252_21": -0.17721518987341767, "volatility_63": 0.3014368272098843, "drawdown_252": -0.16570848833276963, "net_margin": 0.24199968367777308, "operating_roa": 0.08241499706768865, "debt_to_assets": 0.21468269663534323, "cash_to_assets": 0.05690388838122775, "fcf_to_assets": 0.0799804967199029, "accruals_to_assets": -0.031034764937739528, "filing_age_days": 63}
HTTP STATUS: 200
RESPONSE HEADERS: {"date": "Mon, 16 Nov 2026 08:40:57 GMT", "server": "uvicorn", "content-length": "80", "content-type": "application/json", "x-request-id": "838f661338c64f01a4f29cfe399aebcb", "Connection": "close"}
RESPONSE BODY: {"prediction":-0.10751624117051217,"model_version":"hw4-yixiao-ridge-2016q2-v1"}

RESULT: PASS - native valid/invalid requests and native process restart.
Docker build/run/restart was not executed in this environment.
```
<!-- evidence-record:end:evidence/local_http_transcript.txt -->

</details>

### A4. Model export verification

This records the model export checks and comparison with the frozen HW4 predictions.

<details>
<summary>Expand the original model_export.json record</summary>

Original record name: `evidence/model_export.json`. Its full content is retained below.

<!-- evidence-record:start:evidence/model_export.json -->
```json
{
  "all_checks_passed": true,
  "model_version": "hw4-yixiao-ridge-2016q2-v1",
  "quarter": "2016Q2",
  "fit_date": "2016-03-31",
  "training_rows": 141407,
  "frozen_forecasts_compared": 5979,
  "max_absolute_forecast_difference": 0.0,
  "absolute_tolerance": 1e-10,
  "relative_tolerance": 1e-10,
  "original_clipper_ast_matches": true,
  "clipper_fitted_state_unchanged": true,
  "relocated_class_predictions_bitwise_equal": true,
  "serialized_reload_predictions_bitwise_equal": true,
  "artifact_sha256": "b0011872418e857c65e07c3f8cfcb5b4711695c93f24b0e3dea58626fb544952",
  "artifact_bytes": 3310,
  "hw4_source_commit": "80c17f2fd6217926c18f7548d5d03dc4cfbf2c53",
  "hw4_source_sha256": {
    "hw4.py": "290feeccf6a8dd2f31b8acd7d630e18c0952685ec7a6b278e79c423c7eb60f31",
    "data.py": "981712f7c771c6cd395734a634879d959c36c6cdb59690eaf01b55d4bcf3f3cb",
    "backtester.py": "4b23b947c217e7014c7e4cffcf6e07ac692e1674383d220841ddacb4ef852dcc"
  },
  "database_sha256": "a028641d96c12e6f29d5dd34e3a6ebc059074dd38351648bf0d970b1dd4095b9",
  "frozen_predictions_sha256": "fdb6fc0c8a176b72b00d8125e4350f98a32c1afd77f53b96929c96f70295ac82",
  "golden_reference_symbol": "AAPL",
  "golden_reference_prediction": -0.1075162411705124,
  "reference_taken_from_prior_hw4": true,
  "runtime_seconds": 96.056
}
```
<!-- evidence-record:end:evidence/model_export.json -->

</details>

### A5. Requirement audit

This audit records completion of the implementation and verification work before the GitHub submission step. Its fully_completed_and_submitted field remains false until the pull request and assignees are in place.

<details>
<summary>Expand the original requirements_audit.json record</summary>

Original record name: `evidence/requirements_audit.json`. Its full content is retained below.

<!-- evidence-record:start:evidence/requirements_audit.json -->
```json
{
  "audited_at_utc": "2026-11-16T08:42:11.181584+00:00",
  "assignment_url": "https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/blob/hw_5_vishal_soni/Homeworks/HW5/hw_5.md",
  "assignment_git_blob_sha": "7ac875c69621215a5392bba6cddf223b777b21e8",
  "fully_completed_and_submitted": false,
  "code_and_local_checks": "passed after validation fix",
  "tests_passed": 106,
  "dependency_deprecation_warnings": 2,
  "ruff_check": "passed",
  "ruff_format_check": "passed",
  "native_http_valid_invalid_restart": "passed",
  "model_artifact_unchanged": true,
  "original_hw4_forecasts_independently_rechecked": 5979,
  "maximum_absolute_prediction_difference": 0.0,
  "fixed_issue": "Reject integer filing ages above 2**53-1 before float64 conversion; oversized integers previously caused HTTP 500.",
  "new_regression_inputs": [
    "filing_age_days=2**53",
    "filing_age_days=10**400"
  ],
  "new_regression_expected": "HTTP 422, inference never called",
  "dockerfile_and_verifier": "statically reviewed; actual Docker build and valid/invalid/restart verification passed on the user Mac",
  "container_evidence_present": true,
  "github_search": {
    "query": "Yixiao",
    "repository": "yang-guo/UCB-MFE-python-preprogram-2026-winter",
    "matching_pr_numbers": [
      33,
      89,
      166,
      227
    ],
    "hw5_pr_found": false,
    "note": "Recorded during the initial audit; not repeated as part of the later Docker-evidence review."
  },
  "remaining_required_steps": [
    "Submit Homeworks/HW5/Yixiao_Yang via a pull request and assign kevinramlal and raghuramkowdeed."
  ],
  "optional_not_required": [
    "GitHub Actions CI",
    "type checking"
  ],
  "latency": {
    "method": "in-process FastAPI TestClient; no TCP or Docker",
    "requests": 100,
    "median_ms": 1.601161500047965,
    "p95_ms": 5.960515999959171
  },
  "container_evidence_reviewed_at_utc": "2026-11-16T13:09:00.190178+00:00",
  "container_verification": {
    "evidence_file": "evidence/container_transcript.txt",
    "evidence_sha256": "3d9e05b4ce06d27e43390681bf4a22f20baf06f824db007de8a07e02965ac363",
    "source": "Unmodified actual execution transcript uploaded by the user from their Apple silicon Mac",
    "final_result": "PASS",
    "execution_started_at_utc": "2026-11-16T12:54:18.258775+00:00",
    "execution_finished_at_utc": "2026-11-16T12:59:24.290401+00:00",
    "docker_desktop_version": "4.99.0",
    "docker_engine_version": "29.9.3",
    "host_os_arch": "darwin/arm64",
    "server_os_arch": "linux/arm64",
    "docker_command_exit_codes": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "valid_prediction_status": 200,
    "invalid_extra_field_status": 422,
    "prediction_after_restart_status": 200,
    "image_id_before_and_after_restart": "sha256:c97486d12657b11c3f56970bbd9bfe469d073fc983ec925a185419ac5e487cfd",
    "prediction_before_and_after_restart": -0.10751624117051217,
    "model_version": "hw4-yixiao-ridge-2016q2-v1",
    "reference_prediction": -0.1075162411705124,
    "absolute_error_vs_hw4_reference": 2.220446049250313e-16,
    "same_container_stopped_and_started": true,
    "temporary_container_removed": true
  }
}
```
<!-- evidence-record:end:evidence/requirements_audit.json -->

</details>
