# HW5: Arav Shanghavi

This folder serves my HW4 model through `POST /predict`, tests it, and runs it in Docker.

Send the six raw HW4 factors for one stock, and the service returns the **blend score** HW4 used to pick stocks. The score runs from 0 to 1, and higher means "buy first".

On the example stock (INCY, Sep 25, 2026) the API returns `0.8302236756491833`. HW4's own code gave exactly the same value. It does the same for all 487 stocks HW4 could trade that day.

## Which HW4 model this is

In HW4, development picked the **blend, 20 names** (`results/frozen_choice.json` on my HW4 branch, commit `860d538`). For each stock on a signal date, HW4:

1. took six factors
2. turned each one into a percentile among that date's tradable stocks, flipping volatility so calm stocks rank high
3. set a missing value to 0.5
4. averaged the six percentiles

There are no regression weights. The only thing learned from data is the **cross-section the percentiles are measured against**, so that is what gets saved.

| | |
|---|---|
| Model version | `blend20-sep25-2026-v1` |
| Learned preprocessing | `CrossSectionPercentile`: one sorted array per factor from the 487 tradable stocks on Sep 25, 2026, the last HW4 signal date with a full tradable universe. Sep 30 has none because no outcome exists after it. |
| Model | `EqualWeightScore`: the mean of the six percentiles |
| Both pieces | One sklearn `Pipeline` in `blend_model.py` |
| Artifact | `artifacts/hw4_blend_v1.joblib`, 25 KB, in Git. It's a dict holding the pipeline, version, feature order, reference date, the HW4 frozen choice, and the library versions used to build it. |

**Matching HW4 exactly.** HW4 used pandas `rank(pct=True)`. That ranks with average ties, and its denominator counts only the non-missing values in each column. The transformer copies this:

- **The formula:** a value with `below` reference values under it and `same` values equal to it gets `(below + (same + 1) / 2) / n`.
- **Per-column `n`:** on Sep 25, `net_margin` has 486 values, not 487, so each factor keeps its own count.
- **Stocks in the reference:** this reproduces HW4 to the last bit. The export checks all 487 and the largest gap is 0.
- **New values:** a value HW4 never saw lands between its neighbours. Anything past the reference maximum is capped at 1.

**Limitation.** A stock is ranked against the frozen Sep 25, 2026 cross-section, not against the other stocks on its own date. That's what the version name pins down. A newer reference date would be a new version.

### Rebuilding the artifact

`freeze_artifact.py` runs once, in the HW4 environment (the repo `.venv` with Python 3.13 and the same library versions). It imports my HW4 `data_prep.py` and `signal_model.py` unchanged and reads HW4's cached panel. Then it:

1. fits the pipeline
2. stops if any of the 487 scores differs from HW4's by more than 1e-12
3. writes the artifact, `example_request.json`, `tests/frozen_reference.json` and `tests/sep25_cross_section.csv`

The HW4 code lives on the HW4 branch, so pull it out first, together with its git-ignored `outputs/` cache:

```bash
git archive hw_4_arav_shanghavi Homeworks/HW4/Arav_Shanghavi | tar -x -C /some/folder
../../../.venv/Scripts/python freeze_artifact.py --hw4-dir /some/folder/Homeworks/HW4/Arav_Shanghavi
../../../.venv/Scripts/python freeze_artifact.py --show     # print what's inside
```

The service never runs this script, never refits anything, and never touches the database. It only `joblib.load`s a file I built myself.

## Input and output contract

`POST /predict` takes one JSON object with exactly these six fields. The service rebuilds them as a one-row DataFrame in this order, whatever order the keys arrive in.

| # | Field | Meaning (units) | Allowed | null |
|---|---|---|---|---|
| 1 | `momentum_252_21` | Return from 252 to 21 sessions back, decimal | > -1 and ≤ 50 | no |
| 2 | `trend_6m` | Return over the last 126 sessions, decimal | > -1 and ≤ 25 | no |
| 3 | `volatility_63` | Annualized std of 63 daily returns, decimal | ≥ 0 and ≤ 10 | no |
| 4 | `drawdown_252` | Close / 252-session high, minus 1 | ≥ -1 and ≤ 0 | no |
| 5 | `operating_roa` | Operating income / total assets | -1e5 to 1e5 | yes |
| 6 | `net_margin` | Net income / revenue | -1e5 to 1e5 | yes |

**Why these limits:**
- **Returns** can't reach -100% while prices stay positive. The caps sit just above the largest values in any tradable HW4 row: 47.2 for 12-1 momentum and 23.5 for the 6-month return.
- **Volatility** can't be negative. The HW4 maximum was 9.56.
- **Drawdown** is between -1 and 0 by definition.
- **The two ratios** are wild in raw data (-6,240 to 34,157 across HW4). Percentiles make the size of an outlier harmless, so their cap only blocks absurd input.
- **Price factors** were never missing for a stock HW4 could trade, so null is rejected for them. HW4 handled a missing ratio as 0.5, so null is allowed there.
- **Strict types:** `"0.3"` and `true` are rejected rather than converted. JSON integers like `0` are fine. NaN and ±Infinity are rejected. A test checks that every one of the 487 real rows passes these limits.

**Responses:**
- **200** `{"prediction": float, "model_version": str}`. The prediction is the blend score in (0, 1]. **It's a ranking score, not a return.** 0.83 means near the top of the book, not +83%.
- **422** for any contract violation. The body lists `loc`, `msg` and `type` only. FastAPI's default reply repeats the input, and a NaN in the input can't be written as JSON. That would turn a clean 422 into a crash, so a small handler drops the input.
- **500** `{"detail": "Scoring error, see server log"}` when the pipeline raises or returns a non-finite number. The traceback goes to the server log only.

`example_request.json` is INCY on Sep 25, 2026. `rejected_request.json` sends volatility as text and a positive drawdown.

## Setup and running (Python 3.12)

From this folder:

```bash
py -3.12 -m venv .venv                      # macOS/Linux: python3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The docs page is at http://127.0.0.1:8000/docs. The pinned model stack (numpy 2.3.5, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.16.3, joblib 1.5.3, threadpoolctl 3.6.0) is exactly what HW4 trained with. Only the Python minor version differs (3.13 for HW4, 3.12 here). The artifact holds plain numpy arrays and sklearn base classes, and the tests load it under 3.12.

`make` wasn't installed on my machine. These are the commands behind each target:

| Target | Command |
|---|---|
| `make test` | `python -m pytest -q` |
| `make lint` | `python -m ruff check .` and `python -m ruff format --check .` |
| `make format` | `python -m ruff format .` |
| `make build` | `docker build -t mfe-hw5:local .` |
| `make run` | `docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local` |

## Tests

There are 36 tests in `tests/test_app.py`. The full output is in `proof/checks.txt`: ruff check and ruff format both pass, and 36 passed. The one warning is a Starlette deprecation coming from the template's pinned fastapi and httpx versions.

1. **Valid prediction, real model:**
   - Status 200 with exactly `prediction` and `model_version`.
   - The API equals a direct pipeline call (rel 1e-12).
   - The API equals the saved HW4 score for INCY (abs 1e-9). That reference was written by `freeze_artifact.py` from HW4's own `fixed_rule_score` before the API existed.
   - A second test scores all 487 stocks of the reference day and requires a match to HW4 within 1e-12.
   - Others check that every real row passes the contract, and that a null ratio scores exactly `(other five percentiles + 0.5) / 6`, worked out by hand.
   - One sends shuffled keys and an integer `0`.
2. **Validation:** 26 bodies must give 422:
   - each field missing
   - a seventh field
   - number as text
   - boolean
   - null price factor
   - a list in a field
   - NaN, Infinity and -Infinity tokens sent as raw text
   - each bound broken
   - bare lists of six and of five values
   - `{}`
   - an empty body

   A fixture swaps the model's `predict` for a counter, and every case asserts the counter stayed at 0. Another test shows the counter does move on a valid request, so those zeros mean something.
3. **Inference failure:** valid input with `predict` patched to raise a `ValueError` gives 500, and the exception text is not in the body. A `predict` that returns infinity also gives 500.
4. **Latency:** 15 warm-up requests, then 40 timed with `time.perf_counter`. p95 is taken by nearest rank. No millisecond limit is asserted, because laptop timings move around. The test only asserts that every request returned 200.

### Latency report

Measured on an AMD Ryzen 7 6800HS (8 cores, 16 threads), 15.2 GB RAM, Windows 11, Python 3.12.9:

| How | Median | p95 |
|---|---:|---:|
| In-process `TestClient` (the pytest test above, run saved in `proof/checks.txt`) | 2.55 ms | 2.93 ms |
| Real HTTP to the running container, `time_http.py`, one keep-alive connection (in `proof/container_session.txt`) | 2.32 ms | 2.80 ms |

The TestClient number includes validation, the one-row DataFrame, the pipeline and JSON encoding, but no network. The HTTP number goes through Docker's port mapping and uvicorn. Both are small next to anything a 90-day holding strategy cares about.

## Docker

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

The image is `python:3.12-slim` with `/app` as the working directory. It installs the pinned requirements, then copies only `app.py`, `blend_model.py` and `artifacts/`. `blend_model.py` has to be there because the saved pipeline imports it by that name.

`.dockerignore` keeps out:
- `.venv`, `.git` and caches
- `.env` files and databases
- the tests, proof files and the export script, none of which are needed to serve

Anyone with Docker and this folder can rebuild it. No database and no HW4 checkout are needed. Stop anything else on port 8000 first, or map another host port such as `-p 127.0.0.1:8001:8000`.

**Container evidence** is in `proof/container_session.txt`, recorded on Nov 14, 2026 with Docker 29.8.0. It shows:
- `docker run`, then a valid request: **200**, prediction `0.8302236756491833`, version `blend20-sep25-2026-v1`
- `rejected_request.json`: **422**, naming both broken fields
- the HTTP latency run
- `docker stop` (with `--rm` the container is gone), then a fresh `docker run` of the **same image ID** (`sha256:5171b0d8ff6f...`) and the same request: **200** with the identical prediction
- the container log

To send the requests yourself in PowerShell, use `curl.exe` so the alias for Invoke-WebRequest isn't picked up:

```bash
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@example_request.json"
curl.exe -i http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@rejected_request.json"
```

## Files

| File | What it is |
|---|---|
| `app.py` | HTTP layer only: contract, lifespan loading, `/predict`, error replies |
| `blend_model.py` | `CrossSectionPercentile` and `EqualWeightScore`, the custom sklearn pieces |
| `freeze_artifact.py` | One-off export from HW4, not used by the service |
| `artifacts/hw4_blend_v1.joblib` | The fitted pipeline plus metadata |
| `tests/test_app.py` | The 36 tests |
| `tests/frozen_reference.json` | INCY input and its HW4 score |
| `tests/sep25_cross_section.csv` | All 487 reference stocks with their HW4 scores |
| `example_request.json`, `rejected_request.json` | One good and one bad request |
| `time_http.py` | Real-HTTP latency against a running server |
| `proof/checks.txt`, `proof/container_session.txt` | Saved lint, test and container output |

## Before a real deployment

- **Fresh reference every signal date.** A real service should rebuild the percentile reference from that day's point-in-time cross-section and version each one, instead of serving one frozen day.
- **Batch scoring.** HW4 ranks about 490 stocks per date and buys the top 20, so a batch endpoint or an offline job fits better than one call per stock.
- **Trust in the artifact.** joblib can run code when it loads, so artifacts should come from a signed release with a checksum check, not a folder.
- **Operations:**
  - a non-root user and a pinned base image digest
  - keep pytest and ruff out of the runtime image
  - health and readiness endpoints
  - request IDs in structured logs
  - authentication, rate and size limits
  - worker and memory limits
- **Monitoring:**
  - input drift against the reference cross-section
  - share of null ratios
  - score distribution
  - once 90-day outcomes land, realized rank IC, the same check HW4 used in development

## Code reused

- **Template:** the HW5 template gave the Makefile, `pyproject.toml`, the `.gitignore` and `.dockerignore` starters, the test fixtures and the Dockerfile skeleton.
- **Lecture demo:** lifespan loading and the logged 500 follow `Lectures/Lecture 9/demo/app.py`. The idea of proving bad input never reaches the model comes from the demo tests.
- **HW4:** `freeze_artifact.py` calls my HW4 `data_prep.load_or_build`, `signal_model.percentile_view` and `signal_model.fixed_rule_score` unchanged.

**AI assistance:** I used Claude Code while building this submission. It helped with writing and debugging parts of the service, the tests and the Docker setup, and with editing this README.
