# HW 5

I'm serving the gradient boosting model from HW4 version 2. It's the January 2, 2026 fit, which generated the 2026 predictions in the HW4 final test.

`model_version`: `hw4-v2-boosting-fit-2026-01-02`

## Input and output

`POST /predict` takes these 8 fields, all required. The table lists them in the order the model was trained on, but the JSON keys can come in any order. Each is a JSON number between -0.5 and 0.5: the stock's rank on that HW4 feature among the day's eligible stocks, `rank(pct=True) - 0.5`, and 0 when the raw value is missing (HW4 section 1.6).

| Field | Rank of |
|---|---|
| `ret_1m` | 1-month return |
| `ret_12_1` | 12-1 month momentum |
| `vol_3m` | 3-month volatility of daily returns |
| `dist_high_1y` | distance to the 1-year high |
| `size` | market-cap proxy |
| `gross_profitability` | gross profit / total assets |
| `roa` | net income / total assets |
| `debt_to_assets` | total debt / total assets |

A rank can't be outside [-0.5, 0.5], so those values get a 422, as do missing or extra fields, strings, booleans, NaN and infinity. The ranking needs all of the day's stocks, so it is done before calling the API. Nothing in it is fitted, so the only saved object is the model.

Output: `prediction` is the predicted excess return over the next 90 calendar days against the average eligible stock, as a fraction (0.01 = 1%); `model_version` is the version above. If the model fails, the response is a 500 and the details go only to the server log.

## Model artifact

`artifacts/hw4_v2_boosting_2026.joblib` (385 KB) is loaded once at startup. It was saved with the versions my HW4 notebook ran with (Python 3.12, scikit-learn 1.8.0, pandas 2.3.3, numpy 2.3.5), which are the ones in `requirements.txt` and the Docker image.

`export_model.py` recreates it and `example_request.json` by running the HW4 notebook's code for sections 1-2. It needs `equities_top2000.duckdb` in `Lectures/modeling/data/`, my HW4 notebook, and `pip install duckdb==1.5.6 matplotlib==3.10.7`.

## Commands

Setup, lint and tests:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make lint
make test
```

Run locally (docs at http://127.0.0.1:8000/docs):

```sh
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Run in Docker:

```sh
make build
make run
```

Both servers run in the foreground on port 8000, so stop Uvicorn (Ctrl+C) before `make run`. Send the request from a second terminal:

```sh
curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
```

## Tests and results

- The reference in the valid-input test is the prediction from HW4's `predict_scores` for `example_request.json` (AAPL on 2026-09-29), printed by `export_model.py`. The tolerance is 1e-9: the library versions are the same, so only float rounding could differ.
- Latency is measured in-process with `TestClient` (no network): 10 warm-up requests, then 200 timed with `time.perf_counter`. On an Apple M4 (16 GB, macOS 26.3), with Docker Desktop running, the median was 5-6 ms and the p95 was 7-23 ms across runs. Run `python -m pytest -q -s -k performance` to see it.
- Test and lint output: [evidence/checks.txt](evidence/checks.txt). Container requests (valid, invalid, after a restart): [evidence/container_transcript.txt](evidence/container_transcript.txt).

## Before a real deployment

- I'd set up a daily job to compute the ranks across all eligible stocks, then retrain the model yearly as in HW4 and track which fit is in production.
- Authentication, and an image that doesn't run as root or include the test tools.

## Code reuse and AI

The lifespan loading and the try/except that returns a 500 follow the Lecture 9 demo `app.py`. I used Claude Code to assist in writing the code and tests.
