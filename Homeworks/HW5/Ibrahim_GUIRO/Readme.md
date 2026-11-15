# HW5 — Serve and containerize the HW4 model

This FastAPI application serves the final 90-day stock-return model from HW4
(commit `ff8f868`). The fitted linear regression is stored in `hw4_model.joblib`.
The application loads it once per server process at startup. Serving requires
only the saved model and its Python dependencies. The HW4 database is used
during export only.

## API contract

`POST /predict` accepts a JSON object with one field, `score_rank`. This is
the percentile rank of the HW4 risk-adjusted momentum score among eligible
stocks in the same signal-date cohort. The caller computes the rank using the
HW4 procedure. The model expects that single feature in a DataFrame column
named `score_rank` (called `score_percentile` in the HW4 README).

The value must be a finite JSON number with `0 < score_rank <= 1`. HW4
computes it with `rank(pct=True)`, which gives the lowest-ranked stock `1/n`
(greater than 0) and the highest exactly 1, so values outside this range
cannot come from the HW4 procedure.
Missing fields, extra fields, numeric strings, booleans, and invalid values
return HTTP 422. Unexpected inference failures return HTTP 500, with
exception details recorded in the server logs.

A successful request returns HTTP 200, a predicted 90-day return expressed
as a decimal, and the model version. For the input in `example_request.json`
the response is:

```json
{
  "prediction": 0.032419466139285105,
  "model_version": "hw4-ff8f868"
}
```

## Model and reference prediction

`export_hw4_model.py` reproduces the final HW4 fit using the code from
`ff8f868` and the course database. The model has no learned feature
transformer. The percentile rank is computed upstream in HW4.

The export needs `duckdb` and the course database at
`Lectures/modeling/data/equities_top2000.duckdb` (from bCourses, see
`Lectures/modeling/README.md`), which the serving environment does not
include. To regenerate `hw4_model.joblib`, `example_request.json` and
`reference_prediction.json`, run it from the repository root with the course
environment:

```bash
.venv/bin/python Homeworks/HW5/Ibrahim_GUIRO/export_hw4_model.py
```

`reference_prediction.json` records a direct model prediction made during
export, before the API call, for ticker `A` with a signal dated 2016-03-31.
The automated test compares the API response with both this recorded
prediction and a direct call to the saved model, using an absolute
tolerance of `1e-12`.

## Local setup and checks

Run these commands from this HW5 submission folder using Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q -s
```

All 14 tests passed. They cover the reference prediction, input validation
before inference, a simulated inference failure, and endpoint latency.

On a MacBook Air M5, the median latency was **0.861 ms** and the p95 latency
was **1.150 ms**. I measured 30 successful requests after 5 warm-up requests
with `time.perf_counter` and the in-process FastAPI `TestClient`.

## Docker demonstration

From this folder, build the image and run the container.

```bash
docker build -t mfe-hw5:local .
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
```

In another terminal in the same folder, send a valid request and then a
request with a string in place of a numeric rank.

```bash
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @example_request.json

curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary '{"score_rank":"0.42"}'
```

To reproduce the restart check, stop the container with `Ctrl+C`, start the
same image again with the `docker run` command above, and repeat the valid
request once startup is complete.

- [Valid Docker request](evidence/docker-valid.png)
- [Rejected Docker request](evidence/docker-invalid.png)
- [Successful request after restart](evidence/docker-restart.png)

## Before production deployment

The caller's rank calculation would need to use the same stock universe and
scoring procedure as HW4. I would also add access controls, monitor request
errors and latency, and track prediction quality as realized returns become
available.

## Attribution

The application, tests, Dockerfile, Makefile and configuration files start
from the course template in `Homeworks/HW5/template`.

I used AI to help me work through the assignment, debug issues, and draft this README. I
understand how the code works and can explain each step, from exporting the HW4 model to serving
it through the API, testing it, and running it in Docker.
