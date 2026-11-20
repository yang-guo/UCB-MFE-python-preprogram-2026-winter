# HW5: Serving the HW4 Ridge Model

## Model

I used the final `Ridge(alpha=10)` model from my HW4 experiment. I fit it on the 2010–2015 development data using returns completed before April 1, 2016. I exported the fitted model, the 1st and 99th percentile feature bounds, and the fitted `StandardScaler` together in `hw4_ridge.joblib`. FastAPI loads the bundle once at startup, then clips and scales each input before predicting. Serving requires only the saved bundle and the pinned dependencies. [HW4 notebook and submission](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/pull/235)

## API

`POST /predict` takes five required finite JSON numbers. The service builds a DataFrame in the same column order as HW4. All feature windows use trading sessions, and decimal values such as `0.23` represent 23%.

| Order | Feature | Definition and units | Accepted range |
| --- | --- | --- | --- |
| 1 | `momentum_63d` | Adjusted price return over 63 sessions, decimal | ≥ −1 |
| 2 | `volatility_21d` | Standard deviation of 21 daily returns × √252, annualized decimal | ≥ 0 |
| 3 | `beta_126d` | Beta against `^SP500TR` over 126 sessions, unitless | Any finite value |
| 4 | `max_drawdown_63d` | Worst peak-to-trough return across 64 adjusted closes, decimal | −1 to 0 |
| 5 | `relative_momentum_63d` | Stock momentum minus `^SP500TR` momentum over 63 sessions, decimal | Any finite value |

A successful request returns `200` with a finite `prediction` and `model_version` (`hw4-ridge-alpha10-v1`). The prediction is a gross decimal return from the next trading session’s close to the first valid close on or after 90 calendar days from entry. Missing or extra fields, numeric strings, booleans, nulls, non-finite values, and values outside the stated ranges return `422`. Unexpected inference failures return `500` with a generic message, and the details go to the server log. [Example request](example_request.json)

## How to Run

Run these commands from `Homeworks/HW5/Nico_Morrone/`. The requirements pin the model libraries to the versions used for the HW4 export, including PyArrow.

```bash
conda create -n mfe-hw5 python=3.12 pip
conda activate mfe-hw5
python -m pip install -r requirements.txt
make lint
make test
python -m pytest -q -s tests/test_app.py::test_predict_performance
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Local API documentation is at `http://127.0.0.1:8000/docs`. To reproduce the container check, run the following in a Terminal with Docker running. I used host port `8001` and container port `8000`.

```bash
make build
docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8001:8000 mfe-hw5:local
docker logs mfe-hw5
```

Once startup completes, send both requests:

```bash
curl -i http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' --data-binary @example_request.json
curl -i http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' --data '{}'
docker stop mfe-hw5
```

Start another container with the same `docker run` command, wait for startup, and repeat the valid request. Stop it again with `docker stop mfe-hw5`.

## Results

All 17 tests and both Ruff checks passed. I compared the API prediction with a direct model call and with the full-precision prediction saved from the HW4 notebook for ticker `A` on March 31, 2016. On an Apple M2 running macOS 26.6 and Python 3.12.16, 30 sequential in-process `TestClient` requests after five warmups had a median latency of **2.039 ms** and a p95 of **2.221 ms**. [HW4 reference](hw4_reference.json) · [Test and lint output](verification.txt)

The container returned `200`, then `422` for the empty request, then `200` after restart. Both successful requests returned `0.03241820288799492` with the same model version. [Container transcript](container-transcript.txt)


I chose the input bounds based on how each feature is calculated. With nonnegative prices, momentum cannot fall below −100%, and drawdown ranges from −100% to 0%. Volatility is nonnegative because it is based on a standard deviation. Beta and relative momentum can take any finite value because their formulas give them no fixed bounds.

## Before Production

I would add authentication, readiness checks, and monitoring for stale features, prediction drift, and inference errors. I would also verify the model artifact before loading it and run the container as a non-root user.

I adapted the application structure, test fixtures, Dockerfile, and Makefile from the course template. I used ChatGPT to help with implementation, debugging, and writing this README. [Course HW5 template](https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter/tree/main/Homeworks/HW5/template)
