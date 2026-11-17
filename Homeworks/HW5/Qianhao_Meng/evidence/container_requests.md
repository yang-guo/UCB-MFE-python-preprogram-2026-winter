# Evidence: requests to the running container

Recorded on 2026-11-17 (17:24 to 17:26 Asia/Shanghai), macOS on Apple silicon, Docker Desktop 4.99.0 (Engine 29.9.3, linux/arm64). This is the final run, made right after the export, `make lint` and `make test` in `evidence/checks_output.md`, from the committed files. Commands run from `Homeworks/HW5/Qianhao_Meng` in two terminals: one for `make build` and `make run`, one for `curl`. Output is copied verbatim with the shell prompt shortened to `$`.

## 1. Build

```
$ make build
docker build -t mfe-hw5:local .
[+] Building 52.8s (12/12) FINISHED                        docker:desktop-linux
 => [internal] load build definition from Dockerfile                       0.0s
 => => transferring dockerfile: 288B                                       0.0s
 => [internal] load metadata for docker.io/library/python:3.12-slim        1.9s
 => [auth] library/python:pull token for registry-1.docker.io              0.0s
 => [internal] load .dockerignore                                          0.0s
 => => transferring context: 121B                                          0.0s
 => [1/6] FROM docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d50  0.0s
 => => resolve docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d50  0.0s
 => [internal] load build context                                          0.0s
 => => transferring context: 4.50kB                                        0.0s
 => CACHED [2/6] WORKDIR /app                                              0.0s
 => [3/6] COPY requirements.txt .                                          0.0s
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt              43.2s
 => [5/6] COPY app.py model_service.py ./                                  0.0s
 => [6/6] COPY artifacts/ artifacts/                                       0.0s
 => exporting to image                                                     7.7s
 => => exporting layers                                                    6.1s
 => => exporting manifest sha256:9c1f30ca8878085e06cc36a267e5f9e0a3ce4729  0.0s
 => => exporting config sha256:b9a2c16116efd526b8d6531d4942e618c6489fb888  0.0s
 => => exporting attestation manifest sha256:2da34829ca7e96a8ff972545f3fa  0.0s
 => => exporting manifest list sha256:c3db0ef340333dd7df3fe0d051964a6e799  0.0s
 => => naming to docker.io/library/mfe-hw5:local                           0.0s
 => => unpacking to docker.io/library/mfe-hw5:local                        1.5s
```

The base image was already present on this machine. `requirements.txt` had changed since the previous build, so the `pip install` layer and everything after it were rebuilt. `.dockerignore` keeps `.venv`, caches, `.env` files and databases out of the 4.50 kB build context.

## 2. Start the container

```
$ make run
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     loaded model_version=hw4-ridge-ranks-20141231-v1
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 3. Valid request, 200

```
$ curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
date: Tue, 17 Nov 2026 09:25:32 GMT
server: uvicorn
content-length: 81
content-type: application/json
x-request-id: 950872f10a3f

{"prediction":0.041160472595577126,"model_version":"hw4-ridge-ranks-20141231-v1"}
```

The prediction equals `expected_prediction` in `artifacts/reference_prediction.json`, which `export_model.py` recorded from HW4's own scoring code before the API existed.

## 4. Invalid request, 422

A numeric string instead of a number.

```
$ curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary '{"rank_momentum_252_21": "0.5", "rank_drawdown_252": 0.5, "rank_volatility_63": 0.5, "rank_operating_roa": 0.5}'
HTTP/1.1 422 Unprocessable Entity
date: Tue, 17 Nov 2026 09:25:32 GMT
server: uvicorn
content-length: 111
content-type: application/json

{"detail":[{"loc":["body","rank_momentum_252_21"],"msg":"Input should be a valid number","type":"float_type"}]}
```

The model was not called. The error body names the field and the rule without echoing the input.

Server log for sections 3 and 4:

```
INFO:     172.17.0.1:63538 - "POST /predict HTTP/1.1" 200 OK
INFO:     172.17.0.1:63552 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
```

## 5. Stop and restart the same image, then request again

Ctrl-C in the `make run` terminal, then `make run` again with the same image.

```
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [1]

$ make run
docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     loaded model_version=hw4-ridge-ranks-20141231-v1
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

```
$ curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
HTTP/1.1 200 OK
date: Tue, 17 Nov 2026 09:25:54 GMT
server: uvicorn
content-length: 81
content-type: application/json
x-request-id: bd3f80278904

{"prediction":0.041160472595577126,"model_version":"hw4-ridge-ranks-20141231-v1"}
```

Server log: `INFO:     172.17.0.1:56916 - "POST /predict HTTP/1.1" 200 OK`. The model is loaded from the image on every start, so it is available again after a restart with the same prediction.
