#!/usr/bin/env bash
# Reproduce evidence/container_transcript.txt (run from the submission folder after `make build`):
#   bash evidence/container_demo.sh 2>&1 | tee evidence/container_transcript.txt
set -u
run() { echo; echo "\$ $*"; "$@"; echo; }
wait_ready() { until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 0.5; done; }

echo "# HW5 container demo, $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
run docker image ls mfe-hw5:local
run docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
wait_ready
run curl -s http://127.0.0.1:8000/health
echo "## 1. Valid request (example_request.json = stock A on 2026-04-09; HW4 prediction 0.004700122008907656)"
run curl -sS -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
echo "## 2. Invalid request: numeric string, out-of-range value, extra field -> 422"
run curl -sS -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @evidence/invalid_request.json
echo "## 3. Invalid request: NaN -> 422 (not a crash)"
run curl -sS -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary "$(sed 's/0.4098442993470618/NaN/' example_request.json)"
echo "## 4. Real localhost HTTP latency against the container"
run .venv/Scripts/python.exe measure_latency.py -n 200
echo "## 5. Server log (model loaded once at startup)"
echo "$ docker logs mfe-hw5 | head -8   (then one access-log line per request)"; docker logs mfe-hw5 2>&1 | head -8; echo "... $(docker logs mfe-hw5 2>&1 | grep -c "POST /predict") POST /predict lines in total"
echo "## 6. Stop the container, then start a NEW container from the same image"
run docker stop mfe-hw5
run docker ps -a --filter name=mfe-hw5 --format '{{.Names}} {{.Status}}'
run docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
wait_ready
echo "## 7. Same request after restart: same prediction and model version"
run curl -sS -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json
run docker stop mfe-hw5
