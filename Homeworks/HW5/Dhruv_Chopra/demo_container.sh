#!/usr/bin/env bash
# Milestone 5 evidence: valid + invalid requests to the container, then stop it,
# start a new container from the same image, and request again.
# Usage (after `make build`):  ./demo_container.sh | tee evidence/container_transcript.txt
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000

run() { echo; echo "\$ $*"; "$@"; echo; }

start() {
  run docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
  for _ in $(seq 1 50); do curl -sf "$URL/health" >/dev/null && return; sleep 0.2; done
  echo "container did not become ready" >&2
  exit 1
}

echo "# $(date -u '+%Y-%m-%d %H:%M:%S UTC'), $(docker version --format 'Docker {{.Server.Version}}')"
run docker image ls "$IMAGE"
start

echo "## 1. Valid request (AAPL, 2026-09-28 ranks from HW4)"
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' \
  --data-binary @example_request.json

echo "## 2. Invalid request: numeric string -> 422"
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' \
  --data-binary "$(sed 's/"mom_5": [0-9.e-]*/"mom_5": "0.5"/' example_request.json)"

echo "## 3. Invalid request: only 2 of 20 features -> 422"
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' \
  --data '{"mom_5": 0.5, "mom_21": 0.5}'

echo "## 4. Stop the container, start a new one from the same image, request again"
run docker stop "$NAME"
start
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' \
  --data-binary @example_request.json

echo "## Container log after restart (model loaded once at startup)"
run docker logs "$NAME"
docker stop "$NAME" >/dev/null
