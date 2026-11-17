#!/usr/bin/env bash
# Milestone 5 evidence: valid request, invalid request, stop/restart, valid again.
# Run from this folder in Git Bash (or any bash) after `docker build -t mfe-hw5:local .`
#   bash container_demo.sh > evidence/container_transcript.txt 2>&1
set -u
IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000

wait_ready() {
  for _ in $(seq 1 40); do
    curl -fsS "$URL/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "container did not become ready"; return 1
}

run() { echo; printf '$'; printf ' %q' "$@"; echo; "$@"; echo; }  # %q keeps the quoting

echo "# HW5 container transcript - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
run docker image ls "$IMAGE" --format '{{.Repository}}:{{.Tag}}  id={{.ID}}  size={{.Size}}'

echo "## start the container (detached here so the script can continue)"
run docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
wait_ready

echo "## 1. valid request (KLAC, 2016-03-31 signal from the HW4 final test)"
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_request.json

echo "## 2. invalid request: numeric string + extra field -> 422, no prediction"
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' \
  --data-binary '{"r_ey": "0.5", "sector": "Technology"}'

echo "## 3. invalid request: NaN in an otherwise valid body -> 422"
sed 's/"r_ey": [0-9.e-]*/"r_ey": NaN/' example_request.json > /tmp/hw5_nan.json
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary @/tmp/hw5_nan.json

echo "## 4. stop the container, start the SAME image again"
run docker stop "$NAME"
run docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
wait_ready
run curl -sS -i "$URL/health"

echo "## 5. valid request after restart (same prediction and model_version)"
run curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_request.json

echo "## container log (startup lines + access log)"
run docker logs "$NAME"
