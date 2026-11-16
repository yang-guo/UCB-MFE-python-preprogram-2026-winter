#!/usr/bin/env bash
# Builds the image, sends a valid and an invalid request, restarts the
# container and sends the valid request again. Output: evidence/container_requests.txt
set -u
mkdir -p evidence
OUT=evidence/container_requests.txt

wait_ready() {
  for _ in $(seq 1 30); do
    curl -s -o /dev/null http://127.0.0.1:8000/docs && return 0
    sleep 1
  done
  echo "server did not start" && return 1
}

request() {
  echo "\$ curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @$1"
  curl -s -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @"$1"
  echo; echo
}

docker build -t mfe-hw5:local . || exit 1
{
  echo "=== $(date) - container started"
  docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
  wait_ready
  request example_request.json
  request invalid_request.json

  echo "=== stop and restart the same image"
  docker stop mfe-hw5
  docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
  wait_ready
  request example_request.json
  docker stop mfe-hw5
} 2>&1 | tee "$OUT"
