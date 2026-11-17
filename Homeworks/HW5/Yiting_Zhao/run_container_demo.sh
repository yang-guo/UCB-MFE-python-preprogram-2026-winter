#!/usr/bin/env bash
# Builds the image, sends valid/invalid requests, restarts, and re-checks.
# Output is saved to evidence/container_transcript.txt.
set -u
mkdir -p evidence
exec > >(tee evidence/container_transcript.txt) 2>&1

run() { echo; echo "\$ $*"; "$@"; }
wait_ready() {
  for _ in $(seq 1 30); do
    curl -s http://127.0.0.1:8000/health >/dev/null && return 0
    sleep 1
  done
  echo "server did not start"; docker logs mfe-hw5; exit 1
}

date
run docker build -t mfe-hw5:local .
docker rm -f mfe-hw5 >/dev/null 2>&1

echo; echo "===== Start container ====="
run docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
wait_ready

echo; echo "===== Valid request (expect 200) ====="
run curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json

echo; echo; echo "===== Invalid request: numeric string + extra field (expect 422) ====="
run curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @invalid_request.json

echo; echo; echo "===== Stop and restart the same image ====="
run docker stop mfe-hw5
run docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
wait_ready

echo; echo "===== Valid request after restart (expect 200, same prediction) ====="
run curl -i http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @example_request.json

echo; echo; run docker stop mfe-hw5
