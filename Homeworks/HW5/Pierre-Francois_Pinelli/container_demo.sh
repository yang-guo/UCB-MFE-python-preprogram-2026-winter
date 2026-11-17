#!/usr/bin/env bash
# One command for the container evidence. It needs only Docker and curl.
#   1-6. build the image, start the container, send a valid and an invalid request,
#        stop it, start the same image again and send the valid request again
#   7.   run the lint and the tests (with the latency report) inside the same image
# Everything printed is saved in evidence/container_transcript.txt.
# Run it with: bash container_demo.sh   (if port 8000 is busy: PORT=8001 bash container_demo.sh)
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=mfe-hw5:local
NAME=mfe-hw5
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"

mkdir -p evidence
exec > >(tee evidence/container_transcript.txt) 2>&1

# print a command, then run it
run() {
  echo
  echo "\$ $*"
  "$@"
}

start_container() {
  run docker run -d --rm --name "$NAME" -p "127.0.0.1:$PORT:8000" "$IMAGE"
  # wait until the server in the container answers
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$URL/health"; then return 0; fi
    sleep 0.5
  done
  echo "the server did not start"
  docker logs "$NAME" || true
  exit 1
}

stop_container() {
  run docker stop "$NAME"
  # --rm deletes the container after the stop, we wait until its name is free again
  while docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; do sleep 0.2; done
}

# send one JSON file to /predict and show the full answer
send() {
  echo
  echo "\$ curl -i -s $URL/predict -H 'Content-Type: application/json' --data-binary @$1"
  curl -i -s "$URL/predict" -H 'Content-Type: application/json' --data-binary "@$1"
  echo
}

# run a check inside the image, with this folder mounted (same Python and same versions)
in_image() {
  run docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/work" -w /work "$IMAGE" "$@"
}

echo "date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "machine: $(uname -srm)"
run docker --version
run docker build -t "$IMAGE" .
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo
echo "===== 1. start the container"
start_container

echo
echo "===== 2. valid request: AAPL at the 2026-08-31 signal, expect 200"
send example_request.json

echo
echo "===== 3. invalid request: z_beta_252 = 0.9, outside [-0.5, 0.5], expect 422"
send example_invalid_request.json

echo
echo "===== 4. what the server logged"
run docker logs "$NAME"

echo
echo "===== 5. stop the container, then start the same image again"
stop_container
start_container
run docker images "$IMAGE"

echo
echo "===== 6. same valid request after the restart, expect the same prediction"
send example_request.json
run curl -s "$URL/health"
echo
stop_container

echo
echo "===== 7. lint and tests inside the same image (latency = in-process TestClient)"
in_image python -m ruff check --no-cache .
in_image python -m ruff format --check --no-cache .
in_image python -m pytest -q -s -p no:cacheprovider

echo
echo "done, transcript saved in evidence/container_transcript.txt"
