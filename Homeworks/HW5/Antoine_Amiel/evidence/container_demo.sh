#!/usr/bin/env bash
# Milestone 5: build the image, run it, send a valid and an invalid request,
# stop and restart the same image, and send the valid request again.
# Run from the submission folder: bash evidence/container_demo.sh
# The transcript is written to evidence/container_transcript.txt.
set -u
cd "$(dirname "$0")/.."
IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000

show() { echo "\$ $*"; "$@"; echo; echo; }
wait_ready() {
  for _ in $(seq 1 30); do curl -sf "$URL/health" > /dev/null && return 0; sleep 1; done
  echo "service did not start"; docker logs "$NAME"; return 1
}
predict() {
  show curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary "@$1"
}

{
  echo "# $(date -u '+%Y-%m-%d %H:%M UTC'), $(uname -sm), Docker $(docker version --format '{{.Server.Version}}')"
  echo
  show docker build -q -t "$IMAGE" .
  show docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
  wait_ready
  predict example_request.json
  predict evidence/invalid_request.json
  show docker stop "$NAME"
  echo "# Restart the same image"
  show docker image inspect --format '{{.Id}}' "$IMAGE"
  show docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
  wait_ready
  predict example_request.json
  show docker logs "$NAME"
  show docker stop "$NAME"
} 2>&1 | tee evidence/container_transcript.txt
