#!/usr/bin/env bash
# Milestone 3 + 5 end to end: build, run, request, stop, restart, request again.
# Usage (Docker running, host port 8000 free): scripts/container_demo.sh | tee evidence/container_transcript.txt
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000

wait_ready() {
  for _ in $(seq 1 60); do
    curl -sf "$URL/health" >/dev/null && return 0
    sleep 0.5
  done
  echo "service did not become ready" >&2; docker logs "$NAME" >&2; return 1
}

echo "=== docker version"; docker version --format 'client {{.Client.Version}} / server {{.Server.Version}}'
echo "=== docker build -t $IMAGE ."
docker build -q -t "$IMAGE" .
echo "image digest: $(docker image inspect --format '{{.Id}}' "$IMAGE")"

echo; echo "=== run 1: docker run -d --rm --name $NAME -p 127.0.0.1:8000:8000 $IMAGE"
docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
wait_ready
scripts/request_demo.sh "$URL"
python scripts/http_latency.py "$URL" 200

echo; echo "=== docker stop $NAME (container removed by --rm)"
docker stop "$NAME"
curl -s -o /dev/null "$URL/health" && echo "unexpected: still up" || echo "service is down"

echo; echo "=== run 2: same image restarted"
docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
wait_ready
echo "\$ curl -s $URL/predict --data-binary @example_request.json"
curl -s "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_request.json
echo
echo; echo "=== container logs (run 2)"; docker logs "$NAME" 2>&1 | tail -5
docker stop "$NAME" >/dev/null
