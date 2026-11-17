#!/usr/bin/env bash
# Milestone 5 evidence: valid request, invalid request, then the same after a restart.
# Build the image first (make build). Then run:
#     bash demo_container.sh 2>&1 | tee evidence/container_transcript.txt
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000

start() {
    docker run --rm -d --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE" > /dev/null
    for _ in $(seq 1 60); do
        if curl -s -o /dev/null "$URL/health"; then return 0; fi
        sleep 0.5
    done
    echo "The service did not start" >&2
    docker logs "$NAME" >&2
    exit 1
}

stop() {
    docker stop "$NAME" > /dev/null
    # --rm removes the container a moment after it stops. Wait, so the name is free again.
    while docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; do sleep 0.2; done
}

request() {
    echo "\$ curl -i $URL/predict -H 'Content-Type: application/json' --data-binary @$1"
    curl -s -i "$URL/predict" -H 'Content-Type: application/json' --data-binary "@$1"
    echo
    echo
}

trap 'docker stop "$NAME" > /dev/null 2>&1 || true' EXIT

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') | image $(docker image inspect -f '{{.Id}}' "$IMAGE")"
echo
echo "=== 1. Start the container"
start
docker ps --filter "name=$NAME" --format 'container {{.ID}} | {{.Image}} | {{.Status}} | {{.Ports}}'
echo
echo "=== 2. Valid request (expect 200, a prediction, and the model version)"
request example_request.json
echo "=== 3. Invalid request: a numeric string and an extra field (expect 422)"
request invalid_request.json
echo "=== 4. Stop the container, then start the same image again"
stop
echo "stopped and removed"
start
docker ps --filter "name=$NAME" --format 'container {{.ID}} | {{.Image}} | {{.Status}} | {{.Ports}}'
echo
echo "=== 5. Valid request after the restart (the model is in the image, so it is still there)"
request example_request.json
echo "=== 6. Server log of the second container"
docker logs "$NAME" 2>&1 | tail -n 5
