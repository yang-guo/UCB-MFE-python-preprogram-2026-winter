#!/usr/bin/env sh
# Milestone 3 + 5 evidence: build the image, run it, send valid/invalid requests,
# measure latency through Docker's port forwarding, stop the container, start the
# SAME image again and show that the model is still served.
#
# Usage (from the submission folder, with Docker running and port 8000 free):
#   sh scripts/container_demo.sh [PYTHON] 2>&1 | tee evidence/container_transcript.txt
set -eu
PYTHON="${1:-python}"
IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000

wait_ready() {
    for _ in $(seq 1 60); do
        curl -sf "$URL/health" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "service did not become ready" >&2
    docker logs "$NAME" >&2 || true
    return 1
}

echo "# Container demo, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker --version
set -x
docker build -t "$IMAGE" .
docker image inspect "$IMAGE" --format 'image id {{.Id}}, size {{.Size}} bytes'
docker run --rm -d --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
set +x
wait_ready

echo "## 1. First container: valid and invalid requests"
sh scripts/demo_requests.sh "$URL"
"$PYTHON" scripts/measure_latency.py --url "$URL" --label "Docker container, localhost HTTP"

echo "## 2. Stop the container"
set -x
docker logs "$NAME"
docker stop "$NAME"
set +x
curl -sS --max-time 2 "$URL/health" || echo "(no response: container stopped, as expected)"

echo "## 3. Restart the same image; the model is loaded again from the image"
set -x
docker run --rm -d --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
set +x
wait_ready
set -x
curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_request.json
echo
docker logs "$NAME"
docker stop "$NAME"
