#!/usr/bin/env bash
# Build the image, then demonstrate valid/invalid requests over real localhost HTTP, stop the
# container, restart the SAME image and predict again. Run from the submission folder:
#   bash evidence/container_demo.sh 2>&1 | tee evidence/container_transcript.txt
set -euo pipefail
IMAGE=mfe-hw5:local
NAME=mfe-hw5
URL=http://127.0.0.1:8000/predict

# Print each command with shell quoting so it can be copied and re-run exactly.
run() { echo; printf '$'; printf ' %q' "$@"; echo; "$@"; }

start_container() {
  run docker run -d --rm --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
  for _ in $(seq 1 60); do  # wait until uvicorn has loaded the model (lifespan finished)
    curl -fs -o /dev/null http://127.0.0.1:8000/openapi.json && return 0
    sleep 0.5
  done
  docker logs "$NAME"; exit 1
}

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -sm), $(docker --version)"
run docker build -q -t "$IMAGE" .
run docker image inspect "$IMAGE" --format 'image id: {{.Id}}'

echo; echo "=== 1. First container: valid request (example_request.json)"
start_container
run curl -sS -i "$URL" -H 'Content-Type: application/json' --data-binary @example_request.json
echo
echo; echo "=== 2. Invalid request (numeric string, missing field, extra field) -> 422"
run curl -sS -i "$URL" -H 'Content-Type: application/json' --data-binary @evidence/invalid_request.json
echo
echo; echo "=== 3. Stop and restart the same image"
run docker stop "$NAME"
start_container
run docker inspect "$NAME" --format 'running container image: {{.Image}}'
run curl -sS -i "$URL" -H 'Content-Type: application/json' --data-binary @example_request.json
echo
echo; echo "=== 4. Real-HTTP latency against the container (25 sequential curl requests after 5 warm-up)"
for _ in 1 2 3 4 5; do curl -s -o /dev/null "$URL" -H 'Content-Type: application/json' --data-binary @example_request.json; done
for _ in $(seq 1 25); do
  curl -s -o /dev/null -w '%{time_total}\n' "$URL" -H 'Content-Type: application/json' --data-binary @example_request.json
done | sort -n | awk '{t[NR]=$1} END {printf "n=%d median=%.1f ms p95=%.1f ms (curl time_total)\n", NR, 1000*(t[int((NR+1)/2)]), 1000*t[int(NR*0.95+0.999)]}'
echo; echo "=== Server log"
docker logs "$NAME"
run docker stop "$NAME"
