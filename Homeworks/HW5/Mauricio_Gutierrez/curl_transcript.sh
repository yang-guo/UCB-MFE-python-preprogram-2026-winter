#!/usr/bin/env bash
# Milestone 5: checked curl requests before and after a container stop/start.
# Usage: bash curl_transcript.sh > evidence/container_curl.txt 2>&1
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=mfe-hw5:local
NAME="mfe-hw5-curl-$$"
URL=http://127.0.0.1:8000
container_id=""

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=.venv/Scripts/python.exe
  elif [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python3
  fi
fi
"$PYTHON" -c 'import json, math'

temporary=$(mktemp -d)
headers="$temporary/headers.txt"
body="$temporary/body.json"

cleanup() {
  local status=$?
  # Only remove the container created by this run.
  if [[ -n "$container_id" ]]; then
    docker rm -f "$container_id" >/dev/null 2>&1 || true
  fi
  rm -f "$headers" "$body"
  rmdir "$temporary"
  exit "$status"
}
trap cleanup EXIT

print_command() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
}

show() {
  print_command "$@"
  local status=0
  "$@" || status=$?
  printf '\n'
  return "$status"
}

wait_ready() {
  for _ in $(seq 1 120); do
    if curl -s -f --max-time 2 "$URL/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "Service did not become ready" >&2
  return 1
}

post() {
  local request_file=$1
  local expected_status=$2
  local actual_status
  print_command curl -sS -i "$URL/predict" -H 'Content-Type: application/json' \
    --data-binary "@$request_file"
  # Capture headers and body separately so assertions do not depend on formatting.
  actual_status=$(curl -sS --max-time 15 -D "$headers" -o "$body" -w '%{http_code}' \
    "$URL/predict" -H 'Content-Type: application/json' --data-binary "@$request_file")
  cat "$headers" "$body"
  printf '\n'
  if [[ "$actual_status" != "$expected_status" ]]; then
    echo "Expected HTTP $expected_status; received $actual_status" >&2
    return 1
  fi
  "$PYTHON" - "$body" "$expected_status" artifacts/reference_prediction.json <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if sys.argv[2] == "200":
    with open(sys.argv[3], encoding="utf-8") as stream:
        reference = json.load(stream)
    assert set(result) == {"prediction", "model_version"}, result
    prediction = result["prediction"]
    assert type(prediction) in (int, float) and math.isfinite(prediction), result
    assert result["model_version"] == reference["model_version"], result
    assert math.isclose(
        prediction, reference["prediction"],
        rel_tol=reference["relative_tolerance"],
        abs_tol=reference["absolute_tolerance"],
    ), result
else:
    assert isinstance(result, dict) and isinstance(result.get("detail"), list), result
    assert result["detail"], result
print("Response check passed.")
PY
}

echo "=== 1. Start a container from the image ==="
show docker image inspect --format '{{.Id}}' "$IMAGE"
print_command docker run -d --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
container_id=$(docker run -d --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE")
echo "$container_id"
image_id=$(docker inspect --format '{{.Image}}' "$container_id")
wait_ready

echo "=== 2. Valid request ==="
show cat example_request.json
post example_request.json 200

echo "=== 3. Invalid request: negative volatility_63 ==="
show cat invalid_request.json
post invalid_request.json 422

echo "=== 4. Stop the container ==="
show docker stop "$container_id"
print_command curl -sS --max-time 3 "$URL/health"
connection_status=0
curl -sS --max-time 3 "$URL/health" || connection_status=$?
if [[ "$connection_status" != 7 ]]; then
  echo "Expected curl connection failure 7 while stopped; got $connection_status" >&2
  exit 1
fi
echo "Expected connection failure confirmed."

echo "=== 5. Start the same container and verify the image ==="
show docker start "$container_id"
restarted_image=$(docker inspect --format '{{.Image}}' "$container_id")
if [[ "$restarted_image" != "$image_id" ]]; then
  echo "Image changed during restart" >&2
  exit 1
fi
echo "Same image confirmed: $image_id"
wait_ready

echo "=== 6. Valid request after restart ==="
post example_request.json 200

echo "=== 7. Server log and cleanup ==="
show docker logs "$container_id"
show docker rm -f "$container_id"
container_id=""
echo "All container and response checks passed."
