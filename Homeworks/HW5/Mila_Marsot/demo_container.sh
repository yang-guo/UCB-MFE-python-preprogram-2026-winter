#!/usr/bin/env bash
# The Docker demo (milestone 5).
# Open Docker Desktop, then run from this folder:
#   ./demo_container.sh 2>&1 | tee evidence/container_transcript.txt
set -euo pipefail
cd "$(dirname "$0")"

# Show a command, then run it.
show() {
  echo "\$ $*"
  "$@"
}

# Start the container and wait until the server answers.
start() {
  show docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
  for _ in $(seq 1 30); do
    curl -s -o /dev/null http://127.0.0.1:8000/openapi.json && return 0
    sleep 1
  done
  echo "The server did not start"
  exit 1
}

# Send a JSON file to the API.
send() {
  show curl -i -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@$1"
  echo
}

echo "### 1. Build the image and start the container"
show docker build -q -t mfe-hw5:local .
start

echo; echo "### 2. Good request -> 200"
show cat example_request.json
send example_request.json

echo; echo "### 3. Bad request -> 422"
show cat example_invalid_request.json
send example_invalid_request.json

echo; echo "### 4. Stop the container and start it again"
show docker stop mfe-hw5
start

echo; echo "### 5. Same good request -> 200"
send example_request.json

echo; echo "### 6. Server log, then stop"
show docker logs mfe-hw5
show docker stop mfe-hw5
