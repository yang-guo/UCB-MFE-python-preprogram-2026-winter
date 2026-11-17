#!/usr/bin/env bash
# Run from Homeworks/HW5/Louis_Maurice after `make build`. Writes evidence/container_transcript.txt:
# a valid request, a rejected request, then the same requests after restarting the same image.
set -u
OUT=evidence/container_transcript.txt
URL=http://127.0.0.1:8000/predict
start() {
  docker run -d --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local >/dev/null
  until curl -s -o /dev/null http://127.0.0.1:8000/docs; do sleep 0.5; done
}
requests() {
  echo "\$ curl -i $URL --data-binary @example_request.json"
  curl -si "$URL" -H 'Content-Type: application/json' --data-binary @example_request.json
  printf '\n\n$ curl -i %s -d %s\n' "$URL" "'{\"momentum_21\": \"0.05\"}'"
  curl -si "$URL" -H 'Content-Type: application/json' -d '{"momentum_21": "0.05"}'
  printf '\n\n'
}
{
  echo "# docker image: $(docker image inspect mfe-hw5:local --format '{{.Id}}')"
  start; requests
  docker stop mfe-hw5 >/dev/null; echo "# container stopped; restarting the same image"
  start; requests
  docker stop mfe-hw5 >/dev/null
} > "$OUT" 2>&1
cat "$OUT"
