#!/usr/bin/env bash
# Milestone 5: build the image, call the running container, restart it and call it again.
# Every command is printed before it runs, and the script stops if a status code is not the
# expected one. The saved transcript was made with:
#     ./container_demo.sh > evidence/container_demo.txt 2>&1
# To use another host port: HOST_PORT=8080 ./container_demo.sh
set -euo pipefail
cd "$(dirname "$0")"

HOST_PORT="${HOST_PORT:-8000}"
URL="http://127.0.0.1:$HOST_PORT"
# The command behind `make run`, detached so that the script can go on.
RUN="docker run -d --rm --name mfe-hw5 -p 127.0.0.1:$HOST_PORT:8000 mfe-hw5:local"
POST="curl -si $URL/predict -H 'Content-Type: application/json' --data-binary"
IMAGE_ID="docker inspect mfe-hw5 --format '{{.Image}}'" # which image the container runs

show() { # print a command, then run it
  printf '\n$ %s\n' "$1"
  eval "$1"
}
expect() { # expect <status> <command>: as show, but fail unless the HTTP status matches
  local output
  printf '\n$ %s\n' "$2"
  output=$(eval "$2")
  printf '%s\n' "$output"
  [[ $output == "HTTP/1.1 $1"* ]] || { echo "expected HTTP $1" >&2; exit 1; }
}
wait_until_ready() { # the server answers /health once the model is loaded
  for _ in $(seq 150); do
    curl -sf -o /dev/null "$URL/health" && return
    sleep 0.2
  done
  echo "the service did not become ready" >&2
  exit 1
}

if [ -n "$(docker ps -aq --filter 'name=^mfe-hw5$')" ]; then
  echo "A container named mfe-hw5 already exists. Stop it first: docker stop mfe-hw5" >&2
  exit 1
fi
trap 'docker stop mfe-hw5 >/dev/null 2>&1 || true' EXIT

echo "# 1. Build the image and start a container."
show "docker build -q -t mfe-hw5:local ."
show "$RUN"
show "$IMAGE_ID"
wait_until_ready
show "curl -s $URL/health"

printf '\n\n# 2. Valid request: NVDA on 2026-09-28, a real row of my HW4 experiment.\n'
expect 200 "$POST @example_request.json"

printf '\n\n# 3. Invalid requests are rejected with 422: a raw dollar volume where its rank is\n'
echo "#    expected, a number sent as a string, and a NaN."
expect 422 "sed 's/\"dollar_vol_63\": 0.49947286/\"dollar_vol_63\": 2500000000/' example_request.json | $POST @-"
expect 422 "sed 's/\"mom_21\": 0.32076964/\"mom_21\": \"0.32076964\"/' example_request.json | $POST @-"
expect 422 "sed 's/\"mom_63\": 0.3798102/\"mom_63\": NaN/' example_request.json | $POST @-"

printf '\n\n# 4. Stop the container, start the same image again: the model is still served.\n'
show "docker stop mfe-hw5"
show "$RUN"
show "$IMAGE_ID"
wait_until_ready
expect 200 "$POST @example_request.json"
show "docker logs mfe-hw5"

printf '\n\n# 5. Latency over real localhost HTTP into the container, measured by curl\n'
echo "#    (time_total, a new connection per request, 200 requests after 20 warm-up)."
for _ in $(seq 220); do
  curl -s -o /dev/null -w '%{http_code} %{time_total}\n' "$URL/predict" \
    -H 'Content-Type: application/json' --data-binary @example_request.json
done | tail -n 200 | sort -n -k 2 | awk '
  $1 != 200 { failed++ }
  { seconds[NR] = $2 }
  END {
    if (failed) { print failed " of 200 requests did not return 200"; exit 1 }
    printf "200 of 200 returned 200; median %.1f ms, p95 %.1f ms\n",
      500 * (seconds[100] + seconds[101]), 1000 * seconds[190]
  }'

show "docker stop mfe-hw5"
