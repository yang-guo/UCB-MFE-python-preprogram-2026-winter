#!/usr/bin/env bash
# Milestone 5 evidence: build the image, send valid and invalid requests to the running
# container, stop and restart the same image, request again, and time real HTTP calls.
#
# Needs a running Docker daemon, curl, and a free host port (default 8000).  From the
# submission folder:
#     bash scripts/container_demo.sh 2>&1 | tee evidence/container_transcript.txt
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=mfe-hw5:local
NAME=mfe-hw5
PORT=${PORT:-8000}
URL="http://127.0.0.1:${PORT}"
export PS4='$ '

run() { (set -x; "$@"); }  # echo each command (with quoting) before running it

wait_ready() {
    for _ in $(seq 1 60); do
        curl -fsS "${URL}/health" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "service did not become ready" >&2
    docker logs "$NAME" >&2
    return 1
}

echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) | $(uname -sr) | Docker $(docker version --format '{{.Server.Version}}')"

echo; echo "## Build"
run docker build -t "$IMAGE" .
run docker image ls "$IMAGE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo; echo "## Run the container"
run docker run -d --rm --name "$NAME" -p "127.0.0.1:${PORT}:8000" "$IMAGE"
wait_ready
run docker ps --filter "name=${NAME}" --format '{{.Names}}  {{.Image}}  {{.Status}}  {{.Ports}}'

echo; echo "## Valid request (example_request.json: AAPL on 2015-12-31, HW4 score 0.10577686157588898)"
run cat example_request.json
run curl -i -sS "${URL}/predict" -H 'Content-Type: application/json' --data-binary @example_request.json
echo

echo; echo "## Invalid request: numeric string instead of a number -> 422"
BAD_TYPE=$(sed 's/"vol_63_csr": [^,]*/"vol_63_csr": "-0.19"/' example_request.json)
run curl -i -sS "${URL}/predict" -H 'Content-Type: application/json' --data-binary "$BAD_TYPE"
echo

echo; echo "## Invalid request: NaN and a missing field -> 422"
BAD_NAN=$(sed -e 's/"ret_21_csr": .*/"ret_21_csr": NaN/' -e '/"vol_63_csr"/d' example_request.json)
run curl -i -sS "${URL}/predict" -H 'Content-Type: application/json' --data-binary "$BAD_NAN"
echo

echo; echo "## Stop the container, then start the same image again"
run docker stop "$NAME"
run docker ps --all --filter "name=${NAME}" --format '{{.Names}} {{.Status}}'
run docker run -d --rm --name "$NAME" -p "127.0.0.1:${PORT}:8000" "$IMAGE"
wait_ready
run curl -sS "${URL}/health"
echo
run curl -i -sS "${URL}/predict" -H 'Content-Type: application/json' --data-binary @example_request.json
echo

echo; echo "## Real HTTP latency against the container (curl, new TCP connection per request)"
for _ in $(seq 1 10); do  # warm-up
    curl -sS -o /dev/null "${URL}/predict" -H 'Content-Type: application/json' \
        --data-binary @example_request.json
done
for _ in $(seq 1 100); do
    curl -sS -o /dev/null -w '%{http_code} %{time_total}\n' "${URL}/predict" \
        -H 'Content-Type: application/json' --data-binary @example_request.json
done | awk '$1 != 200 { bad++ } { print $2 * 1000 }
            END { if (bad) { print "non-200 responses: " bad > "/dev/stderr"; exit 1 } }' |
    sort -n | awk '{ a[NR] = $1 }
        END { med = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
              p95 = a[int(0.95 * NR + 0.999999)]
              printf "100 requests after 10 warm-up: median %.2f ms, p95 %.2f ms, max %.2f ms\n",
                     med, p95, a[NR] }'

echo; echo "## Container log (startup line and request log)"
# sed reads the whole stream; head would close the pipe early and, under pipefail,
# end the script before the container is stopped.
docker logs "$NAME" 2>&1 | sed -n '1,12p'

echo; echo "## Clean up"
run docker stop "$NAME"
echo "# demo finished OK"
