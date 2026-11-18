#!/usr/bin/env sh
# Send the valid and invalid example requests to a running service and print the
# full HTTP exchange (used for evidence/*.txt). Usage: sh scripts/demo_requests.sh [URL]
URL="${1:-http://127.0.0.1:8000}"
set -x
curl -sS -i "$URL/health"
echo
curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_request.json
echo
curl -sS -i "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_invalid_request.json
echo
