#!/usr/bin/env bash
# Milestone 5 evidence: valid + invalid request to a running service.
# Usage: scripts/request_demo.sh [base_url]   (default http://127.0.0.1:8000)
set -u
URL="${1:-http://127.0.0.1:8000}"
cd "$(dirname "$0")/.."

echo "\$ curl -s $URL/health"
curl -s "$URL/health"; echo; echo

echo "\$ curl -i $URL/predict -H 'Content-Type: application/json' --data-binary @example_request.json"
curl -i -s "$URL/predict" -H 'Content-Type: application/json' --data-binary @example_request.json
echo; echo

INVALID='{"pairs_divergence_z": null, "fcf_quality_score": "0.6", "revenue_growth_vs_industry": 3.09, "momentum_percentile": 97}'
echo "\$ curl -i $URL/predict -H 'Content-Type: application/json' --data-binary '$INVALID'"
curl -i -s "$URL/predict" -H 'Content-Type: application/json' --data-binary "$INVALID"
echo
