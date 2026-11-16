"""Measure /predict latency over real HTTP (a local uvicorn server or the container).

Start the server first (`python -m uvicorn app:app --port 8000` or `make run`), then:

    python measure_latency.py [base_url] [n_requests]
"""

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WARMUP = 10

payload = json.loads((Path(__file__).parent / "example_request.json").read_text())

with httpx.Client(base_url=URL, timeout=5.0) as client:  # keep-alive connection
    for _ in range(WARMUP):
        client.post("/predict", json=payload).raise_for_status()
    latencies_ms = []
    for _ in range(N):
        start = time.perf_counter()
        response = client.post("/predict", json=payload)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        response.raise_for_status()

p95 = statistics.quantiles(latencies_ms, n=100, method="inclusive")[94]
print(
    f"real HTTP to {URL}: {N} sequential requests after {WARMUP} warm-ups, one keep-alive "
    f"connection: median {statistics.median(latencies_ms):.2f} ms, p95 {p95:.2f} ms"
)
