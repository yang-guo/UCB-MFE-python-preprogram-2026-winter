"""Real-HTTP latency of a running service (local uvicorn or the container).

Usage: python scripts/http_latency.py [base_url] [n_requests]
Default: http://127.0.0.1:8000, 200 requests after 20 warm-up requests,
one keep-alive connection, sequential requests, time.perf_counter.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main(base_url="http://127.0.0.1:8000", n=200, warmup=20):
    payload = json.loads((ROOT / "example_request.json").read_text())
    with httpx.Client(base_url=base_url, timeout=5.0, trust_env=False) as client:
        for _ in range(warmup):
            client.post("/predict", json=payload).raise_for_status()
        latencies = []
        for _ in range(int(n)):
            start = time.perf_counter()
            response = client.post("/predict", json=payload)
            latencies.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
    p95 = statistics.quantiles(latencies, n=100, method="inclusive")[94]
    print(
        f"Real HTTP {base_url}/predict: {n} sequential requests after {warmup} warm-up, "
        f"keep-alive httpx client, {platform.machine()} {platform.system()} "
        f"Python {platform.python_version()}\n"
        f"median {statistics.median(latencies):.2f} ms, p95 {p95:.2f} ms, "
        f"max {max(latencies):.2f} ms"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
