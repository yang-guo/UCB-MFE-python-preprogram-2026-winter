"""Real HTTP latency against a running server (e.g. the Docker container).

Usage:  python latency_http.py [URL] [N]      default http://127.0.0.1:8000, 200 requests
One keep-alive httpx connection, 10 warm-up requests, then N timed with perf_counter.
"""

import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000") + "/predict"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
payload = json.loads((Path(__file__).parent / "example_request.json").read_text())

with httpx.Client(timeout=10) as client:
    for _ in range(10):
        client.post(url, json=payload).raise_for_status()
    timings = []
    for _ in range(n):
        start = time.perf_counter()
        client.post(url, json=payload).raise_for_status()
        timings.append((time.perf_counter() - start) * 1000)

timings.sort()
p95 = timings[math.ceil(0.95 * n) - 1]
print(
    f"real HTTP {url}, {n} requests after 10 warm-up, client {platform.machine()} "
    f"{platform.system()}: median {statistics.median(timings):.2f} ms, p95 {p95:.2f} ms, "
    f"max {timings[-1]:.2f} ms"
)
