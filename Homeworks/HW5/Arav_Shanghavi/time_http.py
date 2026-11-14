"""Latency over real HTTP, for a server that is already running (local uvicorn or
the container). The pytest timing uses an in-process TestClient, so this one adds
the network stack and uvicorn itself.

    python time_http.py                      # http://127.0.0.1:8000
    python time_http.py http://127.0.0.1:8001
"""

import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
body = json.loads((Path(__file__).parent / "example_request.json").read_text())

with httpx.Client(base_url=base, timeout=10) as http:
    for _ in range(15):
        http.post("/predict", json=body).raise_for_status()
    ms = []
    for _ in range(40):
        t0 = time.perf_counter()
        http.post("/predict", json=body).raise_for_status()
        ms.append((time.perf_counter() - t0) * 1000)

ordered = sorted(ms)
p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
print(
    f"real HTTP to {base}, 15 warm-up + 40 timed, one keep-alive connection: "
    f"median {statistics.median(ms):.2f} ms, p95 {p95:.2f} ms"
)
