"""
Latency over real localhost HTTP (unlike the in-process TestClient test).

Start the container (or `uvicorn app:app --port 8000`) first, then run
``python measure_latency.py``. One keep-alive connection, 5 warm-up requests,
then 200 timed requests with time.perf_counter.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import httpx
import numpy as np

URL = "http://127.0.0.1:8000/predict"
N_WARMUP, N_MEASURED = 5, 200

payload = json.loads((Path(__file__).resolve().parent / "example_request.json").read_text())

with httpx.Client() as client:
    for _ in range(N_WARMUP):
        client.post(URL, json=payload).raise_for_status()
    latencies_ms = []
    for _ in range(N_MEASURED):
        start = time.perf_counter()
        client.post(URL, json=payload).raise_for_status()
        latencies_ms.append((time.perf_counter() - start) * 1000)

print(
    f"Real HTTP to {URL}, {N_MEASURED} requests after {N_WARMUP} warm-up, client on "
    f"{platform.machine()} {platform.system()}: median {statistics.median(latencies_ms):.2f} ms, "
    f"p95 {np.percentile(latencies_ms, 95):.2f} ms"
)
