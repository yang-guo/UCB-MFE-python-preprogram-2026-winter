"""Measure /predict latency over real HTTP against a running server or container.

python measure_latency.py [base_url]     # default http://127.0.0.1:8000
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx

REQUESTS, WARMUP = 200, 10


def main(base_url: str = "http://127.0.0.1:8000") -> None:
    payload = json.loads((Path(__file__).parent / "example_request.json").read_text())
    latencies_ms = []
    with httpx.Client(base_url=base_url) as client:  # one keep-alive connection
        for i in range(WARMUP + REQUESTS):
            start = time.perf_counter()
            response = client.post("/predict", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            if i >= WARMUP:
                latencies_ms.append(elapsed_ms)
    print(
        f"{REQUESTS} sequential HTTP requests to {base_url} from "
        f"{platform.machine()} {platform.system()}: "
        f"median {statistics.median(latencies_ms):.2f} ms, "
        f"p95 {statistics.quantiles(latencies_ms, n=20)[-1]:.2f} ms, "
        f"max {max(latencies_ms):.2f} ms"
    )


if __name__ == "__main__":
    main(*sys.argv[1:2])
