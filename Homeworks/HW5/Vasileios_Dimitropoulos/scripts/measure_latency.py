"""Measure /predict latency over real localhost HTTP (uvicorn or the Docker container).

Start the server first, then, from the submission folder:

    python scripts/measure_latency.py --url http://127.0.0.1:8000 --requests 200

Each request is timed with time.perf_counter around one blocking HTTP call on a
kept-alive connection, after warm-up requests. This includes JSON encoding, the
TCP round trip through localhost (and Docker's port forwarding, for the
container), validation and inference. It is a single-client sequential
measurement, not a throughput/load test.
"""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--label", default="localhost HTTP")
    args = parser.parse_args()
    if args.requests < 20:
        parser.error("measure at least 20 requests")

    body = json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for _ in range(args.warmup):
            client.post("/predict", json=body).raise_for_status()
        latencies = []
        for _ in range(args.requests):
            start = time.perf_counter()
            response = client.post("/predict", json=body)
            latencies.append(time.perf_counter() - start)
            response.raise_for_status()

    print(
        f"[latency] {args.label} ({args.url}), {args.requests} sequential requests after "
        f"{args.warmup} warm-up: median {statistics.median(latencies) * 1000:.2f} ms, "
        f"p95 {np.percentile(latencies, 95) * 1000:.2f} ms, "
        f"max {max(latencies) * 1000:.2f} ms "
        f"(client: Python {platform.python_version()}, {platform.system()} {platform.release()}, "
        f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs)"
    )


if __name__ == "__main__":
    main()
