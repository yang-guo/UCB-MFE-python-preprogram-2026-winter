"""Real localhost HTTP latency of a running service (e.g. the Docker container).

    python measure_latency.py                       # http://127.0.0.1:8000/predict
    python measure_latency.py --url URL -n 200

Uses one keep-alive httpx connection, sends example_request.json sequentially,
discards warm-up requests, and times each successful request with perf_counter.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    payload = json.loads((HERE / "example_request.json").read_text())
    latencies_ms = []
    with httpx.Client(timeout=10) as client:
        for i in range(args.warmup + args.n):
            start = time.perf_counter()
            response = client.post(args.url, json=payload)
            elapsed = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            if i >= args.warmup:
                latencies_ms.append(elapsed)

    print(
        f"real HTTP {args.url}: {args.n} sequential requests after {args.warmup} warm-up, "
        f"median {statistics.median(latencies_ms):.2f} ms, "
        f"p95 {float(np.percentile(latencies_ms, 95)):.2f} ms, "
        f"max {max(latencies_ms):.2f} ms | client: {platform.processor()}, "
        f"{platform.system()} {platform.release()}"
    )


if __name__ == "__main__":
    main()
