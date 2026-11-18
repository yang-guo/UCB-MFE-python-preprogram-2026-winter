"""Measure /predict latency over real localhost HTTP (e.g. against the running container).

Standard library only, one sequential client, a new connection per request.

Usage:  python measure_latency.py [--url http://127.0.0.1:8000/predict] [-n 200]
"""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def post(url: str, body: bytes) -> int:
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()
        return response.status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("-n", type=int, default=200, help="timed requests (>= 20)")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    body = json.dumps(json.loads((ROOT / "example_request.json").read_text())).encode()
    for _ in range(args.warmup):
        assert post(args.url, body) == 200

    latencies_ms = []
    for _ in range(args.n):
        start = time.perf_counter()
        status = post(args.url, body)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert status == 200

    p95 = statistics.quantiles(latencies_ms, n=100)[94]
    print(
        f"Real HTTP to {args.url}: {args.n} sequential requests after {args.warmup} warm-up: "
        f"median {statistics.median(latencies_ms):.2f} ms, p95 {p95:.2f} ms, "
        f"max {max(latencies_ms):.2f} ms"
    )


if __name__ == "__main__":
    main()
