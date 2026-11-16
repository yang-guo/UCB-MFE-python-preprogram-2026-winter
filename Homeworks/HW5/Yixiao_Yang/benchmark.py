"""Measure the real saved pipeline through in-process FastAPI TestClient."""

import argparse
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app import app

ROOT = Path(__file__).resolve().parent


def hardware_details():
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    details = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu,
        "logical_cpus_visible": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        details["logical_cpus_in_affinity"] = len(os.sched_getaffinity(0))
    return details


def measure_requests(client, payload, *, warmups=10, requests=100):
    if requests < 20 or warmups < 1:
        raise ValueError("Use at least 20 measured requests and one warm-up request")

    model_version = None

    def checked_request():
        response = client.post("/predict", json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Benchmark request failed: {response.status_code} {response.text}"
            )
        body = response.json()
        if not math.isfinite(body["prediction"]):
            raise RuntimeError("Benchmark returned a non-finite prediction")
        return body["model_version"]

    for _ in range(warmups):
        model_version = checked_request()

    samples_ms = []
    for _ in range(requests):
        started = time.perf_counter()
        version = checked_request()
        samples_ms.append((time.perf_counter() - started) * 1000)
        if version != model_version:
            raise RuntimeError("Model version changed during measurement")

    ordered = sorted(samples_ms)
    return {
        "measurement": "in-process FastAPI TestClient; no TCP or Docker",
        "timing": "time.perf_counter around synchronous post plus response validation",
        "warmup_requests": warmups,
        "successful_requests": requests,
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[math.ceil(0.95 * requests) - 1],
        "p95_method": "nearest rank",
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "model_version": model_version,
        "hardware": hardware_details(),
        "budget": "No hardware-independent latency cutoff asserted",
        "samples_ms": samples_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "latency.json")
    args = parser.parse_args()
    payload = json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))
    with TestClient(app) as client:
        report = measure_requests(client, payload, warmups=args.warmups, requests=args.requests)
    report["measured_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Method: {report['measurement']}")
    print(
        f"Warm-ups: {report['warmup_requests']}; measured successes: {report['successful_requests']}"
    )
    print(
        f"Median: {report['median_ms']:.3f} ms; p95: {report['p95_ms']:.3f} ms (nearest rank)"
    )
    print(json.dumps(report["hardware"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
