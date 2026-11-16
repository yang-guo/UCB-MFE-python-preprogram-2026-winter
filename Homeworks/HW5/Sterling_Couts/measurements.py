"""Sequential latency measurement; no universal latency pass threshold."""

import os
import platform
import time
from datetime import datetime, timezone

import numpy as np


def measure_requests(client, payload: dict, *, samples=100, warmups=5, method: str):
    """Measure complete successful POST calls, excluding warmup/startup time."""
    if samples < 20 or warmups < 1:
        raise ValueError("Use at least 20 measured requests and one warmup")
    for _ in range(warmups):
        response = client.post("/predict", json=payload)
        response.raise_for_status()
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter()
        response = client.post("/predict", json=payload)
        elapsed.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        if not np.isfinite(response.json()["prediction"]):
            raise ValueError("Non-finite prediction in latency measurement")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "samples": samples,
        "warmup_requests": warmups,
        "concurrency": 1,
        "clock": "time.perf_counter",
        "units": "milliseconds",
        "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.percentile(elapsed, 95, method="linear")),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "raw_ms": elapsed,
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "scope": "Sequential warmed requests; includes client and validation overhead. No load-test claim.",
    }
