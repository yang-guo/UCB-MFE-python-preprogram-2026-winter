"""Real localhost HTTP latency against the running container (not in-process).

    docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
    .venv\\Scripts\\python measure_http_latency.py      (in another terminal)

Sequential requests over one keep-alive connection, so this includes uvicorn, the
loopback network and Docker Desktop's port forwarding (WSL 2 on my machine).
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/predict"


def main(warmup: int = 20, n: int = 200):
    body = json.loads((HERE / "example_request.json").read_text())
    with httpx.Client(timeout=5) as http:
        for _ in range(warmup):
            http.post(URL, json=body).raise_for_status()
        times_ms = []
        for _ in range(n):
            start = time.perf_counter()
            response = http.post(URL, json=body)
            times_ms.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
    times_ms.sort()
    report = {
        "method": f"real HTTP to Docker container at {URL}, httpx keep-alive, sequential, "
        f"{warmup} warm-up + {n} timed",
        "n": n,
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(times_ms[int(0.95 * n) - 1], 3),
        "max_ms": round(times_ms[-1], 3),
        "client_machine": f"{platform.processor()} | {platform.system()} {platform.release()}",
    }
    (HERE / "evidence").mkdir(exist_ok=True)
    (HERE / "evidence" / "latency_http_container.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
