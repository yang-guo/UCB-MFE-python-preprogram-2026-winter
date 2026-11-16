"""Real HTTP evidence. Use --local to start a temporary local Uvicorn server."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent


def wait_ready(base_url):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(base_url + "/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("The service did not become ready within 60 seconds")


def check_requests(base_url, label):
    example = json.loads((ROOT / "example_request.json").read_text())
    reference = json.loads((ROOT / "artifacts/reference_prediction.json").read_text())
    invalid = {**example, "volatility_63": -0.1}
    lines = [label, f"Base URL: {base_url}"]
    with httpx.Client(base_url=base_url, timeout=10) as client:
        for name, payload, status in [("valid", example, 200), ("invalid", invalid, 422)]:
            response = client.post("/predict", json=payload)
            lines.extend(
                [
                    f"POST /predict ({name})",
                    json.dumps(payload),
                    f"HTTP {response.status_code}",
                    response.text,
                ]
            )
            assert response.status_code == status, response.text
            if status == 200:
                result = response.json()
                assert result["model_version"] == reference["model_version"]
                assert math.isclose(
                    result["prediction"],
                    reference["prediction"],
                    rel_tol=reference["relative_tolerance"],
                    abs_tol=reference["absolute_tolerance"],
                )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:18765")
    args = parser.parse_args()
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    if not args.local:
        wait_ready(args.base_url)
        print(check_requests(args.base_url, "User-selected HTTP server"))
        return
    port = int(args.base_url.rsplit(":", 1)[1])
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with (evidence / "localhost_server.txt").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(args.base_url)
            if process.poll() is not None:
                raise RuntimeError("Our server exited; another process may own the port")
            transcript = check_requests(args.base_url, "Real localhost HTTP; not Docker")
            (evidence / "localhost_http.txt").write_text(transcript, encoding="utf-8")
            print(transcript)
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
