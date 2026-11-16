"""Start a real localhost server, record requests, then stop only that process."""

import json
import subprocess
import sys

from scripts.request_demo import ROOT, demonstrate
from scripts.verify_container import wait_ready


def main():
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    # CREATE_NO_WINDOW avoids opening an unnecessary terminal on Windows.
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with (evidence / "local_server.txt").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8874",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        try:
            wait_ready("http://127.0.0.1:8874")
            if server.poll() is not None:
                raise RuntimeError("Our server exited; another process may own port 8874")
            transcript, latency = demonstrate("http://127.0.0.1:8874", "local-uvicorn")
            (evidence / "localhost_requests.txt").write_text(transcript, encoding="utf-8")
            (evidence / "latency_localhost.json").write_text(
                json.dumps(latency, indent=2) + "\n", encoding="utf-8"
            )
            print(transcript)
            print(f"Median {latency['median_ms']:.3f} ms; p95 {latency['p95_ms']:.3f} ms")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
