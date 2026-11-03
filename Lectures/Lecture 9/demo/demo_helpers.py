"""Small classroom helpers; no business logic is hidden here."""

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent


def run_fee_test(version):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "tests/test_fees.py::test_known_fee",
            "--fee-version",
            version,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    expected = 1 if version == "buggy" else 0
    if result.returncode != expected:
        raise RuntimeError(result.stderr or "Unexpected test result")


def start_server(port=8769):
    url = f"http://127.0.0.1:{port}"
    try:
        httpx.get(url + "/health", timeout=0.5, trust_env=False)
    except httpx.TransportError:
        pass
    else:
        raise RuntimeError(
            f"Port {port} is already in use. Stop the previous demo or choose another port."
        )
    log = open(ROOT / "server.log", "w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=log,
        stderr=log,
    )
    log.close()
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Server exited; inspect demo/server.log")
        try:
            response = httpx.get(url + "/health", timeout=0.5, trust_env=False)
            if response.status_code == 200:
                print("Ready:", url, response.json())
                return process, url
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError("Server startup timed out; use the saved demo reader")


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    print("Demo server stopped.")
