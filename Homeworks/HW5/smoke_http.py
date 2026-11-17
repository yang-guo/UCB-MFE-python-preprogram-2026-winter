"""Record valid, invalid, and post-restart requests over real localhost HTTP."""

import argparse
import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
IMAGE = "mfe-hw5:local"
NAME = "mfe-hw5-smoke"


@contextmanager
def server(image, port):
    local = image is None
    if local:
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
    else:
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            NAME,
            "-p",
            f"127.0.0.1:{port}:8000",
            image,
        ]
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with status {process.returncode}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=0.5) as response:
                    if json.load(response)["info"]["title"] == "HW4 Stock Return Model":
                        break
            except (OSError, URLError):
                time.sleep(0.2)
        else:
            raise TimeoutError("Server did not become ready")
        yield
    finally:
        if not local and process.poll() is None:
            print(f"$ docker stop {NAME}", flush=True)
            subprocess.run(["docker", "stop", NAME], check=True, cwd=ROOT)
        if local and process.poll() is None:
            print(f"$ stop Uvicorn process {process.pid}", flush=True)
            process.terminate()
        process.wait(timeout=15)


def post(port, payload, status):
    address = f"http://127.0.0.1:{port}/predict"
    encoded = json.dumps(payload)
    print(f"POST {address}\nContent-Type: application/json\n\n{encoded}", flush=True)
    request = Request(address, encoded.encode(), {"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        body = response.read().decode()
        print(f"HTTP {response.status} {response.reason}\n{body}\n", flush=True)
        if response.status != status:
            raise RuntimeError(f"Expected {status}, got {response.status}")
        return json.loads(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="use Uvicorn directly")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    payload = json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))
    image_id = None
    if not args.local:
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], text=True
        ).strip()
        print(f"Testing Docker image {IMAGE} ({image_id})", flush=True)
    else:
        print("Local Uvicorn smoke test over HTTP", flush=True)
    with server(image_id, args.port):
        before = post(args.port, payload, 200)
        post(args.port, {**payload, "ret21": "0.1"}, 422)
    with server(image_id, args.port):
        after = post(args.port, payload, 200)
    reference = json.loads((ROOT / "reference_prediction.json").read_text(encoding="utf-8"))
    if before["model_version"] != reference["model_version"] or not math.isclose(
        before["prediction"], reference["expected_prediction"], rel_tol=1e-9, abs_tol=1e-12
    ):
        raise RuntimeError("HTTP response did not match the original HW4 reference")
    if before != after:
        raise RuntimeError("Restart changed prediction or model version")
    print("PASS: same prediction and model version after restart.", flush=True)


if __name__ == "__main__":
    main()
