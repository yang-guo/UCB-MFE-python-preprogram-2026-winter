"""Build and verify Docker valid/invalid/restart behavior; save real evidence."""

import argparse
import contextlib
import json
import math
import shlex
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def command(*args):
    print("$ " + shlex.join(args), flush=True)
    result = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="")
    result.check_returncode()
    return result.stdout.strip()


def wait_ready(url):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "/openapi.json", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(0.25)
    raise TimeoutError("API did not become ready in 60 seconds")


def check_request(url, body, expected_status):
    print("POST " + url + "/predict")
    print("Content-Type: application/json")
    print(json.dumps(body))
    request = Request(
        url + "/predict",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        content = response.read().decode()
        print(f"HTTP {response.status}")
        print(str(response.headers).rstrip())
        print(content)
        assert response.status == expected_status, content
    result = json.loads(content)
    if expected_status == 200:
        metadata = json.loads((HERE / "artifacts/metadata.json").read_text())
        assert result["model_version"] == metadata["model_version"]
        assert math.isfinite(result["prediction"])
        assert math.isclose(
            result["prediction"],
            metadata["reference"]["prediction"],
            rel_tol=1e-9,
            abs_tol=1e-10,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image", default="mfe-hw5:local")
    args = parser.parse_args()
    # Fail before creating an evidence file if Docker is absent or not started.
    subprocess.run(["docker", "info"], check=True, capture_output=True)
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    with (evidence / "container.txt").open("w") as transcript:
        with contextlib.redirect_stdout(transcript):
            command("docker", "version")
            command("docker", "build", "-t", args.image, ".")
            command("docker", "image", "inspect", args.image, "--format", "{{.Id}}")
            container = command(
                "docker", "run", "-d", "-p", f"127.0.0.1:{args.port}:8000", args.image
            )
            try:
                url = f"http://127.0.0.1:{args.port}"
                wait_ready(url)
                valid = json.loads((HERE / "example_request.json").read_text())
                before = check_request(url, valid, 200)
                check_request(url, {"features": [0.1]}, 422)
                command("docker", "stop", container)
                command("docker", "start", container)
                wait_ready(url)
                after = check_request(url, valid, 200)
                assert before == after
                print("PASS: valid, invalid, and identical prediction after restart.")
            finally:
                command("docker", "logs", container)
                command("docker", "rm", "-f", container)
    print(f"Verified container. Evidence: {evidence / 'container.txt'}")


if __name__ == "__main__":
    main()
