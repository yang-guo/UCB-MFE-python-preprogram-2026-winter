"""Build and exercise the real container; preserve a reviewable terminal transcript."""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE = "mfe-hw5:local"
NAME = "mfe-hw5-jing-jian-evidence"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    transcript = []

    def run(command, check=True):
        transcript.append("$ " + shlex.join(command))
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        transcript.append(result.stdout + result.stderr)
        transcript.append(f"[exit {result.returncode}]")
        if check and result.returncode:
            raise RuntimeError(transcript[-2])
        return result

    def wait_ready():
        for _ in range(120):
            probe = subprocess.run(
                ["curl", "--silent", "--fail", f"http://127.0.0.1:{args.port}/openapi.json"],
                capture_output=True,
                timeout=5,
            )
            if probe.returncode == 0:
                return
            time.sleep(0.5)
        raise RuntimeError("Container did not become ready within 60 seconds")

    def request(payload, expected):
        result = run(
            [
                "curl",
                "--silent",
                "--show-error",
                "-i",
                f"http://127.0.0.1:{args.port}/predict",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                payload,
            ]
        )
        if f" {expected} " not in result.stdout.splitlines()[0]:
            raise AssertionError(f"Expected HTTP {expected}: {result.stdout}")
        body = json.loads(result.stdout.split("\n\n", 1)[1])
        if expected == 200:
            reference = json.loads((ROOT / "artifacts/reference_prediction.json").read_text())
            if abs(body["prediction"] - reference["prediction"]) > 1e-10:
                raise AssertionError("Container prediction differs from recorded HW4 result")
            if body["model_version"] != "hw4-price-top20-ridge100-20160331-v1":
                raise AssertionError("Unexpected served model version")
        return body

    created = False
    try:
        run(["docker", "version"])
        run(["docker", "build", "-t", IMAGE, "."])
        run(
            [
                "docker",
                "image",
                "inspect",
                IMAGE,
                "--format",
                "{{.Id}} {{.Os}}/{{.Architecture}}",
            ]
        )
        run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                NAME,
                "-p",
                f"127.0.0.1:{args.port}:8000",
                IMAGE,
            ]
        )
        created = True
        wait_ready()
        request("@example_request.json", 200)
        request('{"momentum_21": "0.1"}', 422)
        run(["docker", "stop", NAME])
        run(["docker", "start", NAME])
        wait_ready()
        request("@example_request.json", 200)
        run(["docker", "logs", NAME])
        transcript.append(
            "PASS: valid 200, invalid 422, and identical prediction after restart."
        )
    finally:
        if created:
            run(["docker", "rm", "-f", NAME], check=False)
        (evidence / "container_requests.txt").write_text("\n".join(transcript) + "\n")
    print("Saved evidence/container_requests.txt; all container checks passed.")


if __name__ == "__main__":
    main()
