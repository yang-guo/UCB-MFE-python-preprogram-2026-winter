"""Build the real Docker image and record valid, invalid, and restart requests.

Run from any directory with Python 3.12: ``python verify_container.py``.
Docker Desktop (Windows/macOS) or a running Docker Engine must be available.
This script never substitutes a local Python server for the Docker container.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE = "mfe-hw5:local"


class VerificationError(RuntimeError):
    """A recorded Docker command or HTTP check did not pass."""


class Transcript:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, message):
        text = str(message).rstrip()
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def command(self, arguments, timeout=60, check=True):
        self.write("\nCOMMAND JSON: " + json.dumps(arguments))
        try:
            result = subprocess.run(
                arguments,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            self.write(output)
            raise VerificationError(f"Command exceeded its {timeout}-second timeout.") from exc
        self.write(result.stdout)
        self.write(f"EXIT CODE: {result.returncode}")
        if check and result.returncode:
            raise VerificationError(f"Command failed with exit code {result.returncode}.")
        return result

    def close(self):
        self.handle.close()


def http_call(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if data is None else "POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, dict(response.headers), response.read().decode("utf-8")


def record_http(transcript, method, url, payload, response):
    transcript.write(f"\n{method} {url}")
    if payload is not None:
        transcript.write("REQUEST JSON: " + json.dumps(payload, allow_nan=False))
    status, headers, body = response
    transcript.write(f"HTTP STATUS: {status}")
    transcript.write("RESPONSE HEADERS: " + json.dumps(headers))
    transcript.write("RESPONSE BODY: " + body)


def wait_until_ready(transcript, base_url, timeout):
    deadline = time.perf_counter() + timeout
    last_error = "No HTTP response received."
    url = base_url + "/health"
    while time.perf_counter() < deadline:
        try:
            response = http_call(url, timeout=min(2, max(0.1, deadline - time.perf_counter())))
            if response[0] == 200:
                record_http(transcript, "GET", url, None, response)
                return
            last_error = f"HTTP {response[0]}: {response[2]}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise VerificationError(
        f"Container was not ready within {timeout}s. Last result: {last_error}"
    )


def check_prediction(transcript, base_url, payload, reference):
    response = http_call(base_url + "/predict", payload)
    record_http(transcript, "POST", base_url + "/predict", payload, response)
    if response[0] != 200:
        raise VerificationError(f"Valid input returned {response[0]}, expected 200.")
    result = json.loads(response[2])
    prediction = result.get("prediction")
    if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
        raise VerificationError("Prediction is not a numeric JSON value.")
    if not math.isfinite(prediction):
        raise VerificationError("Prediction is not finite.")
    if result.get("model_version") != reference["model_version"]:
        raise VerificationError("Response model_version differs from the HW4 reference.")
    if not math.isclose(
        prediction,
        reference["prediction"],
        abs_tol=reference["absolute_tolerance"],
        rel_tol=reference["relative_tolerance"],
    ):
        raise VerificationError("Prediction differs from the recorded HW4 reference.")
    transcript.write(
        "CHECK PASSED: finite prediction and model version match the HW4 reference."
    )
    return prediction


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def update_readme_status(passed):
    """Keep the visible status consistent with the actual transcript from this run."""
    path = ROOT / "README.md"
    start = "<!-- container-status:start -->"
    end = "<!-- container-status:end -->"
    original = path.read_text(encoding="utf-8")
    if start not in original or end not in original:
        return
    if passed:
        status = (
            "**Docker verification: passed.** The actual image build, valid request "
            "(200), invalid request (422), and prediction after restarting the same "
            "image passed. See `evidence/container_transcript.txt` for the commands, "
            "image ID, responses and execution time."
        )
    else:
        status = (
            "**Docker verification: incomplete.** The last actual verification run "
            "failed. Inspect `evidence/container_transcript.txt`, fix the reported "
            "problem and rerun `python verify_container.py` before submission."
        )
    before, remaining = original.split(start, 1)
    _, after = remaining.split(end, 1)
    path.write_text(before + start + "\n" + status + "\n" + end + after, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=port_number, default=8000, help="Local host port (8000)."
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Startup timeout in seconds (60)."
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    # Check before creating a transcript: --help or a missing executable is not runtime evidence.
    if shutil.which("docker") is None:
        print(
            "Docker was not found. Install/start Docker Desktop or Docker Engine, then rerun "
            "this command. No container transcript was created or changed.",
            file=sys.stderr,
        )
        return 1
    try:
        payload = json.loads((ROOT / "example_request.json").read_text(encoding="utf-8"))
        reference = json.loads((ROOT / "reference_prediction.json").read_text(encoding="utf-8"))
        if payload != reference["input"]:
            raise ValueError("example_request.json must match the recorded reference input.")
        for key in ("prediction", "absolute_tolerance", "relative_tolerance"):
            if isinstance(reference[key], bool) or not math.isfinite(reference[key]):
                raise ValueError(f"Reference {key} must be a finite number.")
        if reference["absolute_tolerance"] < 0 or reference["relative_tolerance"] < 0:
            raise ValueError("Reference tolerances must be non-negative.")
        if not isinstance(reference["model_version"], str):
            raise ValueError("Reference model_version must be a string.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot read the saved HW4 example/reference: {exc}", file=sys.stderr)
        return 1

    path = ROOT / "evidence" / "container_transcript.txt"
    transcript = Transcript(path)
    name = "mfe-hw5-verify-" + uuid.uuid4().hex[:12]
    base_url = f"http://127.0.0.1:{args.port}"
    container_attempted = False
    passed = False
    try:
        transcript.write("HW5 Docker verification: actual command and HTTP output")
        transcript.write("UTC start: " + datetime.now(timezone.utc).isoformat())
        transcript.write(f"Working directory: {ROOT}")
        transcript.write(f"Owned temporary container: {name}")
        transcript.command(["docker", "version"])
        transcript.command(["docker", "build", "-t", IMAGE, "."], timeout=900)
        image_id = transcript.command(
            ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]
        ).stdout.strip()
        container_attempted = True
        transcript.command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-p",
                f"127.0.0.1:{args.port}:8000",
                IMAGE,
            ]
        )
        wait_until_ready(transcript, base_url, args.timeout)
        first_prediction = check_prediction(transcript, base_url, payload, reference)

        invalid_payload = dict(payload, unexpected_field=123)
        invalid_response = http_call(base_url + "/predict", invalid_payload)
        record_http(
            transcript, "POST", base_url + "/predict", invalid_payload, invalid_response
        )
        if invalid_response[0] != 422:
            raise VerificationError("An extra input field did not return 422.")
        transcript.write("CHECK PASSED: invalid input is rejected with HTTP 422.")

        transcript.command(["docker", "stop", name])
        transcript.command(["docker", "start", name])
        wait_until_ready(transcript, base_url, args.timeout)
        restarted_image = transcript.command(
            ["docker", "inspect", "--format", "{{.Image}}", name]
        ).stdout.strip()
        if restarted_image != image_id:
            raise VerificationError("Container image ID changed after restart.")
        restarted_prediction = check_prediction(transcript, base_url, payload, reference)
        if not math.isclose(
            restarted_prediction,
            first_prediction,
            abs_tol=reference["absolute_tolerance"],
            rel_tol=reference["relative_tolerance"],
        ):
            raise VerificationError("Prediction changed after restarting the same image.")
        transcript.write(
            "CHECK PASSED: the same image still serves the same model after restart."
        )
        passed = True
    except (VerificationError, OSError, ValueError, TypeError, KeyError) as exc:
        transcript.write(f"\nVERIFICATION FAILED: {exc}")
        if container_attempted:
            try:
                transcript.command(["docker", "logs", name], check=False)
            except (VerificationError, OSError) as log_error:
                transcript.write(f"Could not retrieve container logs: {log_error}")
    finally:
        if container_attempted:
            try:
                cleanup = transcript.command(["docker", "rm", "-f", name], check=False)
                if cleanup.returncode:
                    transcript.write(
                        f"Cleanup could not remove {name}; inspect the output above."
                    )
            except (VerificationError, OSError) as exc:
                transcript.write(f"Cleanup error for {name}: {exc}")
        transcript.write("\nFINAL RESULT: " + ("PASS" if passed else "FAIL"))
        transcript.write("UTC finish: " + datetime.now(timezone.utc).isoformat())
        transcript.close()
    print(f"Actual execution transcript: {path}")
    try:
        update_readme_status(passed)
    except OSError as exc:
        print(
            f"Transcript saved, but README status could not be updated: {exc}", file=sys.stderr
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
