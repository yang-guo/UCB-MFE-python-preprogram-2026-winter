"""Real localhost HTTP evidence, including restart of the same Docker image."""

import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "mfe-hw5-david-check"
IMAGE = "mfe-hw5:local"
URL = "http://127.0.0.1:18005/predict"


def docker(*args):
    result = subprocess.check_output(["docker", *args], text=True).strip()
    print("$ docker", " ".join(args), "\n", result, flush=True)
    return result


def request(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        text = response.read().decode()
        status = response.code
    print("POST", URL, json.dumps(payload), "\nHTTP", status, text, flush=True)
    return status, json.loads(text)


def ready():
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                URL.replace("/predict", "/openapi.json"), timeout=1
            ) as r:
                if r.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise RuntimeError("Container did not become ready")


def main():
    reference = json.loads((ROOT / "reference_prediction.json").read_text())
    docker("run", "-d", "--name", NAME, "-p", "127.0.0.1:18005:8000", IMAGE)
    try:
        image_id = docker("inspect", "--format", "{{.Image}}", NAME)
        ready()
        status, result = request(reference["input"])
        assert status == 200
        assert result["model_version"] == reference["expected"]["model_version"]
        assert math.isclose(
            result["prediction"],
            reference["expected"]["prediction"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        assert request(dict(reference["input"], MKT="invalid"))[0] == 422
        status, error = request(dict(reference["input"], symbol="ZZZZZZZ"))
        assert status == 404 and error["detail"]["code"] == "SYMBOL_OUTSIDE_UNIVERSE"
        membership = json.loads((ROOT / "artifacts/membership.json").read_text())
        unavailable = next(
            r["symbol"] for r in membership["members"] if not r["model_available"]
        )
        status, error = request(dict(reference["input"], symbol=unavailable))
        assert status == 404 and error["detail"]["code"] == "MODEL_UNAVAILABLE"
        other = next(s for s in sorted(reference["expected_predictions"]) if s != "AAPL")
        status, second = request(dict(reference["input"], symbol=other))
        assert status == 200 and second["symbol"] == other
        assert math.isclose(
            second["prediction"],
            reference["expected_predictions"][other],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        docker("stop", NAME)
        docker("start", NAME)
        assert docker("inspect", "--format", "{{.Image}}", NAME) == image_id
        ready()
        status, restarted = request(reference["input"])
        assert status == 200
        assert math.isclose(restarted["prediction"], result["prediction"], rel_tol=1e-12)
        print(
            "PASS: real container multiple stocks, invalid input, outside-universe/unavailable-model codes, and same-image stop/start.",
            flush=True,
        )
    finally:
        docker("logs", NAME)
        docker("rm", "-f", NAME)


if __name__ == "__main__":
    main()
