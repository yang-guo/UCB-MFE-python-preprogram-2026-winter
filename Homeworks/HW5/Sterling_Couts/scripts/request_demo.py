"""Record real HTTP evidence against a running service or container."""

import argparse
import json
import math
from pathlib import Path

import httpx

from measurements import measure_requests

ROOT = Path(__file__).resolve().parents[1]


def demonstrate(base_url: str, label: str):
    """Send a valid request and a rejected request, checking a saved HW4 result."""
    payload = json.loads((ROOT / "example_request.json").read_text())
    expected = json.loads((ROOT / "artifacts/example_reference.json").read_text())
    metadata = json.loads((ROOT / "artifacts/model_metadata.json").read_text())
    with httpx.Client(base_url=base_url, timeout=15, trust_env=False) as client:
        valid = client.post("/predict", json=payload)
        valid.raise_for_status()
        assert valid.status_code == 200
        assert math.isclose(
            valid.json()["prediction"],
            expected["expected_prediction"],
            rel_tol=0,
            abs_tol=1e-12,
        )
        assert valid.json()["model_version"] == metadata["model_version"]
        invalid_payload = {k: v for k, v in payload.items() if k != "rank_momentum_21"}
        invalid = client.post("/predict", json=invalid_payload)
        assert invalid.status_code == 422
        latency = measure_requests(
            client,
            payload,
            samples=100,
            warmups=5,
            method=f"Real localhost HTTP to {label}; sequential persistent httpx.Client",
        )
    transcript = (
        f"Target: {base_url}; {label}\n"
        f"POST /predict\nContent-Type: application/json\n{json.dumps(payload, indent=2)}\n"
        f"HTTP {valid.status_code}\n{valid.text}\n"
        f"Rejected request:\n{json.dumps(invalid_payload, indent=2)}\n"
        f"HTTP {invalid.status_code}\n{invalid.text}\n"
        "Prediction matches the pre-existing HW4 reference within absolute 1e-12.\n"
    )
    return transcript, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="local-uvicorn")
    args = parser.parse_args()
    transcript, latency = demonstrate(args.url, args.label)
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / f"evidence/{args.label}_requests.txt").write_text(transcript, encoding="utf-8")
    (ROOT / f"evidence/latency_{args.label}.json").write_text(
        json.dumps(latency, indent=2) + "\n", encoding="utf-8"
    )
    print(transcript)
    print(f"Median {latency['median_ms']:.3f} ms; p95 {latency['p95_ms']:.3f} ms")


if __name__ == "__main__":
    main()
