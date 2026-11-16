"""Build, test and restart the same Docker image; save actual terminal evidence."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from scripts.request_demo import demonstrate

ROOT = Path(__file__).resolve().parents[1]


def run(docker, *args):
    # A freshly installed Windows CLI may be supplied by absolute path before
    # this terminal sees the new PATH. Its credential helper lives beside it.
    environment = os.environ.copy()
    if Path(docker).is_absolute():
        environment["PATH"] = (
            str(Path(docker).parent) + os.pathsep + environment.get("PATH", "")
        )
    result = subprocess.run(
        [docker, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        timeout=900,
    )
    if result.returncode:
        raise RuntimeError(f"Docker command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def wait_ready(url):
    deadline = time.monotonic() + 90
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(url + "/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    raise TimeoutError("Container did not become ready within 90 seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--port", type=int, default=8875)
    args = parser.parse_args()
    docker, image, name = args.docker, "mfe-hw5:local", "mfe-hw5-sterling-check"
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    # Never replace a user's pre-existing container with the same name.
    existing = run(docker, "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}")
    if existing.strip():
        raise RuntimeError(f"Container {name} already exists; inspect it before continuing")
    print("Building the homework image...", flush=True)
    build = run(docker, "build", "-t", image, ".")
    (evidence / "docker_build.txt").write_text(build, encoding="utf-8")
    image_id = run(docker, "image", "inspect", image, "--format", "{{.Id}}").strip()
    url = f"http://127.0.0.1:{args.port}"
    created = False
    transcript = [
        f"Image: {image}\nImage ID: {image_id}\n",
        "$ docker build -t mfe-hw5:local .\n",
    ]
    try:
        transcript.append(
            f"$ docker run -d --name {name} -p 127.0.0.1:{args.port}:8000 {image_id}\n"
        )
        run(docker, "run", "-d", "--name", name, "-p", f"127.0.0.1:{args.port}:8000", image_id)
        created = True
        wait_ready(url)
        print("Container ready; checking valid/invalid requests...", flush=True)
        text, latency = demonstrate(url, "docker-before-restart")
        transcript.append(text)
        (evidence / "latency_docker.json").write_text(json.dumps(latency, indent=2) + "\n")
        transcript.append(f"$ docker stop {name}\n" + run(docker, "stop", name))
        transcript.append(f"$ docker start {name}\n" + run(docker, "start", name))
        wait_ready(url)
        text, _ = demonstrate(url, "docker-after-restart")
        transcript.append(text)
        after_image = run(docker, "inspect", name, "--format", "{{.Image}}").strip()
        assert image_id == after_image
        model_hash = run(
            docker,
            "exec",
            name,
            "python",
            "-c",
            "import hashlib; print(hashlib.sha256(open('/app/artifacts/fixed_anchor_2012.joblib','rb').read()).hexdigest())",
        ).strip()
        metadata = json.loads((ROOT / "artifacts/model_metadata.json").read_text())
        assert model_hash == metadata["artifact_sha256"]
        image_files = json.loads(
            run(
                docker,
                "exec",
                name,
                "python",
                "-c",
                "import json; from pathlib import Path; "
                "print(json.dumps(sorted(str(p.relative_to('/app')) "
                "for p in Path('/app').rglob('*') if p.is_file())))",
            )
        )
        assert set(image_files) == {
            "app.py",
            "schemas.py",
            "serving.py",
            "requirements.txt",
            "hw4/__init__.py",
            "hw4/preprocessing.py",
            "artifacts/fixed_anchor_2012.joblib",
            "artifacts/model_metadata.json",
        }
        verification = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "image": image,
            "image_id": image_id,
            "image_after_restart": after_image,
            "artifact_sha256_inside_container": model_hash,
            "valid_status": 200,
            "invalid_status": 422,
            "after_restart_status": 200,
            "known_result_absolute_tolerance": 1e-12,
            "host_port": args.port,
            "application_files": image_files,
        }
        (evidence / "container_verification.json").write_text(
            json.dumps(verification, indent=2) + "\n", encoding="utf-8"
        )
        (evidence / "container_requests.txt").write_text(
            "\n".join(transcript), encoding="utf-8"
        )
        print(
            "Real container requests, same-image restart and artifact hash verified.",
            flush=True,
        )
    finally:
        if created:
            logs = run(docker, "logs", name)
            (evidence / "container_logs.txt").write_text(logs, encoding="utf-8")
            run(docker, "rm", "-f", name)


if __name__ == "__main__":
    main()
