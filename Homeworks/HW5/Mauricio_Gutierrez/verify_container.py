"""Build the image; record real HTTP requests before and after container restart."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

from verify_http import check_requests, wait_ready

ROOT = Path(__file__).resolve().parent


def main():
    if shutil.which("docker") is None:
        raise SystemExit("Docker is unavailable. Install/start Docker before this check.")
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    name = "mfe-hw5-check-" + uuid.uuid4().hex[:8]
    image = "mfe-hw5:local"
    container = None
    with (evidence / "container.txt").open("w", encoding="utf-8") as log:

        def run(*args):
            log.write("$ " + " ".join(args) + "\n")
            log.flush()
            result = subprocess.run(
                args,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.write(result.stdout + "\n")
            log.flush()
            print(result.stdout, flush=True)
            result.check_returncode()
            return result.stdout.strip()

        try:
            run("docker", "version")
            run("docker", "build", "-t", image, ".")
            container = run(
                "docker", "run", "-d", "--name", name, "-p", "127.0.0.1:18766:8000", image
            ).splitlines()[-1]
            image_id = run("docker", "inspect", "--format", "{{.Image}}", container)
            url = "http://127.0.0.1:18766"
            wait_ready(url)
            log.write(check_requests(url, "Running container: initial requests"))
            log.flush()
            run("docker", "stop", container)
            run("docker", "start", container)
            wait_ready(url)
            assert run("docker", "inspect", "--format", "{{.Image}}", container) == image_id
            log.write(check_requests(url, "Same image: requests after stop/start"))
            log.flush()
            (evidence / "container_status.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "image": image,
                        "image_id": image_id,
                        "checks": ["valid 200", "invalid 422", "same image after stop/start"],
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            if container:
                run("docker", "logs", container)
                run("docker", "rm", "-f", container)


if __name__ == "__main__":
    main()
