"""Run each attempt in its own Python process so its deadline is enforceable."""

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_worker(run, ticker, attempt, timeout):
    stem = Path(run) / f"{ticker}-attempt-{attempt}"
    environment = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1"}
    argv = [sys.executable, "-m", "research_harness.worker", str(run), ticker, str(attempt)]
    with stem.with_suffix(".log").open("wb") as output:
        process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        detail = stem.with_suffix(".log").read_text(errors="replace")[-1500:]
        error_type = ValueError if code == 3 else RuntimeError
        raise error_type(f"Research worker exited {code}: {detail.strip()}")


def main():
    from .agent import ResearchDeps, run_research

    run, ticker, attempt = sys.argv[1:]
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    try:
        asyncio.run(run_research(ResearchDeps(run, ticker, int(attempt)), config))
    except ValueError as error:
        print(f"ValueError: {error}", file=sys.stderr)
        return 3  # Permanent input error: a new model attempt won't repair it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
