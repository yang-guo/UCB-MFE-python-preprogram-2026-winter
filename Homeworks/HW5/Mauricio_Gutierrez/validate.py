"""Run required local checks and save their exact output."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    temp = ROOT / ".setup-temp"
    temp.mkdir(exist_ok=True)
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), PYTHONIOENCODING="utf-8")
    commands = [
        ["-m", "pip", "check"],
        ["-m", "ruff", "check", "."],
        ["-m", "ruff", "format", "--check", "."],
        ["-m", "pytest", "-q"],
    ]
    passed = True
    with (evidence / "checks.txt").open("w", encoding="utf-8") as report:
        for args in commands:
            command = [sys.executable, *args]
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            block = f"$ python {' '.join(args)}\n{result.stdout}\nExit code: {result.returncode}\n\n"
            report.write(block)
            report.flush()
            print(block, flush=True)
            passed = passed and result.returncode == 0
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
