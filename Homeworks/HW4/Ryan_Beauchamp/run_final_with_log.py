"""Run final_run.py once, showing its output live AND saving it to outputs/final_run.log (a portable "tee").

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/run_final_with_log.py

It only launches `final_run.py --i-confirm-frozen` unbuffered (-u) and copies each line to the screen
and the log; it changes no setting. Written with AI assistance (Claude Code), reviewed by the author.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "outputs" / "final_run.log"

if __name__ == "__main__":
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", "-B", str(HERE / "final_run.py"), "--i-confirm-frozen"]
    start = time.perf_counter()
    with open(LOG, "w", encoding="utf-8") as log:
        header = f"[run_final_with_log] started {datetime.now().isoformat(timespec='seconds')}: {' '.join(command)}\n"
        print(header, end="", flush=True)
        log.write(header)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
        footer = (f"[run_final_with_log] finished {datetime.now().isoformat(timespec='seconds')}, exit code {code}, "
                  f"wall clock {time.perf_counter() - start:.0f} s\n")
        print(footer, end="", flush=True)
        log.write(footer)
    sys.exit(code)
