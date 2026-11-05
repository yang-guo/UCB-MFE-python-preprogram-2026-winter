"""A bounded work queue: claim, run, validate, commit, and move to the next task."""

import hashlib
import json
from pathlib import Path
import time

from .memory import connect, event, run_lock
from .worker import ROOT, run_worker
from .report import build_report
from .tools import dispatch, verify_inputs


def code_fingerprint():
    paths = [ROOT / "research.py", ROOT / "pyproject.toml", ROOT / "uv.lock"] + sorted((ROOT / "research_harness").glob("*.py")) + [ROOT / "research_harness/instructions.txt"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def run_project(run, max_tasks=100, seconds=600):
    run = Path(run).resolve()
    config = json.loads((run / "config.json").read_text())
    if max_tasks < 1 or seconds <= 0:
        raise ValueError("max_tasks and seconds must be positive")
    with run_lock(run):
        verify_inputs(config)
        if config["code_sha256"] != code_fingerprint():
            raise ValueError("Harness code changed. Start a new project for the revised code.")
        with connect(run) as db:
            # A previous process died after claiming a task. Its attempt remains charged.
            db.execute("UPDATE tasks SET status='pending',error='Interrupted attempt' WHERE status='running'")
        deadline = time.monotonic() + seconds
        handled = 0
        try:
            while handled < max_tasks and time.monotonic() < deadline:
                with connect(run) as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("UPDATE tasks SET status='failed' WHERE status='pending' AND attempts>=?", (config["max_attempts"],))
                    task = db.execute("SELECT * FROM tasks WHERE status='pending' ORDER BY ticker LIMIT 1").fetchone()
                    if task is None:
                        break
                    ticker, attempt = task["ticker"], task["attempts"] + 1
                    db.execute("UPDATE tasks SET status='running',attempts=?,error=NULL WHERE ticker=?", (attempt, ticker))
                    db.execute("DELETE FROM notes WHERE ticker=?", (ticker,))
                    event(db, "task_started", {"ticker": ticker, "attempt": attempt})
                print(f"{ticker}: attempt {attempt}", flush=True)
                try:
                    run_worker(run, ticker, attempt,
                               min(config["task_timeout"], max(.01, deadline - time.monotonic())))
                    verify_inputs(config)
                    with connect(run) as db:
                        if db.execute("SELECT 1 FROM notes WHERE ticker=?", (ticker,)).fetchone() is None:
                            raise RuntimeError("The agent stopped without saving a valid note")
                        db.execute("UPDATE tasks SET status='complete',error=NULL WHERE ticker=?", (ticker,))
                        event(db, "task_completed", {"ticker": ticker, "attempt": attempt})
                except Exception as error:
                    # Data/schema failures need investigation. Transient process failures
                    # can retry, but the attempt limit survives a program restart.
                    retry = not isinstance(error, ValueError) and attempt < config["max_attempts"]
                    status = "pending" if retry else "failed"
                    with connect(run) as db:
                        db.execute("UPDATE tasks SET status=?,error=? WHERE ticker=?", (status, str(error)[:1000], ticker))
                        event(db, "task_failed", {"ticker": ticker, "attempt": attempt, "error": str(error)[:1000], "retry": retry})
                    if retry:
                        time.sleep(min(2 ** (attempt - 1), max(0, deadline - time.monotonic())))
                handled += 1
                build_report(run)
        except KeyboardInterrupt:
            with connect(run) as db:
                db.execute("UPDATE tasks SET status='pending',error='Interrupted by user' WHERE status='running'")
                event(db, "interrupted", {})
        state = build_report(run)
        print(f"{state}: {run / 'report.html'}")
        return state
