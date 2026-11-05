"""CLI for the final lecture. Run `python research.py --help` from this folder."""

import argparse
from datetime import date
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

from research_harness.data import fingerprint
from research_harness.memory import connect, initialize, run_lock
from research_harness.report import atomic_text, build_report
from research_harness.runner import ROOT, code_fingerprint, run_project
from research_harness.tools import load_sources, verify_inputs


def create_project(args):
    if date.fromisoformat(args.start) >= date.fromisoformat(args.as_of):
        raise ValueError("start must precede as_of")
    tickers = sorted(set(t.strip().upper() for t in args.tickers.split(",")))
    if not tickers or len(tickers) > 50 or any(not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", t) for t in tickers):
        raise ValueError("Use 1–50 ticker symbols")
    if not math.isfinite(args.cash) or args.cash < 0:
        raise ValueError("cash must be a nonnegative finite amount")
    if not 1 <= args.max_attempts <= 5 or not 3 <= args.max_tool_calls <= 50 or not 1 <= args.max_requests <= 50 or args.task_timeout <= 0:
        raise ValueError("Use 1–5 attempts, 3–50 tool calls, 1–50 model requests, and a positive timeout")
    if not 10 <= len(args.question) <= 1000:
        raise ValueError("Use a research question of 10–1000 characters")
    if args.engine == "model" and (not args.model or ":" not in args.model):
        raise ValueError("Choose a model with --model provider:model-id and configure its API key")
    data_path = (ROOT / "data/prices.csv") if args.dataset == "fixture" else (ROOT.parent / "Lecture 4/data/data.db")
    sources_path = ROOT / ("data/sources.json" if args.dataset == "fixture" else "data/course_sources.json")
    config = {
        "contract_version": 2, "question": args.question, "tickers": tickers,
        "start": args.start, "as_of": args.as_of, "dataset": args.dataset,
        "dataset_label": "Generated sample data" if args.dataset == "fixture" else "Lecture 4 price data",
        "engine": args.engine, "model": args.model, "cash": args.cash,
        "max_attempts": args.max_attempts, "max_tool_calls": args.max_tool_calls,
        "max_requests": args.max_requests, "task_timeout": args.task_timeout, "code_sha256": code_fingerprint(),
    }
    for name, path in [("data", Path(args.data) if args.data else data_path),
                       ("sources", Path(args.sources) if args.sources else sources_path),
                       ("holdings", Path(args.holdings) if args.holdings else ROOT / "data/holdings.csv")]:
        config[f"{name}_path"] = str(path.resolve())
        config[f"{name}_sha256"] = fingerprint(path)
    load_sources(config["sources_path"])
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    atomic_text(run / "config.json", json.dumps(config, indent=2))
    initialize(run, tickers)
    build_report(run)
    print(f"Created {run}. Run: python research.py run {args.run}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("init", help="Freeze a question, inputs, and task budgets")
    create.add_argument("run")
    create.add_argument("--dataset", choices=["fixture", "course"], default="fixture")
    create.add_argument("--engine", choices=["demo", "model"], default="demo")
    create.add_argument("--model")
    create.add_argument("--tickers", default="AAPL,MSFT,JPM")
    create.add_argument("--start", default="2024-01-02")
    create.add_argument("--as-of", default="2024-04-30")
    create.add_argument("--question", default="How concentrated is the portfolio, and how do its holdings compare on historical risk?")
    for name in ["data", "sources", "holdings"]:
        create.add_argument(f"--{name}")
    create.add_argument("--cash", type=float, default=5000)
    create.add_argument("--max-attempts", type=int, default=2)
    create.add_argument("--max-tool-calls", type=int, default=20)
    create.add_argument("--max-requests", type=int, default=12)
    create.add_argument("--task-timeout", type=int, default=120)
    run = sub.add_parser("run", help="Run or resume pending work")
    run.add_argument("run")
    run.add_argument("--max-tasks", type=int, default=100, help="Maximum attempts in this invocation")
    run.add_argument("--seconds", type=float, default=600, help="Wall-clock budget for this invocation")
    for name in ["status", "report"]:
        command = sub.add_parser(name)
        command.add_argument("run")
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == "init":
            create_project(args)
        elif args.command == "run":
            return 0 if run_project(args.run, args.max_tasks, args.seconds) == "COMPLETE" else 2
        elif args.command == "status":
            with connect(args.run) as db:
                print(json.dumps([dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY ticker")], indent=2))
        elif args.command == "report":
            with run_lock(args.run):
                verify_inputs(json.loads((Path(args.run) / "config.json").read_text()))
                print(build_report(args.run))
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
