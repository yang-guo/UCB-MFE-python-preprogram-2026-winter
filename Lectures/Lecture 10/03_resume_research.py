"""Interrupt an actual agent after its calculation, then resume in a fresh worker.

Usage: python 03_resume_research.py runs/recovery
Use a new directory each time. No provider credentials are needed.
"""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import research
from research_harness.agent import ResearchDeps, run_research
from research_harness.demo_model import respond
from research_harness.memory import connect, run_lock
from research_harness.runner import run_project


def interrupted_model(messages, info):
    calculated = any(isinstance(part, ToolReturnPart) and part.tool_name == "analyze_prices"
                     for message in messages for part in message.parts)
    if calculated:
        raise RuntimeError("Demonstration: model connection lost after the calculation")
    return respond(messages, info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    project = research.parser().parse_args(["init", args.run, "--tickers", "AAPL"])
    research.create_project(project)
    run = Path(args.run).resolve()
    config = json.loads((run / "config.json").read_text())
    with run_lock(run):
        with connect(run) as db:
            db.execute("UPDATE tasks SET status='running', attempts=1 WHERE ticker='AAPL'")
        try:
            asyncio.run(run_research(ResearchDeps(run, "AAPL", 1), config,
                                    FunctionModel(interrupted_model)))
        except RuntimeError as error:
            print(error)
        # Leave the claim as running, just as an interrupted supervisor would.
    print("\nStarting the supervisor again. It will launch a fresh Python worker.")
    state = run_project(run)
    with connect(run) as db:
        calculations = db.execute("SELECT count(*) FROM calls WHERE tool='analyze_prices'").fetchone()[0]
        task = dict(db.execute("SELECT * FROM tasks").fetchone())
        print(f"Calculation calls across both attempts: {calculations}")
        print(task)
    assert state == "COMPLETE" and calculations == 1 and task["attempts"] == 2


if __name__ == "__main__":
    main()
