"""Worked example: inspect what survives after the Python process exits.

Create a project with research.py first, then pass its directory here.
This script only reads memory. It does not advance the project.
"""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic_ai_harness.memory import SqliteMemoryStore
from pydantic_ai_harness.planning import SqlitePlanStore
from pydantic_ai_harness.step_persistence import SqliteStepStore

from research_harness.memory import connect, evidence


async def inspect_capabilities(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    if (run / "plans.sqlite").exists():
        print("\nPLANS")
        for ticker in config["tickers"]:
            store = SqlitePlanStore(str(run / "plans.sqlite"), session=ticker)
            print(ticker, [(item.content, item.status) for item in await store.get_items()])
    if (run / "notebook.sqlite").exists():
        print("\nWORKING MEMORY")
        store = SqliteMemoryStore(database=run / "notebook.sqlite")
        for path in await store.list_paths(limit=50):
            item = await store.read(path, max_chars=2000)
            print(path, item.content)
    if (run / "steps.sqlite").exists():
        print("\nAGENT CHECKPOINTS")
        store = SqliteStepStore(database=run / "steps.sqlite")
        for record in await store.list_runs():
            snapshot = await store.latest_snapshot(run_id=record.run_id)
            unresolved = await store.list_unresolved_tool_effects(run_id=record.run_id)
            print(record.run_id, "settled messages:", len(snapshot.messages) if snapshot else 0,
                  "unsettled tools:", len(unresolved))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    with connect(args.run) as db:
        print("TASKS")
        for row in db.execute("SELECT * FROM tasks ORDER BY ticker"):
            print(dict(row))
        print("\nEVIDENCE")
        for item in evidence(db):
            print(item["id"], item["kind"], item["ticker"])
        print("\nRECENT RECEIPTS")
        for row in db.execute("SELECT kind,payload FROM events ORDER BY id DESC LIMIT 8"):
            print(row["kind"], json.loads(row["payload"]))
    asyncio.run(inspect_capabilities(args.run))


if __name__ == "__main__":
    main()
