"""Exercise the real Pydantic loop, capability stores, and restart behavior."""

import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.memory import SqliteMemoryStore
from pydantic_ai_harness.planning import SqlitePlanStore
from pydantic_ai_harness.step_persistence import SqliteStepStore

import research
from research_harness.agent import ResearchDeps, run_research
from research_harness.demo_model import respond
from research_harness.memory import connect
from research_harness.runner import run_project


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name) / "project"
        with contextlib.redirect_stdout(io.StringIO()):
            research.create_project(research.parser().parse_args([
                "init", str(self.run), "--tickers", "AAPL",
            ]))
        self.config = json.loads((self.run / "config.json").read_text())
        with connect(self.run) as db:
            db.execute("UPDATE tasks SET status='running', attempts=1")
        self.deps = ResearchDeps(self.run, "AAPL", 1)

    def test_capabilities_persist_and_model_output_is_typed(self):
        result = asyncio.run(run_research(self.deps, self.config))
        self.assertEqual(len(result.output.citations), 2)
        self.assertIn("AAPL", result.output.body)

        async def check():
            # New store objects represent a later Python session.
            steps = SqliteStepStore(database=self.run / "steps.sqlite")
            record = (await steps.list_runs(conversation_id="AAPL"))[0]
            snapshot = await steps.latest_snapshot(run_id=record.run_id)
            self.assertEqual(snapshot.state, "complete")
            self.assertEqual(await steps.list_unresolved_tool_effects(run_id=record.run_id), [])
            plan = await SqlitePlanStore(str(self.run / "plans.sqlite"), session="AAPL").get_items()
            self.assertTrue(all(item.status == "completed" for item in plan))
            memory = SqliteMemoryStore(database=self.run / "notebook.sqlite")
            self.assertIn("AAPL", (await memory.read("AAPL/main/MEMORY.md", max_chars=8000)).content)
            self.assertIsNone(await memory.read("JPM/main/MEMORY.md", max_chars=8000))
        asyncio.run(check())

    def test_search_results_can_support_the_final_typed_note(self):
        def investigate(messages, info):
            observations = {p.tool_name: p.content for m in messages for p in m.parts
                            if isinstance(p, ToolReturnPart)}
            if "search_sources" not in observations:
                return ModelResponse(parts=[ToolCallPart("search_sources", {"query": "generated prices", "limit": 2})])
            if "analyze_prices" not in observations:
                return ModelResponse(parts=[ToolCallPart("analyze_prices", {})])
            found = observations["search_sources"]
            self.assertEqual(found["matched"], 1)
            self.assertEqual(found["items"][0]["ticker"], "AAPL")
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                "body": "AAPL's historical risk calculation uses the dated price series described in the source record.",
                "citations": [observations["analyze_prices"]["id"], found["items"][0]["id"]],
            })])

        result = asyncio.run(run_research(self.deps, self.config, FunctionModel(investigate)))
        self.assertEqual(len(result.output.citations), 2)
        with connect(self.run) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM notes").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM calls WHERE tool='read_source'").fetchone()[0], 0)

    def test_invalid_citation_returns_to_model_for_repair(self):
        invalid_sent = False
        saw_retry = False

        def invalid_once(messages, info):
            nonlocal invalid_sent, saw_retry
            response = respond(messages, info)
            saw_retry |= any(isinstance(p, ToolReturnPart) and p.content == "Unknown or wrong-ticker citation"
                             for m in messages for p in m.parts)
            if response.parts[0].tool_name == info.output_tools[0].name and not invalid_sent:
                invalid_sent = True
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
                    "body": "This note has enough text but its citations do not identify saved evidence.",
                    "citations": ["invented-one", "invented-two"],
                })])
            return response

        result = asyncio.run(run_research(self.deps, self.config, FunctionModel(invalid_once)))
        self.assertTrue(invalid_sent and saw_retry)
        self.assertNotIn("invented-one", result.output.citations)

    def test_resume_after_calculation_in_a_new_worker(self):
        def interrupted(messages, info):
            if any(isinstance(p, ToolReturnPart) and p.tool_name == "analyze_prices"
                   for m in messages for p in m.parts):
                raise RuntimeError("connection lost")
            return respond(messages, info)

        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            asyncio.run(run_research(self.deps, self.config, FunctionModel(interrupted)))
        # The supervisor sees the durable running claim and launches attempt two.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_project(self.run), "COMPLETE")
        with connect(self.run) as db:
            self.assertEqual(db.execute("SELECT attempts FROM tasks").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT count(*) FROM calls WHERE tool='analyze_prices'").fetchone()[0], 1)
            starts = [json.loads(r[0]) for r in db.execute("SELECT payload FROM events WHERE kind='agent_started'")]
            self.assertEqual(starts[-1]["resumed_from"], "AAPL-attempt-1")
            self.assertGreater(starts[-1]["messages"], 0)

    def test_usage_limit_includes_capability_tools(self):
        def endlessly_plan(messages, info):
            return ModelResponse(parts=[ToolCallPart("write_plan", {"items": [{"content": "Still planning"}]})])
        with self.assertRaises(UsageLimitExceeded):
            asyncio.run(run_research(self.deps, {**self.config, "max_tool_calls": 3}, FunctionModel(endlessly_plan)))
        with connect(self.run) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM notes").fetchone()[0], 0)
            usage = json.loads(db.execute("SELECT payload FROM events WHERE kind='agent_usage'").fetchone()[0])
            self.assertEqual(usage["tool_calls"], 3)

    def test_request_limit_stops_an_unproductive_model(self):
        with self.assertRaises(UsageLimitExceeded):
            asyncio.run(run_research(self.deps, {**self.config, "max_requests": 1}))
        with connect(self.run) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM notes").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
