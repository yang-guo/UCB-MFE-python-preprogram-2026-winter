"""Check the failure boundaries taught in the fundamentals experiments."""

import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from research_harness.memory import evidence_record
from research_harness.retrieval import pack_evidence, rank_sources

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FundamentalsTests(unittest.TestCase):
    def setUp(self):
        self.records = json.loads((ROOT / "data/retrieval_lab_sources.json").read_text())

    def test_scope_and_cutoff_precede_ranking(self):
        _, ranked = rank_sources(self.records, ticker="AAPL", as_of="2024-04-30",
                                 query="debt refinancing liquidity")
        self.assertEqual({r["record_id"] for r in ranked}, {"aapl-debt", "aapl-cash"})

    def test_budget_preserves_whole_records_and_reports_omissions(self):
        oversized = evidence_record("source", "AAPL", {"text": "x" * 3000, "units": "USD million"})
        small = evidence_record("source", "AAPL", {"text": "Debt 100", "units": "USD million"})
        result = pack_evidence([oversized, small], limit=2, max_chars=500)
        self.assertEqual(result["items"], [small])
        self.assertEqual(result["omitted"], 1)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 500)

    def test_schema_failure_never_enters_calculation(self):
        _, result, entered, calculated = script("05_tool_call_lifecycle.py").experiment()
        self.assertEqual(entered, [12, 3])
        self.assertEqual(len(calculated), 1)
        self.assertAlmostEqual(result.output.total_return, -.01)

    def test_concurrent_memory_edit_preserves_both_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            text = asyncio.run(script("06_memory_and_retrieval.py").conflicting_writes(Path(directory) / "notes.sqlite"))
        self.assertIn("near-term debt", text)
        self.assertIn("refinancing terms", text)

    def test_lost_acknowledgement_does_not_duplicate_effect(self):
        result = script("07_retries_and_effects.py").experiment()
        self.assertEqual(result["naive_writes"], 2)
        self.assertEqual(result["idempotent_writes"], 1)

    def test_retrieval_evaluation_exposes_synonym_miss(self):
        module = script("08_evaluate_retrieval.py")
        baseline, revised = module.evaluate(), module.evaluate(rewrite=True)
        self.assertEqual(baseline[-1]["recall_at_2"], 0)
        self.assertEqual(revised[-1]["recall_at_2"], 1)


if __name__ == "__main__":
    unittest.main()
