"""Check failures that a successful-looking research report could otherwise hide."""

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research
from research_harness.data import load_prices, portfolio_exposure, risk_metrics
from research_harness.memory import connect, evidence, run_lock
from research_harness.report import build_report
from research_harness.runner import run_project
from research_harness.tools import dispatch, verify_inputs


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name) / "project"

    def create(self, *flags):
        args = research.parser().parse_args(["init", str(self.run), *flags])
        with contextlib.redirect_stdout(io.StringIO()):
            research.create_project(args)
        return json.loads((self.run / "config.json").read_text())

    def execute(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_project(self.run, **kwargs)

    def claim(self, ticker="AAPL"):
        with connect(self.run) as db:
            db.execute("UPDATE tasks SET status='running',attempts=1 WHERE ticker=?", (ticker,))

    def test_arithmetic_and_cash_denominator(self):
        frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=21),
                              "close": [100., 110., 99.] + [99.] * 18,
                              "adj_close": [100., 110., 99.] + [99.] * 18})
        m = risk_metrics(frame)
        self.assertAlmostEqual(m["total_return"], -.01)
        self.assertAlmostEqual(m["max_drawdown"], -.10)
        self.assertAlmostEqual(m["annualized_log_volatility"], np.std(np.log(frame.adj_close / frame.adj_close.shift()).dropna(), ddof=1) * np.sqrt(252))
        holdings = Path(self.directory.name) / "holdings.csv"
        holdings.write_text("ticker,shares,entry_price\nAAPL,1,80\nAAPL,3,100\n")
        p = portfolio_exposure(holdings, {"AAPL": m}, 100)
        self.assertEqual(p["positions"][0]["weighted_entry_price"], 95)
        self.assertAlmostEqual(p["positions"][0]["weight"] + p["cash_weight"], 1)

    def test_pause_resume_does_not_repeat_completed_tasks(self):
        self.create()
        self.assertEqual(self.execute(max_tasks=1), "INCOMPLETE")
        self.assertEqual(self.execute(), "COMPLETE")
        self.assertEqual(self.execute(), "COMPLETE")
        with connect(self.run) as db:
            self.assertEqual(db.execute("SELECT sum(attempts) FROM tasks").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT count(*) FROM calls").fetchone()[0], 12)

    def test_failed_ticker_does_not_disappear(self):
        self.create("--tickers", "AAPL,ZZZZ")
        self.assertEqual(self.execute(), "INCOMPLETE")
        summary = json.loads((self.run / "summary.json").read_text())
        self.assertEqual(summary["tasks"][1]["status"], "failed")
        self.assertIsNone(summary["portfolio"])
        self.assertIn("ZZZZ: failed", (self.run / "report.md").read_text())

    def test_citations_and_source_availability(self):
        self.create()
        self.claim()
        metric = dispatch(self.run, "AAPL", 1, "analyze_prices", {})
        sources = dispatch(self.run, "AAPL", 1, "read_source", {})
        self.assertEqual(len(sources), 1)  # The May note is ineligible in April.
        body = "A sufficiently long note that needs traceable evidence for its claims."
        with self.assertRaisesRegex(ValueError, "Unknown"):
            dispatch(self.run, "AAPL", 1, "save_note", {"body": body, "citations": ["invented-id"]})
        with self.assertRaisesRegex(ValueError, "both"):
            dispatch(self.run, "AAPL", 1, "save_note", {"body": body, "citations": [metric["id"]]})

    def test_budget_counts_failed_calls(self):
        self.create("--max-tool-calls", "3")
        self.claim()
        for _ in range(3):
            with self.assertRaises(ValueError):
                dispatch(self.run, "AAPL", 1, "analyze_prices", {"ticker": "MSFT"})
        with self.assertRaisesRegex(ValueError, "budget"):
            dispatch(self.run, "AAPL", 1, "recall_memory", {})

    def test_source_search_returns_citable_records_within_budget(self):
        self.create("--sources", str(research.ROOT / "data/retrieval_lab_sources.json"))
        self.claim()
        result = dispatch(self.run, "AAPL", 1, "search_sources", {"query": "debt refinancing liquidity", "limit": 2})
        self.assertEqual({r["payload"]["record_id"] for r in result["items"]}, {"aapl-debt", "aapl-cash"})
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 12000)
        with connect(self.run) as db:
            self.assertEqual({r["id"] for r in evidence(db, "AAPL")}, {r["id"] for r in result["items"]})

    def test_changed_snapshot_is_rejected(self):
        data = Path(self.directory.name) / "prices.csv"
        data.write_bytes((research.ROOT / "data/prices.csv").read_bytes())
        config = self.create("--data", str(data))
        data.write_text(data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Input changed"):
            verify_inputs(config)

    def test_internal_gap_and_duplicate_are_rejected(self):
        config = self.create()
        frame = pd.read_csv(config["data_path"])
        file = Path(self.directory.name) / "bad.csv"
        for bad, message in [(frame.drop(index=4), "gaps"), (pd.concat([frame, frame.iloc[[4]]]), "Duplicate")]:
            bad.to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, message):
                load_prices({**config, "data_path": str(file)}, "AAPL")

    def test_no_adjusted_close_fallback(self):
        config = self.create()
        file = Path(self.directory.name) / "test.sqlite"
        with sqlite3.connect(file) as db:
            db.execute("CREATE TABLE ohlc(ticker,ts,close,adj_close)")
            db.execute("INSERT INTO ohlc VALUES('AAPL','2024-01-02',100,NULL)")
        with self.assertRaisesRegex(ValueError, "Invalid adj_close"):
            load_prices({**config, "dataset": "course", "data_path": str(file)}, "AAPL")

    def test_crash_recovery_and_cumulative_retry_limit(self):
        self.create("--tickers", "AAPL")
        self.claim()  # Simulate death after durable claim, before completion.
        with patch("research_harness.runner.run_worker", side_effect=RuntimeError("temporary provider outage")):
            self.assertEqual(self.execute(), "INCOMPLETE")
            self.assertEqual(self.execute(), "INCOMPLETE")
        with connect(self.run) as db:
            task = db.execute("SELECT * FROM tasks").fetchone()
            self.assertEqual(task["attempts"], 2)
            self.assertEqual(task["status"], "failed")

    def test_concurrent_supervisor_is_rejected(self):
        self.create()
        with run_lock(self.run), self.assertRaises(RuntimeError):
            with run_lock(self.run):
                pass

    def test_report_rechecks_evidence_and_escapes_html(self):
        self.create()
        self.execute()
        with connect(self.run) as db:
            db.execute("UPDATE notes SET body='<script>alert(1)</script>' WHERE ticker='AAPL'")
        build_report(self.run)
        self.assertNotIn("<script>", (self.run / "report.html").read_text())
        with connect(self.run) as db:
            db.execute("UPDATE notes SET citations='[\"missing\"]' WHERE ticker='AAPL'")
        with self.assertRaisesRegex(ValueError, "Broken"):
            build_report(self.run)


if __name__ == "__main__":
    unittest.main()
