"""Evaluate the lexical baseline on questions with known supporting excerpts."""

import argparse
import json
from pathlib import Path

from research_harness.retrieval import rank_sources

CASES = [
    ("debt refinancing", {"aapl-debt"}),
    ("liquidity", {"aapl-cash"}),
    ("supplier concentration", {"aapl-supply"}),
    ("rollover obligations", {"aapl-debt"}),
]


def evaluate(rewrite=False):
    path = Path(__file__).resolve().parent / "data/retrieval_lab_sources.json"
    records = json.loads(path.read_text())
    results = []
    for question, expected in CASES:
        query = "debt refinancing" if rewrite and question == "rollover obligations" else question
        _, ranked = rank_sources(records, ticker="AAPL", as_of="2024-04-30", query=query)
        returned = {r["record_id"] for r in ranked[:2]}
        relevant = returned & expected
        results.append({"question": question, "query": query, "returned": sorted(returned),
                        "recall_at_2": len(relevant) / len(expected),
                        "precision_at_2": len(relevant) / len(returned) if returned else 0.0})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite", action="store_true", help="Try one explicit synonym rewrite")
    args = parser.parse_args()
    results = evaluate(args.rewrite)
    print(json.dumps(results, indent=2))
    print("Mean recall@2:", sum(r["recall_at_2"] for r in results) / len(results))
