"""Compare retrieval policies, response budgets, and concurrent memory updates."""

import asyncio
import json
from pathlib import Path
import tempfile

from pydantic_ai_harness.memory import MemoryConflictError, SqliteMemoryStore

from research_harness.memory import evidence_record
from research_harness.retrieval import pack_evidence, rank_sources, terms

ROOT = Path(__file__).resolve().parent


def retrieval_experiment():
    records = json.loads((ROOT / "data/retrieval_lab_sources.json").read_text())
    query = "debt refinancing liquidity"
    # This is the deliberate mistake: spend the top-k slots before eligibility.
    ranked_globally = sorted(records, key=lambda r: -len(terms(query) & terms(r["title"] + " " + r["text"])))
    bad = [r for r in ranked_globally[:2] if r["ticker"] == "AAPL" and r["available_on"] <= "2024-04-30"]
    eligible, ranked = rank_sources(records, ticker="AAPL", as_of="2024-04-30", query=query)
    candidates = [evidence_record("source", "AAPL", r) for r in ranked]
    packed = pack_evidence(candidates, limit=2, max_chars=600)
    larger = pack_evidence(candidates, limit=2, max_chars=850)
    print("Top two before filtering:", [r["record_id"] for r in ranked_globally[:2]])
    print("After filtering those two:", [r["record_id"] for r in bad])
    print("Filter first, then rank:", [r["record_id"] for r in ranked])
    print("Packed evidence:", [r["payload"]["record_id"] for r in packed["items"]])
    print("Matched:", packed["matched"], "Omitted:", packed["omitted"])
    print("Serialized response characters:", len(json.dumps(packed, ensure_ascii=False)))
    print("With 850 characters:", [r["payload"]["record_id"] for r in larger["items"]])
    _, synonym_query = rank_sources(records, ticker="AAPL", as_of="2024-04-30", query="rollover obligations")
    print("Matches for 'rollover obligations':", len(synonym_query))
    assert len(eligible) == 3 and not bad and len(ranked) == 2
    return packed


async def conflicting_writes(database):
    store = SqliteMemoryStore(database=database)
    key = "AAPL/main/MEMORY.md"
    await store.write(key, "Check debt maturities.", expected_version=None)
    researcher_a = await store.read(key, max_chars=8000)
    researcher_b = await store.read(key, max_chars=8000)
    await store.write(key, researcher_a.content + " Compare cash with near-term debt.",
                      expected_version=researcher_a.version)
    try:
        await store.write(key, researcher_b.content + " Inspect refinancing terms.",
                          expected_version=researcher_b.version)
    except MemoryConflictError:
        print("\nWriter B's stale version was rejected.")
    else:
        raise AssertionError("A stale memory update should conflict")
    current = await store.read(key, max_chars=8000)
    # Resolve deliberately against the new version, preserving both additions.
    await store.write(key, current.content + " Inspect refinancing terms.", expected_version=current.version)
    final = await store.read(key, max_chars=8000)
    print("Merged memory:", final.content)
    return final.content


if __name__ == "__main__":
    retrieval_experiment()
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(conflicting_writes(Path(directory) / "memory.sqlite"))
