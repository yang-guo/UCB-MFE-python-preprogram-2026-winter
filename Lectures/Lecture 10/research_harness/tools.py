"""Research tools validate inputs, save exact evidence, and check final notes."""

import json
from datetime import date
from pathlib import Path

from .data import fingerprint, load_prices, risk_metrics
from .memory import connect, evidence, evidence_record, event, remember
from .retrieval import pack_evidence, rank_sources

TOOL_NAMES = {"analyze_prices", "read_source", "search_sources", "recall_memory", "save_note"}


def load_sources(path):
    """A bounded local corpus can hold filing excerpts or research notes."""
    records = json.loads(Path(path).read_text())
    required = {"ticker", "title", "available_on", "source_type", "locator", "text"}
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        raise ValueError("Sources must be a list of 1–1000 records")
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("Source needs ticker, title, available_on, source_type, locator, text")
        if any(not isinstance(record[k], str) or not record[k].strip() for k in required):
            raise ValueError("Source fields must be nonempty strings")
        if len(record["text"]) > 8000 or any(len(record[k]) > 500 for k in required - {"text"}):
            raise ValueError("Source excerpt exceeds the context limit")
        if date.fromisoformat(record["available_on"]).isoformat() != record["available_on"]:
            raise ValueError("Use ISO YYYY-MM-DD source availability dates")
    return records


def dispatch(run, ticker, attempt, name, args):
    if name not in TOOL_NAMES:
        raise ValueError(f"Unknown tool: {name}")
    config = json.loads((Path(run) / "config.json").read_text())
    verify_inputs(config)
    # Reserve a call before doing work, so failures also consume the budget.
    with connect(run) as db:
        db.execute("BEGIN IMMEDIATE")
        task = db.execute("SELECT * FROM tasks WHERE ticker=?", (ticker,)).fetchone()
        if task is None or task["status"] != "running" or task["attempts"] != attempt:
            raise ValueError("This task attempt is no longer active")
        used = db.execute("SELECT count(*) FROM calls WHERE ticker=? AND attempt=?", (ticker, attempt)).fetchone()[0]
        if used >= config["max_tool_calls"]:
            raise ValueError("Tool-call budget exhausted for this attempt")
        db.execute("INSERT INTO calls(ticker,attempt,tool,created_at) VALUES(?,?,?,datetime('now'))", (ticker, attempt, name))
    if name == "analyze_prices":
        if args:
            raise ValueError("analyze_prices takes no arguments; the supervisor fixes ticker and dates")
        payload = risk_metrics(load_prices(config, ticker))
        verify_inputs(config)
        payload["source_sha256"] = config["data_sha256"]
        payload["requested_start"] = config["start"]
        payload["requested_as_of"] = config["as_of"]
        with connect(run) as db:
            return remember(db, "metrics", ticker, payload)
    if name == "read_source":
        if args:
            raise ValueError("read_source takes no arguments")
        source = load_sources(config["sources_path"])
        # Availability belongs to the source, not to a model's recollection.
        eligible = [s for s in source if s["ticker"] == ticker and s["available_on"] <= config["as_of"]]
        records = sorted(eligible, key=lambda s: (s["available_on"], s["title"]), reverse=True)[:5]
        verify_inputs(config)
        with connect(run) as db:
            return [remember(db, "source", ticker, {**s, "source_sha256": config["sources_sha256"]}) for s in records]
    if name == "search_sources":
        if not isinstance(args, dict) or not {"query"}.issubset(args) or set(args) - {"query", "limit"}:
            raise ValueError("search_sources requires query and an optional limit")
        query, limit = args["query"], args.get("limit", 3)
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError("Use a search query of 1–200 characters")
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValueError("Use a result limit of 1–5")
        _, ranked = rank_sources(load_sources(config["sources_path"]), ticker=ticker,
                                 as_of=config["as_of"], query=query)
        records = [evidence_record("source", ticker, {**source, "source_sha256": config["sources_sha256"]})
                   for source in ranked]
        result = pack_evidence(records, limit=limit, max_chars=12000)
        verify_inputs(config)
        with connect(run) as db:
            for item in result["items"]:
                remember(db, "source", ticker, item["payload"])
            event(db, "sources_retrieved", {"ticker": ticker, "query": query,
                                           "matched": result["matched"], "returned": len(result["items"]),
                                           "omitted": result["omitted"]})
        return result
    if name == "recall_memory":
        if args:
            raise ValueError("recall_memory takes no arguments")
        with connect(run) as db:
            return evidence(db, ticker)
    if set(args) != {"body", "citations"}:
        raise ValueError("save_note requires body and citations")
    body, citations = args["body"], args["citations"]
    with connect(run) as db:
        db.execute("BEGIN IMMEDIATE")
        task = db.execute("SELECT status,attempts FROM tasks WHERE ticker=?", (ticker,)).fetchone()
        if task["status"] != "running" or task["attempts"] != attempt:
            raise ValueError("Attempt expired before note commit")
        validate_note(db, ticker, body, citations)
        db.execute("INSERT OR REPLACE INTO notes VALUES(?,?,?)", (ticker, body, json.dumps(citations)))
        event(db, "note_saved", {"ticker": ticker, "attempt": attempt, "citations": citations})
    return {"saved": True, "ticker": ticker}


def validate_note(db, ticker, body, citations):
    """The same evidence check runs during model validation and before commit."""
    if not isinstance(body, str) or not 40 <= len(body) <= 3000:
        raise ValueError("A note must contain 40–3000 characters")
    if not isinstance(citations, list) or not citations or not all(isinstance(c, str) for c in citations):
        raise ValueError("Citations must be a nonempty list of evidence IDs")
    known = {e["id"]: e for e in evidence(db, ticker)}
    if any(c not in known for c in citations):
        raise ValueError("Unknown or wrong-ticker citation")
    if not {"metrics", "source"}.issubset({known[c]["kind"] for c in citations}):
        raise ValueError("Cite both computed metrics and a dated source")


def verify_inputs(config):
    for path_key, hash_key in [("data_path", "data_sha256"), ("sources_path", "sources_sha256"), ("holdings_path", "holdings_sha256")]:
        if fingerprint(config[path_key]) != config[hash_key]:
            raise ValueError(f"Input changed: {path_key}. Start a new project to use a new snapshot.")
