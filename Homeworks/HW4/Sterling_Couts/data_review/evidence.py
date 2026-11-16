"""Replay pinned public facts offline; never silently consume a changed source."""

import hashlib
import json

from data_review.facts import flatten


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_facts(root, entity, *, offline=False):
    cik = entity["cik"]
    evidence = root / "data_review/evidence"
    manifest = json.loads((evidence / "source_manifest.json").read_text())
    expected = next(
        row["sha256"]
        for row in manifest
        if row["cik"] == cik and row["kind"] == "facts" and row["status"] == "cached"
    )
    path = root / f"outputs/data_review/sec/{cik:010d}_facts.json"
    if path.exists() and not offline:
        if digest(path) != expected:
            raise ValueError("SEC cache changed; review its new vintage before use.")
        payload = json.loads(path.read_text())
        if int(payload["cik"]) != cik:
            raise ValueError("SEC payload belongs to another issuer.")
        facts = flatten(payload, tag_overrides=entity.get("tag_overrides"))
    else:
        compact = evidence / "selected_facts.json"
        if digest(compact) != (evidence / "selected_facts.sha256").read_text().strip():
            raise ValueError("The compact evidence snapshot has changed.")
        saved = next(
            row for row in json.loads(compact.read_text()) if row["cik"] == cik
        )
        if saved["source_sha256"] != expected:
            raise ValueError("Compact facts do not match the pinned source manifest.")
        if saved.get("tag_overrides", {}) != entity.get("tag_overrides", {}):
            raise ValueError("Tag configuration differs from the reviewed snapshot.")
        facts = saved["facts"]
    return {
        "cik": cik,
        "source_sha256": expected,
        "tag_overrides": entity.get("tag_overrides", {}),
        "facts": facts,
    }
