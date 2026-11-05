"""Small, inspectable retrieval policies for a project-owned source corpus."""

import json
import re


def terms(text):
    """An explicit lexical baseline. It does not recognize synonyms."""
    stopwords = {"a", "an", "and", "are", "for", "in", "is", "of", "on", "the", "to", "with"}
    return set(re.findall(r"[a-z0-9]+", text.lower())) - stopwords


def rank_sources(records, *, ticker, as_of, query):
    """Filter eligibility before ranking. Scope comes from the application."""
    eligible = [r for r in records if r["ticker"] == ticker and r["available_on"] <= as_of]
    query_terms = terms(query)
    scored = [(len(query_terms & terms(r["title"] + " " + r["text"])), r) for r in eligible]
    matches = [(score, r) for score, r in scored if score > 0]
    matches.sort(key=lambda pair: (-pair[0], pair[1]["available_on"], pair[1]["title"], pair[1]["locator"]))
    return eligible, [record for _, record in matches]


def pack_evidence(records, *, limit, max_chars):
    """Fit whole evidence records into a serialized response budget.

    Dates, units, and identifiers travel with each excerpt. Skipped records remain
    in the source corpus. The caller can issue a narrower query or use read_source.
    """
    result = {"items": [], "matched": len(records), "omitted": len(records),
              "budget_chars": max_chars, "limit": limit}
    if len(json.dumps(result, ensure_ascii=False)) > max_chars:
        raise ValueError("Response budget is smaller than the result metadata")
    for record in records:
        if len(result["items"]) == limit:
            break
        candidate = {**result, "items": [*result["items"], record],
                     "omitted": len(records) - len(result["items"]) - 1}
        if len(json.dumps(candidate, ensure_ascii=False)) <= max_chars:
            result = candidate
    return result
