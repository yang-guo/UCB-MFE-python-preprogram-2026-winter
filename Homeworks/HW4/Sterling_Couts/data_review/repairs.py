"""Produce field-level, reversible repairs only for reviewed statement contexts.

These rows are evidence for a future revised experiment, not edits to the raw
database and not a claim that every other field in the vendor row is verified.
"""

import math


def equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1)


def validate_correction(vendor, reference, correction):
    """An evidence-backed exception must fail closed if inputs later change."""
    for side, values in (
        (vendor, correction["expected_vendor"]),
        (reference, correction["expected_reference"]),
    ):
        if not all(equal(side[key], expected) for key, expected in values.items()):
            raise ValueError(
                "Reviewed correction no longer matches its exact evidence."
            )


def repair_rows(vendor, reference, reason):
    rows = []
    for field in (
        "reported_currency",
        "filing_date",
        "operating_income",
        "total_assets",
    ):
        original, corrected = vendor[field], reference[field]
        if equal(original, corrected):
            continue
        rows.append(
            {
                "symbol": reference["symbol"],
                "cik": reference["cik"],
                "fiscal_date": reference["fiscal_date"],
                "period": "quarter",
                "field": field,
                "original": original,
                "corrected": corrected,
                "available_date": reference["available_date"],
                "accession": reference["accession"],
                "reason": reason,
                "source": reference["filing_index"],
            }
        )
    return rows
