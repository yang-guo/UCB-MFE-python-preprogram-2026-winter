"""Reconstruct selected statement facts with explicit units and filing vintages.

Do not use a vendor's latest value at an old fiscal date. Every input fact carries
its actual SEC accession and filing date. Quarterly cash flow may require
subtracting two fiscal-year-to-date disclosures; market-session counts are never
used to identify a fiscal quarter. Unsupported or ambiguous cases raise, rather
than acquiring an invented date, currency, zero or amount.
"""

from datetime import date, timedelta
import math

TAGS = {
    "operating_income": "OperatingIncomeLoss",
    "total_assets": "Assets",
    "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
    "capex_ppe": "PaymentsToAcquirePropertyPlantAndEquipment",
    "capex_intangibles": "PaymentsToAcquireIntangibleAssets",
}
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


class UnsupportedFact(ValueError):
    """Missing, inconsistent or ambiguously identified source evidence."""


def day(value):
    return date.fromisoformat(str(value)[:10])


def flatten(payload, start="2009-01-01", end="2018-12-31", tag_overrides=None):
    """Keep exact fact contexts, including all filed vintages, for selected tags."""
    rows = []
    tags = {**TAGS, **(tag_overrides or {})}
    for metric, tag in tags.items():
        concept = payload.get("facts", {}).get("us-gaap", {}).get(tag, {})
        for unit, values in concept.get("units", {}).items():
            for fact in values:
                if start <= fact["end"] <= end:
                    rows.append(
                        {
                            "metric": metric,
                            "tag": tag,
                            "unit": unit,
                            "value": fact["val"],
                            "start": fact.get("start"),
                            "end": fact["end"],
                            "filed": fact["filed"],
                            "accession": fact["accn"],
                            "form": fact["form"],
                        }
                    )
    return rows


def choose(rows, end, asof, start=None, accession=None, unit=None):
    """Select the latest known vintage for an exact period and optional filing."""
    available = [
        r
        for r in rows
        if r["end"] == end
        and r["form"] in FORMS
        and day(r["filed"]) <= day(asof)
        and day(r["end"]) <= day(r["filed"])
        and (start is None or r.get("start") == start)
        and (accession is None or r["accession"] == accession)
        and (unit is None or r["unit"] == unit)
        and math.isfinite(r["value"])
    ]
    if not available:
        raise UnsupportedFact("No supported fact available for this period/filing.")
    latest = max(r["filed"] for r in available)
    latest_rows = [r for r in available if r["filed"] == latest]
    signatures = {
        (r.get("start"), r["end"], r["unit"], r["value"]) for r in latest_rows
    }
    if len(signatures) != 1:
        raise UnsupportedFact(
            "Conflicting units, periods or values in the latest filing date."
        )
    # Identical duplicate disclosures are harmless; the tie-break is deterministic.
    return sorted(latest_rows, key=lambda r: r["accession"])[0]


def quarter(rows, end, asof, accession, unit):
    """Direct quarter, or YTD minus prior YTD using only then-available facts."""
    current = [
        r
        for r in rows
        if r["end"] == end
        and r.get("start")
        and r["accession"] == accession
        and r["unit"] == unit
        and r["form"] in FORMS
        and r["filed"] <= asof
    ]
    direct = [r for r in current if 70 <= (day(end) - day(r["start"])).days + 1 <= 110]
    if direct:
        fact = choose(direct, end, asof, accession=accession, unit=unit)
        return {
            "value": fact["value"],
            "start": fact["start"],
            "end": end,
            "unit": unit,
            "method": "direct_quarter",
            "facts": [fact],
        }
    cumulative = [
        r for r in current if 111 <= (day(end) - day(r["start"])).days + 1 <= 380
    ]
    fact = choose(cumulative, end, asof, accession=accession, unit=unit)
    preceding = [
        r
        for r in rows
        if r.get("start") == fact["start"]
        and r["unit"] == unit
        and r["filed"] <= asof
        and r["form"] in FORMS
        and 70 <= (day(end) - day(r["end"])).days <= 110
    ]
    prior_ends = {r["end"] for r in preceding}
    if len(prior_ends) != 1:
        raise UnsupportedFact("No unique preceding fiscal YTD period.")
    prior = choose(preceding, prior_ends.pop(), asof, start=fact["start"], unit=unit)
    return {
        "value": fact["value"] - prior["value"],
        "start": (day(prior["end"]) + timedelta(days=1)).isoformat(),
        "end": end,
        "unit": unit,
        "method": "ytd_difference",
        "facts": [fact, prior],
    }


def statement(rows, end):
    """Earliest supported ROA statement for an exact fiscal end; optional flows.

    Later availability is carried into output; never backdate a fact to its fiscal
    end. Missing optional capex/cash-flow evidence stays missing. PP&E-only FCF
    is a named standardized alternative, not a silent overwrite of vendor FCF.
    """
    by_metric = {name: [r for r in rows if r["metric"] == name] for name in TAGS}
    candidates = sorted(
        {
            (r["filed"], r["accession"], r["unit"])
            for r in by_metric["operating_income"]
            if r["end"] == end and r["form"] in FORMS and r["filed"] > end
        }
    )
    for filed, accession, unit in candidates:
        units = {
            r["unit"]
            for r in by_metric["operating_income"]
            if r["end"] == end and r["accession"] == accession
        }
        if len(units) != 1 or len(unit) != 3 or not unit.isalpha():
            # Do not pick whichever translated currency happens to sort first.
            continue
        try:
            income = quarter(by_metric["operating_income"], end, filed, accession, unit)
            assets = choose(
                by_metric["total_assets"], end, filed, accession=accession, unit=unit
            )
        except UnsupportedFact:
            continue
        if assets["value"] <= 0:
            continue
        values = {"operating_income": income}
        optional_issues = {}
        for name in ("operating_cash_flow", "capex_ppe", "capex_intangibles"):
            try:
                value = quarter(by_metric[name], end, filed, accession, unit)
                if value["start"] != income["start"]:
                    raise UnsupportedFact(
                        "Cash-flow and operating-income quarters differ."
                    )
                if name.startswith("capex") and value["value"] < 0:
                    raise UnsupportedFact(
                        "Unexpected negative payment amount; inspect sign convention."
                    )
                values[name] = value
            except UnsupportedFact as exc:
                optional_issues[name] = str(exc)
        result = {
            "fiscal_date": end,
            "period_start": income["start"],
            "filing_date": filed,
            "available_date": (day(filed) + timedelta(days=1)).isoformat(),
            "reported_currency": unit,
            "accession": accession,
            "operating_income": income["value"],
            "total_assets": assets["value"],
            "operating_roa": income["value"] / assets["value"],
            "flow_provenance": values,
            "assets_provenance": assets,
            "optional_issues": optional_issues,
        }
        for name in ("operating_cash_flow", "capex_ppe", "capex_intangibles"):
            result[name] = values[name]["value"] if name in values else None
        cf, capex = result["operating_cash_flow"], result["capex_ppe"]
        result["free_cash_flow_ppe_only"] = (
            cf - capex if cf is not None and capex is not None else None
        )
        return result
    raise UnsupportedFact(
        "No unambiguous statement with an exact fiscal date and common filing/units."
    )
