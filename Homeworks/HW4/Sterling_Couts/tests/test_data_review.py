"""Independent edge cases for the data-review layer, not copies of live outputs."""

import pytest
import pandas as pd

from data_review.facts import UnsupportedFact, choose, quarter, statement
from data_review.identity import attach_identities, collapse_identical_aliases


def fact(
    value, end, filed, start=None, metric="operating_cash_flow", acc="a", unit="USD"
):
    return {
        "value": value,
        "start": start,
        "end": end,
        "filed": filed,
        "metric": metric,
        "tag": metric,
        "unit": unit,
        "accession": acc,
        "form": "10-Q",
    }


def test_future_restatement_cannot_enter_old_asof():
    rows = [
        fact(10, "2015-03-31", "2015-05-01"),
        fact(99, "2015-03-31", "2016-05-01", acc="later"),
    ]
    assert choose(rows, "2015-03-31", "2015-06-01")["value"] == 10


def test_conflicting_currency_is_not_assumed_usd():
    rows = [
        fact(10, "2015-03-31", "2015-05-01"),
        fact(10, "2015-03-31", "2015-05-01", unit="GBP"),
    ]
    with pytest.raises(UnsupportedFact):
        choose(rows, "2015-03-31", "2015-06-01")


def test_quarterly_cash_flow_is_ytd_difference():
    rows = [
        fact(100, "2015-03-31", "2015-05-01", "2015-01-01", acc="q1"),
        fact(250, "2015-06-30", "2015-08-01", "2015-01-01", acc="q2"),
        fact(900, "2015-03-31", "2016-05-01", "2015-01-01", acc="future"),
    ]
    q = quarter(rows, "2015-06-30", "2015-08-01", "q2", "USD")
    assert q["value"] == 150
    assert q["start"] == "2015-04-01"
    assert q["method"] == "ytd_difference"


def test_weekend_fiscal_end_is_preserved():
    rows = [
        fact(20, "2015-03-28", "2015-04-28", "2014-12-28", "operating_income"),
        fact(200, "2015-03-28", "2015-04-28", metric="total_assets"),
    ]
    result = statement(rows, "2015-03-28")
    assert result["operating_roa"] == 0.1
    assert result["available_date"] == "2015-04-29"
    assert result["fiscal_date"] == "2015-03-28"
    assert result["free_cash_flow_ppe_only"] is None


def test_same_day_filing_is_not_repaired_by_shift():
    rows = [
        fact(20, "2015-03-31", "2015-03-31", "2015-01-01", "operating_income"),
        fact(200, "2015-03-31", "2015-03-31", metric="total_assets"),
    ]
    with pytest.raises(UnsupportedFact):
        statement(rows, "2015-03-31")


def test_unmatched_fiscal_date_is_not_rounded_to_calendar_quarter():
    rows = [
        fact(20, "2015-03-28", "2015-04-28", "2014-12-28", "operating_income"),
        fact(200, "2015-03-28", "2015-04-28", metric="total_assets"),
    ]
    with pytest.raises(UnsupportedFact):
        statement(rows, "2015-03-31")


def test_cash_flow_period_must_match_income_period():
    rows = [
        fact(20, "2015-03-31", "2015-05-01", "2015-01-01", "operating_income"),
        fact(200, "2015-03-31", "2015-05-01", metric="total_assets"),
        fact(50, "2015-03-31", "2015-05-01", "2014-12-25"),
    ]
    assert statement(rows, "2015-03-31")["operating_cash_flow"] is None


def test_assets_from_another_filing_do_not_silently_mix():
    rows = [
        fact(20, "2015-03-31", "2015-05-01", "2015-01-01", "operating_income"),
        fact(200, "2015-03-31", "2016-05-01", metric="total_assets", acc="later"),
    ]
    with pytest.raises(UnsupportedFact):
        statement(rows, "2015-03-31")


REGISTRY = [
    {
        "security_id": "one",
        "issuer_id": "issuer",
        "tables": ["prices"],
        "scope_start": "2015-01-01",
        "scope_end": "2015-12-31",
        "source_priority": ["OLD", "NEW"],
    }
]


def test_alias_correction_is_date_bounded_and_preserves_other_classes():
    raw = pd.DataFrame(
        {
            "symbol": ["OLD", "NEW", "CLASS_B", "NEW"],
            "date": ["2015-03-31"] * 3 + ["2025-03-31"],
            "value": [1, 1, 2, 3],
        }
    )
    identified = attach_identities(raw, REGISTRY, "date", table="prices")
    clean, removed, conflicts = collapse_identical_aliases(
        identified, ["security_id", "date"], ["value"]
    )
    assert len(clean) == 3
    assert len(removed) == 1 and removed.symbol.iloc[0] == "NEW"
    assert (
        clean.loc[clean.date.eq("2025-03-31"), "identity_status"].iloc[0]
        == "unverified"
    )
    assert "CLASS_B" in clean.symbol.to_list()
    assert conflicts.empty


def test_alias_conflict_is_quarantined_not_chosen_for_better_result():
    raw = pd.DataFrame(
        {"symbol": ["OLD", "NEW"], "date": ["2015-03-31"] * 2, "value": [1, 999]}
    )
    clean, removed, conflicts = collapse_identical_aliases(
        attach_identities(raw, REGISTRY, "date", table="prices"),
        ["security_id", "date"],
        ["value"],
    )
    assert clean.empty and removed.empty and len(conflicts) == 2


def test_price_identity_does_not_authorize_financial_identity():
    raw = pd.DataFrame({"symbol": ["OLD", "NEW"], "date": ["2015-03-31"] * 2})
    result = attach_identities(raw, REGISTRY, "date", table="fundamentals")
    assert result.identity_status.eq("unverified").all()
    assert result.security_id.nunique() == 2


def test_annual_minus_nine_months_gives_fourth_quarter():
    rows = [
        fact(300, "2015-09-30", "2015-11-01", "2015-01-01", acc="q3"),
        fact(420, "2015-12-31", "2016-02-01", "2015-01-01", acc="annual"),
    ]
    rows[1]["form"] = "10-K"
    result = quarter(rows, "2015-12-31", "2016-02-01", "annual", "USD")
    assert result["value"] == 120 and result["start"] == "2015-10-01"


def test_statement_does_not_arbitrarily_select_translation_currency():
    rows = [
        fact(20, "2015-03-31", "2015-05-01", "2015-01-01", "operating_income"),
        fact(200, "2015-03-31", "2015-05-01", metric="total_assets"),
    ]
    rows += [{**row, "unit": "GBP", "value": row["value"] / 2} for row in rows]
    with pytest.raises(UnsupportedFact):
        statement(rows, "2015-03-31")


def test_reviewed_correction_rejects_changed_vendor_input():
    from data_review.repairs import validate_correction

    correction = {
        "expected_vendor": {"total_assets": 10},
        "expected_reference": {"total_assets": 20},
    }
    with pytest.raises(ValueError):
        validate_correction({"total_assets": 999}, {"total_assets": 20}, correction)


def test_pinned_apple_filing_reconstructs_known_quarter_and_vintage():
    from pathlib import Path
    from data_review.evidence import load_facts

    root = Path(__file__).resolve().parents[1]
    facts = load_facts(
        root,
        {
            "cik": 320193,
            "tag_overrides": {
                "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
            },
        },
        offline=True,
    )["facts"]
    original_q4 = statement(facts, "2012-09-29")
    assert original_q4["operating_income"] == (55241 - 44297) * 1_000_000
    assert original_q4["available_date"] == "2012-11-01"
    original_2015 = statement(facts, "2015-09-26")
    assert original_2015["total_assets"] == 290479 * 1_000_000
    assert original_2015["accession"] == "0001193125-15-356351"


def test_pinned_fluidigm_fcf_definitions_remain_distinct():
    from pathlib import Path
    from data_review.evidence import load_facts

    facts = load_facts(
        Path(__file__).resolve().parents[1], {"cik": 1162194}, offline=True
    )["facts"]
    q1 = statement(facts, "2015-03-31")
    assert q1["operating_cash_flow"] == -9_800_000
    assert q1["capex_ppe"] == 909_000 and q1["capex_intangibles"] == 120_000
    assert q1["free_cash_flow_ppe_only"] == -10_709_000
    assert q1["free_cash_flow_ppe_only"] - q1["capex_intangibles"] == -10_829_000
