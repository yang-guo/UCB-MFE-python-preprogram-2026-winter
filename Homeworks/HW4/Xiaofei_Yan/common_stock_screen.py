"""Conservative ordinary-share candidate screen, not a certified PIT master.

The course database lacks dated issue types. Quarantine clearly non-common
and ambiguous subordinate issues; do not infer historical types from future
returns, active status, or a later delisting. REIT common shares remain eligible.
"""

import re
import pandas as pd


POLICY_VERSION = "common-stock-review-v1"
# These additional issue symbols need verified issue-type histories before
# inclusion. Exclusion is conservative uncertainty, not a claim that snapshot
# issuer names identify their security classes accurately.
REVIEW_ISSUES = set("AESC BDXA BDXB BWNB BWSN CFXA DCUE DTP DTY FCRW HCXZ HTFB HTIA IFFT PCGU PBY RBCP RZA RZB UGIC".split())


def review_universe(universe):
    """Return a reproducible candidate mask and reason for every source symbol."""
    records = []
    for row in universe.itertuples():
        symbol, name = row.symbol, str(row.name)
        reason, source = "ordinary_share_candidate", "course universe; no certified historical type field"
        if symbol in REVIEW_ISSUES:
            reason = "subordinate_issue_type_unconfirmed"
        elif re.search(r"warrant|wrrt|\d+(?:\.\d+)?\s*%|CAP SECS", name, re.I):
            reason = "explicit_noncommon_name_or_debt_coupon"
        elif symbol.endswith("-P"):
            reason = "preferred_issue_symbol_review"
        elif len(symbol) == 5 and symbol.isalpha() and symbol[-1] in "CGHILMNOPRSTUVWXYZ":
            # Modern five-letter roots exist: this is a conservative review
            # quarantine, not a universal suffix-to-security classification.
            reason = "five_character_subordinate_or_uncertain_issue"
        elif re.search(r"\bL\.?P\.?\b|royalty trust", name, re.I):
            reason = "partnership_or_royalty_units_not_common_stock"
        records.append(dict(symbol=symbol, name=name, common_stock_candidate=reason == "ordinary_share_candidate",
                            reason=reason, source=source, policy_version=POLICY_VERSION))
    return pd.DataFrame(records)


def load_security_review(database):
    """Review source metadata without reading prices or subsequent outcomes."""
    import duckdb
    with duckdb.connect(str(database), read_only=True) as connection:
        universe = connection.execute("SELECT symbol,name FROM universe ORDER BY symbol").fetchdf()
    return review_universe(universe)
