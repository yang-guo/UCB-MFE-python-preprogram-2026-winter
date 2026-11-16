"""Causal financial-change features, extending Lecture 7's ratio features.

AI-assisted implementation reviewed with executable counterexamples. These are
research hypotheses, not data repairs. The source is a restated snapshot, not
an independently verified archive of original filings. Unknown period/currency
metadata stays unknown; no currency or missing financial amount is invented.
"""

import numpy as np
import pandas as pd


FEATURES = (
    "operating_margin_change",
    "cash_profitability_change",
    "net_deleveraging",
)
META = (
    "period_start",
    "reported_currency",
    "accounting_basis",
    "source",
)


def _known(value):
    return pd.notna(value) and str(value).strip() != ""


def _ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return numerator / denominator


def _flow_valid(row):
    # The course labels records quarterly. Accept that convention when the
    # start date is absent, but reject a contradictory known period length.
    start = row.get("period_start", pd.NaT)
    return pd.isna(start) or 60 <= (row["fiscal_date"] - start).days + 1 <= 120


def _net_leverage(row):
    debt, cash = row["total_debt"], row["cash_and_equivalents"]
    if pd.isna(debt) or pd.isna(cash) or debt < 0 or cash < 0:
        return np.nan
    return _ratio(debt - cash, row["total_assets"])


def build_change_events(financials, cutoff="2015-12-31"):
    """Compare the newest known quarter to a known quarter 350–380 days earlier.

    The filing must follow the fiscal end; information becomes usable the next
    calendar day, exactly as in the course. Pair by fiscal dates, not row count:
    shift(4) silently fails when a quarter is missing. A late prior-year filing
    can enable a comparison only when it actually becomes available. It never
    rewrites an earlier event, and cannot replace the latest fiscal period.

    Same-source ratios are dimensionless. Missing currency/period metadata is
    flagged under the course convention. Known source, currency or accounting
    basis conflicts suppress all three changes. Contradictory flow durations
    suppress only flow ratios. Ambiguous prior-year matches remain missing.
    """
    f = financials.copy()
    for col in META:
        if col not in f:
            f[col] = pd.NaT if col == "period_start" else None
    for col in ["fiscal_date", "filing_date", "period_start"]:
        f[col] = pd.to_datetime(f[col]).astype("datetime64[ns]")
    f["change_available_date"] = f.filing_date + pd.Timedelta(days=1)
    valid = (
        f.period.eq("quarter")
        & f.filing_date.gt(f.fiscal_date)
        & f.change_available_date.le(pd.Timestamp(cutoff))
    )
    f = f.loc[valid].sort_values(["symbol", "change_available_date", "fiscal_date"])
    if f.duplicated(["symbol", "fiscal_date", "change_available_date"]).any():
        raise ValueError("Ambiguous duplicate financial events")
    rows = []
    for symbol, group in f.groupby("symbol", sort=False):
        known = {}
        for available, batch in group.groupby("change_available_date", sort=True):
            for record in batch.to_dict("records"):
                known[record["fiscal_date"]] = record
            current_date = max(known)
            current = known[current_date]
            priors = [
                v for d, v in known.items() if 350 <= (current_date - d).days <= 380
            ]
            row = dict(
                symbol=symbol,
                change_available_date=available,
                change_fiscal_date=current_date,
                current_available_date=current["change_available_date"],
                prior_fiscal_date=pd.NaT,
                prior_available_date=pd.NaT,
                pair_status="no_prior_year" if not priors else "ambiguous_prior_year",
                unknown_currency=True,
                unknown_period_length=True,
                unknown_accounting_basis=True,
                unknown_source=True,
                **{feature: np.nan for feature in FEATURES},
            )
            if len(priors) == 1:
                prior = priors[0]
                row.update(
                    prior_fiscal_date=prior["fiscal_date"],
                    prior_available_date=prior["change_available_date"],
                    pair_status="paired",
                    unknown_currency=not all(
                        _known(r["reported_currency"]) for r in [current, prior]
                    ),
                    unknown_period_length=not all(
                        _known(r["period_start"]) for r in [current, prior]
                    ),
                    unknown_accounting_basis=not all(
                        _known(r["accounting_basis"]) for r in [current, prior]
                    ),
                    unknown_source=not all(
                        _known(r["source"]) for r in [current, prior]
                    ),
                )
                conflicts = [
                    col
                    for col in META[1:]
                    if _known(current[col])
                    and _known(prior[col])
                    and current[col] != prior[col]
                ]
                if conflicts:
                    row["pair_status"] = "conflict:" + ",".join(conflicts)
                else:
                    row["net_deleveraging"] = _net_leverage(prior) - _net_leverage(
                        current
                    )
                    if _flow_valid(current) and _flow_valid(prior):
                        row["operating_margin_change"] = _ratio(
                            current["operating_income"], current["revenue"]
                        ) - _ratio(prior["operating_income"], prior["revenue"])
                        row["cash_profitability_change"] = _ratio(
                            current["operating_cash_flow"], current["total_assets"]
                        ) - _ratio(prior["operating_cash_flow"], prior["total_assets"])
                    else:
                        row["pair_status"] = "invalid_flow_duration"
            rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError("No valid financial events before the study cutoff")
    for col in ["prior_fiscal_date", "prior_available_date"]:
        events[col] = pd.to_datetime(events[col]).astype("datetime64[ns]")
    assert events.change_available_date.le(pd.Timestamp(cutoff)).all()
    assert events.current_available_date.le(events.change_available_date).all()
    assert (
        events.prior_available_date.isna()
        | events.prior_available_date.le(events.change_available_date)
    ).all()
    assert not np.isinf(events[list(FEATURES)].to_numpy()).any()
    return events


def join_changes(panel, events):
    """Backward as-of join, 550-day fiscal staleness rule, same-date peer ranks."""
    p = panel.copy()
    p["_original_row"] = np.arange(len(p))
    joined = pd.merge_asof(
        p.sort_values("date"),
        events.sort_values("change_available_date"),
        left_on="date",
        right_on="change_available_date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    joined["change_stale"] = (joined.date - joined.change_fiscal_date).dt.days.gt(550)
    joined.loc[joined.change_stale, list(FEATURES)] = np.nan
    eligible = joined.eligible & joined.risk_screen_pass
    for feature in FEATURES:
        joined[f"rank_{feature}"] = (
            joined[feature].where(eligible).groupby(joined.date).rank(pct=True)
        )
    assert (
        joined.change_available_date.isna()
        | joined.change_available_date.le(joined.date)
    ).all()
    joined = joined.sort_values("_original_row").drop(columns="_original_row")
    joined.index = panel.index
    return joined
