"""Bounded data overlays and causal entry rules for development comparisons.

The original core is unchanged. Unit corrections use documented split terms
or independent historical price anchors, never a factor chosen for performance.
Financial reference repairs are not inserted into these price-only experiments.
"""

import json

import duckdb
import numpy as np
import pandas as pd

from hw4.data import ResearchData, holding_labels, stock_features
from hw4.settings import DATABASE, PROJECT

CUTOFF = pd.Timestamp("2015-12-31")


def normalize_prices(frame, registry):
    """Copy one stock's history and apply each reviewed, bounded unit conversion."""
    result = frame.copy()
    if result.date.max() > pd.Timestamp(registry["scope_end"]):
        raise ValueError(
            "Price repairs are restricted to the reviewed development period."
        )
    original = frame.set_index("date").adj_close
    symbol = frame.symbol.iloc[0]
    if not frame.symbol.eq(symbol).all():
        raise ValueError("Normalize one stock at a time.")
    rules = [r for r in registry["repairs"] if r["symbol"] == symbol]
    for rule in rules:
        for date, expected in [
            (rule["previous_session"], rule["expected_previous"]),
            (rule["boundary"], rule["expected_boundary"]),
        ]:
            if not np.isclose(
                original.loc[pd.Timestamp(date)], expected, rtol=1e-10, atol=1e-6
            ):
                raise ValueError(
                    "Source price differs from the reviewed repair evidence."
                )
        if rule["side"] == "before":
            mask = result.date.lt(rule["boundary"])
        elif rule["side"] == "on_or_after":
            mask = result.date.ge(rule["boundary"])
        else:
            raise ValueError("Unknown price-repair interval.")
        mask &= result.date.ge(registry["scope_start"])
        result.loc[mask, "adj_close"] *= rule["factor"]
    return result


def corrected_data(original):
    """Recompute affected features and targets; retain supplied dated membership."""
    if original.quotes.index.max() > CUTOFF:
        raise ValueError("Development only: later prices are not permitted.")
    registry = json.loads(
        (PROJECT / "data_review/development_price_repairs.json").read_text()
    )
    panel, quotes = original.panel.copy(), original.quotes.copy()
    changes = []
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        for symbol in sorted({r["symbol"] for r in registry["repairs"]}):
            raw = con.execute(
                "SELECT symbol,date,close,adj_close,volume FROM prices "
                "WHERE symbol=? AND date BETWEEN ? AND ? ORDER BY date",
                [symbol, registry["scope_start"], registry["scope_end"]],
            ).df()
            raw["date"] = pd.to_datetime(raw.date).astype("datetime64[ns]")
            raw = raw.loc[raw.date.isin(quotes.index)]
            revised = normalize_prices(raw, registry)
            revised_series = revised.set_index("date").adj_close.reindex(quotes.index)
            quotes[symbol] = revised_series
            indexes = panel.index[panel.symbol.eq(symbol)]
            dates = panel.loc[indexes, "date"]
            features = stock_features(revised, quotes.index).reindex(dates)
            labels = holding_labels(dates, revised_series, quotes.index)
            for col in features:
                panel.loc[indexes, col] = features[col].to_numpy()
            for col in ("entry_date", "scheduled_exit", "label_end", "target"):
                panel.loc[indexes, col] = labels[col].to_numpy()
            panel.loc[indexes, "eligible"] = (
                panel.loc[indexes, "eligible_history"] & panel.loc[indexes, "is_member"]
            )
            for rule in [r for r in registry["repairs"] if r["symbol"] == symbol]:
                old = raw.set_index("date").adj_close
                before, at = (
                    pd.Timestamp(rule["previous_session"]),
                    pd.Timestamp(rule["boundary"]),
                )
                changes.append(
                    {
                        "repair": rule["id"],
                        "symbol": symbol,
                        "date": at,
                        "original_daily_return": old.loc[at] / old.loc[before] - 1,
                        "corrected_daily_return": revised_series.loc[at]
                        / revised_series.loc[before]
                        - 1,
                    }
                )
    anchor_checks = []
    for date, expected in registry["lci_anchors"].items():
        observed = quotes.loc[pd.Timestamp(date), "LCI"]
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-6)
        anchor_checks.append(
            {"date": date, "reference": expected, "corrected": observed}
        )
    changed = sorted({r["symbol"] for r in registry["repairs"]})
    unaffected = ~original.panel.symbol.isin(changed)
    pd.testing.assert_frame_equal(panel.loc[unaffected], original.panel.loc[unaffected])
    pd.testing.assert_frame_equal(
        quotes.drop(columns=changed), original.quotes.drop(columns=changed)
    )
    return (
        ResearchData(quotes, original.benchmark.copy(), panel, dict(original.audit)),
        changes,
        anchor_checks,
    )


def remove_fluidigm_alias(data):
    """Use historical FLDM as the fixed source for its documented LAB alias.

    This is a declared source-priority policy, not a choice of the better return.
    Prices remain for audit. Neither current active status nor later survival
    determines training or entries. No other same-name pair is merged.
    """
    panel = data.panel.copy()
    scope = panel.date.between("2011-02-10", "2015-12-31")
    remove = scope & panel.symbol.eq("LAB")
    count = int((remove & panel.eligible & panel.risk_screen_pass).sum())
    panel.loc[remove, "eligible"] = False
    return ResearchData(data.quotes, data.benchmark, panel, dict(data.audit)), count


def select_entries(scores, panel, rule):
    """Change only new-entry candidates; do not filter on targets or later prices."""
    if rule == "unchanged":
        return scores.copy()
    features = panel[["date", "symbol", "momentum_63", "volatility_63"]]
    joined = scores.merge(features, on=["date", "symbol"], validate="one_to_one")
    if rule == "positive_63d_trend":
        keep = joined.momentum_63.gt(0)
    elif rule == "volatility_40pct":
        keep = joined.volatility_63.le(0.40)
    else:
        raise ValueError("Unknown entry rule.")
    return joined.loc[keep, scores.columns].copy()
