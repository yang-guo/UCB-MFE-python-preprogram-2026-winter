"""Conservative price-quality flags; thresholds are review rules, not facts.

All detection uses only the current and preceding observations of a symbol.
No prices are interpolated, rescaled, or replaced by future observations.
"""

import numpy as np
import pandas as pd


ZERO_VOLUME_SCALE = 100.0
REVIEW_SCALE = 10.0


def screen_prices(prices):
    """Quarantine zero-volume scale conflicts and flag unresolved transitions.

    A zero-volume quote at least 100-fold away from the latest preceding
    positive-volume quote is not trusted. Other >=10-fold changes between
    retained observed prices are flagged for review, not declared errors.
    These conservative choices were made using development data only.
    """
    frame = prices.sort_values(["symbol", "date"]).copy()
    reasons = pd.Series("", index=frame.index, dtype="object")
    valid_price = np.isfinite(frame.adj_close) & frame.adj_close.gt(0)
    positive_volume_prices = frame.adj_close.where(valid_price & frame.volume.gt(0))
    reference_prices = positive_volume_prices.groupby(frame.symbol).ffill().groupby(frame.symbol).shift()
    ratio = frame.adj_close / reference_prices
    zero_conflict = frame.volume.eq(0) & (ratio.ge(ZERO_VOLUME_SCALE) | ratio.le(1 / ZERO_VOLUME_SCALE))
    reasons.loc[zero_conflict] = "zero_volume_scale_conflict"
    reasons.loc[~valid_price] = "invalid_adjusted_price"

    keep = reasons.eq("")
    clean = frame.loc[keep].copy()
    previous = clean.groupby("symbol").adj_close.shift()
    scale = clean.adj_close / previous
    review = scale.ge(REVIEW_SCALE) | scale.le(1 / REVIEW_SCALE)
    reasons.loc[clean.index[review]] = "unresolved_price_scale_transition"
    reference_prices.loc[clean.index[review]] = previous.loc[review]
    audit = frame.loc[reasons.ne(""), ["symbol", "date", "close", "adj_close", "volume"]].copy()
    audit["reason"] = reasons.loc[audit.index]
    audit["reference_adj_close"] = reference_prices.loc[audit.index]
    audit["quarantined_quote"] = audit.reason.ne("unresolved_price_scale_transition")
    return clean, audit


def apply_quality_flags(panel, audit, calendar):
    """Keep diagnosis, censor affected labels, and guard historical features.

    The entry mask uses trailing information only. Label censoring is an
    outcome-quality check and must never be used as a future entry filter.
    Original labels remain in target_before_quality for accounting/audit.
    """
    result = panel.copy()
    result["price_quality_lookback_flag"] = False
    result["label_quality_flag"] = False
    result["target_before_quality"] = result.target
    for symbol, group in result.groupby("symbol", sort=False):
        dates = pd.DatetimeIndex(audit.loc[audit.symbol.eq(symbol), "date"].unique()).sort_values()
        if dates.empty:
            continue
        indicator = pd.Series(calendar.isin(dates), index=calendar)
        # 253 prices cover the longest 252-return feature window.
        contaminated = indicator.rolling(253, min_periods=1).max().astype(bool)
        result.loc[group.index, "price_quality_lookback_flag"] = group.date.map(contaminated).to_numpy()
        valid = group.entry_date.notna() & group.label_end.notna()
        rows = group.loc[valid]
        left = dates.searchsorted(pd.DatetimeIndex(rows.entry_date), side="left")
        right = dates.searchsorted(pd.DatetimeIndex(rows.label_end), side="right")
        result.loc[rows.index, "label_quality_flag"] = right > left
    result["eligible"] &= ~result.price_quality_lookback_flag
    result.loc[result.label_quality_flag, "target"] = np.nan
    return result
