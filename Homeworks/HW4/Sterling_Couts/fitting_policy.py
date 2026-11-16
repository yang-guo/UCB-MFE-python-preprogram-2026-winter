"""Combine return forecasts from two causally fitted model vintages.

This is a new research hypothesis beyond the lecture's single-model refits.
Each component has its own training-fitted preprocessing. Average predictions
in common 90-day return units, not raw coefficients on different training scales.
"""

import numpy as np


def combine_forecasts(anchor, latest, latest_weight=0.5):
    """Blend matching predictions while preserving censored future outcomes.

    Date/symbol are the identity, not row order. Fit provenance is retained so
    no component trained after a signal can enter that signal's blend. Outcomes
    are consistency metadata only: they never determine weights or eligibility.
    """
    if not np.isfinite(latest_weight) or not 0 <= latest_weight <= 1:
        raise ValueError("Latest-model weight must be finite and between zero and one.")
    keys = ["date", "symbol"]
    required = keys + ["target", "label_end", "score", "fit_date"]
    for frame in [anchor, latest]:
        if frame.empty or any(c not in frame for c in required):
            raise ValueError(
                "Both forecast tables must be nonempty and contain all required columns."
            )
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Forecast identities must be complete and unique.")
        if not np.isfinite(frame.score).all() or frame.fit_date.isna().any():
            raise ValueError("Scores must be finite and every fit date known.")
        if not frame.fit_date.le(frame.date).all():
            raise ValueError("A component model was fitted after its prediction date.")
    joined = anchor[required].merge(
        latest[required],
        on=keys,
        how="outer",
        suffixes=("_anchor", "_latest"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Components must cover exactly the same candidates.")
    if not joined.fit_date_latest.ge(joined.fit_date_anchor).all():
        raise ValueError("The latest fit cannot predate the anchor fit.")
    # CSV round trips can move an outcome by floating-point roundoff.
    same_target = np.isclose(
        joined.target_anchor, joined.target_latest, rtol=0, atol=1e-12, equal_nan=True
    )
    same_end = joined.label_end_anchor.eq(joined.label_end_latest) | (
        joined.label_end_anchor.isna() & joined.label_end_latest.isna()
    )
    if not same_target.all() or not same_end.all():
        raise ValueError("Component forecasts must refer to the same future outcome.")
    result = joined[keys].copy()
    result["target"] = joined.target_anchor
    result["label_end"] = joined.label_end_anchor
    result["score"] = (
        1 - latest_weight
    ) * joined.score_anchor + latest_weight * joined.score_latest
    result["fit_date"] = joined.fit_date_latest
    result["anchor_fit_date"] = joined.fit_date_anchor
    result["latest_fit_date"] = joined.fit_date_latest
    result["latest_weight"] = latest_weight
    assert np.isfinite(result.score).all()
    return result.sort_values(keys).reset_index(drop=True)
