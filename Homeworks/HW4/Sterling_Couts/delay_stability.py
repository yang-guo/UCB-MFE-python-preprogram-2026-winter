"""Calendar and chronology checks for annual financial-delay diagnostics.

Annual accounts start from fresh cash and mark remaining lots at year-end.
They must not be spliced into an investable continuous portfolio.
"""

import numpy as np
import pandas as pd


def scores_for_entry_year(scores, calendar, year):
    """Slice by the next-session ENTRY year, not the signal or label-end year.

    A December signal can initiate a January purchase. A late-year signal with
    an unknown future outcome must remain a candidate for that year's account.
    """
    if year not in [2013, 2014, 2015]:
        raise ValueError(
            "This declared review permits only development years 2013–2015."
        )
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise ValueError("Market calendar must be sorted and unique.")
    locations = calendar.get_indexer(scores.date)
    if (locations < 0).any():
        raise ValueError("A signal is absent from the market calendar.")
    valid = locations + 1 < len(calendar)
    entries = pd.Series(pd.NaT, index=scores.index, dtype="datetime64[ns]")
    entries.loc[valid] = calendar[locations[valid] + 1].to_numpy()
    return scores.loc[entries.dt.year.eq(year)].reset_index(drop=True)


def check_fit_chronology(audit, scores):
    """Fail if a fit or preprocessing step can use an immature target."""
    assert len(audit) == 1 and len(scores) > 0
    row = audit.iloc[0]
    assert row.first_training_signal >= pd.Timestamp("2010-01-01")
    assert row.first_training_signal <= row.last_training_signal < row.fit_date
    assert row.last_training_label < row.fit_date <= scores.date.min()
    assert scores.fit_date.eq(row.fit_date).all()
    assert scores.date.max() <= pd.Timestamp("2015-12-31")
    assert row.alpha == 1000


def check_trade_targets(fills, scores, calendar):
    """Independently match actual buys and closed lots to their forecast horizon."""
    buys = fills.loc[fills.side.eq("BUY")].copy()
    if buys.empty:
        return dict(checked_buys=0, matched_closed_targets=0)
    calendar = pd.DatetimeIndex(calendar)
    locations = calendar.get_indexer(buys.signal_date)
    assert (locations >= 0).all() and (locations + 1 < len(calendar)).all()
    np.testing.assert_array_equal(
        calendar[locations + 1].to_numpy(), buys.date.to_numpy()
    )
    assert (buys.maturity - buys.date).dt.days.eq(90).all()
    joined = buys.merge(
        scores.rename(columns={"date": "signal_date"}),
        on=["signal_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    assert joined.score.notna().all()
    assert joined.fit_date.le(joined.signal_date).all()
    sells = fills.loc[fills.side.eq("SELL"), ["lot_id", "date", "price"]]
    closed = joined.merge(
        sells, on="lot_id", suffixes=("_entry", "_exit"), validate="one_to_one"
    )
    assert closed.label_end.eq(closed.date_exit).all()
    np.testing.assert_allclose(
        closed.price_exit / closed.price_entry - 1, closed.target, rtol=0, atol=1e-12
    )
    return dict(checked_buys=len(buys), matched_closed_targets=len(closed))
