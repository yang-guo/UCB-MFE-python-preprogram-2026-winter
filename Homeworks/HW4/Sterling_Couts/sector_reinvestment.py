"""Causal daily eligibility for carrying monthly rankings between signal dates.

Scores are not recomputed daily. We recheck the existing risk/history screens
and the entry quarter's supplied membership before spending sale proceeds.
No future target, later survival or future quote availability enters this mask.
"""

import duckdb
import numpy as np
import pandas as pd

from hw4.data import stock_features


def eligibility_at_entry(daily_screen, membership):
    """Shift screens one session; use entry-quarter membership known by then."""
    calendar = daily_screen.index
    result = daily_screen.shift(1, fill_value=False).astype(bool)
    previous = pd.Series(calendar, index=calendar).shift(1)
    quarter = calendar.to_period("Q").start_time
    for start in quarter.unique():
        days = calendar[quarter == start]
        selected = membership.loc[membership.quarter_start.eq(start)]
        if selected.symbol.duplicated().any():
            raise ValueError("Duplicate membership within an entry quarter.")
        known = selected.set_index("symbol").selection_date.reindex(result.columns)
        available = known.to_numpy()[None, :] <= previous.loc[days].to_numpy()[:, None]
        result.loc[days] &= available
    return result


def build_entry_eligibility(database, quotes, panel):
    """Rebuild original screens from bounded raw inputs and reviewed price units.

    The monthly rows must reproduce the saved D2 candidate screen. That check
    guards against accidentally changing the original strategy while extending
    its eligibility checks to nonmonthly trading dates.
    """
    calendar = quotes.index
    if calendar.max() > pd.Timestamp("2015-12-31"):
        raise ValueError("This experiment is restricted to development.")
    screen = pd.DataFrame(False, index=calendar, columns=quotes.columns)
    with duckdb.connect(str(database), read_only=True) as con:
        membership = con.execute(
            "SELECT symbol,quarter_start,selection_date FROM universe_membership "
            "WHERE quarter_start<=?",
            [calendar[-1]],
        ).df()
        for column in ["quarter_start", "selection_date"]:
            membership[column] = pd.to_datetime(membership[column]).astype(
                "datetime64[ns]"
            )
        symbols = quotes.columns.tolist()
        for offset in range(0, len(symbols), 64):
            raw = con.execute(
                "SELECT symbol,date,close,volume FROM prices WHERE date BETWEEN ? AND ? "
                "AND symbol IN (SELECT unnest(?)) ORDER BY symbol,date",
                [calendar[0], calendar[-1], symbols[offset : offset + 64]],
            ).df()
            raw["date"] = pd.to_datetime(raw.date).astype("datetime64[ns]")
            raw = raw.loc[raw.date.isin(calendar)]
            for symbol, group in raw.groupby("symbol", sort=False):
                group = group.copy()
                group["adj_close"] = quotes[symbol].reindex(group.date).to_numpy()
                feature = stock_features(group, calendar)
                screen[symbol] = feature.eligible_history & feature.risk_screen_pass
    # Same bounded issuer-alias convention as the D2 monthly model panel.
    if "LAB" in screen:
        screen.loc[screen.index >= "2011-02-10", "LAB"] = False
    eligible = eligibility_at_entry(screen, membership)
    rows = calendar.get_indexer(pd.DatetimeIndex(panel.entry_date))
    columns = quotes.columns.get_indexer(panel.symbol)
    if (rows < 0).any() or (columns < 0).any():
        raise ValueError("Monthly eligibility reference is outside the daily grid.")
    np.testing.assert_array_equal(
        eligible.to_numpy()[rows, columns],
        (panel.eligible & panel.risk_screen_pass).to_numpy(),
    )
    return eligible
