"""HW4 module 1: build the feature/label panel and audit its timing.

Reused code (cited): Lectures/modeling/course_data.py
    load_data, price_features, financial_events, join_financials,
    load_membership, apply_membership, make_labels.
New code in this file: path constants, build_panel (thin wrapper),
and audit_label_timing (an independent re-check of the label rules).

Timing convention (same as the course helpers):
    date        = signal close t (features use data through close t)
    entry_date  = next market session t+1 (buy at that close)
    scheduled_exit = entry_date + 90 calendar days
    label_end   = first quoted close on/after scheduled_exit (≤ 5 sessions late)
    target      = adj_close[label_end] / adj_close[entry_date] - 1
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# --- Paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
MODELING = REPO / "Lectures" / "modeling"
CLASS_DB = MODELING / "data" / "equities_class.duckdb"   # 30 stocks: development
FULL_DB = MODELING / "data" / "equities_top2000.duckdb"  # 3,849 stocks: final run

sys.path.insert(0, str(MODELING))
import course_data as cd  # noqa: E402  (reused lecture code, cited above)

HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5  # market sessions a missing exit quote may slip


def build_panel(database=CLASS_DB, end="2015-12-31", symbols=None):
    """Return (panel, prices, calendar) with features, membership, and labels.

    `end` cuts ALL data in SQL, so no label can use a price after `end`.
    `symbols` (optional) builds one ticker batch; every step here is per ticker,
    so batches concatenate to exactly the unbatched result (bounded memory).
    """
    prices, fundamentals, universe, benchmark = cd.load_data(database, end=end, symbols=symbols)
    calendar = pd.DatetimeIndex(benchmark.date)
    panel = cd.price_features(prices, calendar)
    panel = cd.join_financials(panel, cd.financial_events(fundamentals), universe)
    panel = cd.apply_membership(panel, cd.load_membership(database, end=end), calendar)
    panel = cd.make_labels(panel, prices, calendar, HOLDING_DAYS, MAX_EXIT_DELAY)
    return panel, prices, calendar


def audit_label_timing(panel, prices, calendar):
    """Recompute every label from raw prices; raise on any mismatch.

    Written independently of make_labels so a bug there cannot hide here.
    Returns a small dict of counts for the README.
    """
    cal = pd.DatetimeIndex(calendar)
    p = panel[["symbol", "date", "entry_date", "scheduled_exit",
               "label_end", "entry_price", "exit_price", "target"]]

    # 1. Entry is exactly the next market session after the signal date.
    pos = cal.get_indexer(p.date)
    assert (pos >= 0).all(), "signal date not on calendar"
    has_next = pos + 1 < len(cal)
    expected_entry = pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns]")
    expected_entry[has_next] = cal[pos[has_next] + 1]
    same = (p.entry_date == expected_entry) | (p.entry_date.isna() & expected_entry.isna())
    assert same.all(), "entry_date is not the next session"

    # 2. Scheduled exit is entry + 90 calendar days.
    assert ((p.scheduled_exit - p.entry_date).dropna() == pd.Timedelta(days=HOLDING_DAYS)).all()

    # 3. Labeled rows: exit on/after schedule, within delay, never past data end.
    lab = p.loc[p.target.notna()]
    assert (lab.label_end >= lab.scheduled_exit).all(), "exit before maturity"
    assert (lab.label_end <= cal[-1]).all(), "label uses price after data end"
    delay = cal.get_indexer(lab.label_end) - cal.searchsorted(lab.scheduled_exit)
    assert (delay >= 0).all() and (delay <= MAX_EXIT_DELAY).all()

    # 4. Prices and target match raw adj_close at entry/exit dates.
    px = prices.set_index(["symbol", "date"]).adj_close
    entry_px = px.reindex(pd.MultiIndex.from_arrays([lab.symbol, lab.entry_date])).to_numpy()
    exit_px = px.reindex(pd.MultiIndex.from_arrays([lab.symbol, lab.label_end])).to_numpy()
    assert np.allclose(entry_px, lab.entry_price) and np.allclose(exit_px, lab.exit_price)
    assert np.allclose(exit_px / entry_px - 1, lab.target)

    # 5. Exit is the FIRST quote on/after schedule (no skipped quotes).
    first_exit = []
    for sym, g in lab.groupby("symbol"):
        q = px.loc[sym].index  # sorted quote dates for this stock
        first_exit.append(pd.Series(q[q.searchsorted(g.scheduled_exit)], index=g.index))
    assert pd.concat(first_exit).reindex(lab.index).equals(lab.label_end), "skipped a quote"

    return {
        "rows": len(p),
        "labeled": len(lab),
        "censored_or_missing": int(p.target.isna().sum()),
        "exit_on_schedule_day": int((lab.label_end == lab.scheduled_exit).sum()),
        "exit_rolled_forward": int((lab.label_end > lab.scheduled_exit).sum()),
        "last_label_end": lab.label_end.max().date(),
    }


if __name__ == "__main__":
    panel, prices, calendar = build_panel(CLASS_DB)
    print(audit_label_timing(panel, prices, calendar))
    # One row to verify by hand against the raw prices table.
    row = panel.loc[panel.symbol.eq("AAPL") & panel.date.eq(pd.Timestamp("2012-06-29"))].iloc[0]
    print(row[["date", "entry_date", "scheduled_exit", "label_end",
               "entry_price", "exit_price", "target", "is_universe_member", "eligible"]])
