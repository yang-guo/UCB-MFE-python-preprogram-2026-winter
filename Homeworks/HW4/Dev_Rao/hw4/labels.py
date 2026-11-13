"""90-calendar-day forward-return labels.

Adapted from `Lectures/modeling/course_data.py::make_labels` (same rule, written
for one reindexed stock series so it can run inside the symbol-batch loop):

* signal after the close on t, entry at the NEXT market session's close;
  the entry session must have a quote, otherwise no label;
* exit at the first observed close on/after entry + 90 calendar days, allowing
  at most 5 market sessions of delay past the scheduled session, otherwise the
  label is missing (censored, NOT a filter);
* label_end = actual exit date. Training at fit date F may only use rows with
  label_end < F.
"""

import numpy as np
import pandas as pd

from . import config


def add_bench_forward(panel, bench):
    """Benchmark return over each row's own entry->exit window (NaN if no label).
    Used only to build the beta-residual TRAINING target; it is realised at the
    same moment as the label, so the label_end < fit_date purge covers it."""
    b = bench.astype("float64")
    out = panel.copy()
    out["bench_fwd"] = (out.label_end.map(b) / out.entry_date.map(b) - 1).where(out.target.notna())
    return out


def labels_for_symbol(px, calendar, signal_dates, holding_days=config.HOLDING_DAYS,
                      max_exit_delay=config.MAX_EXIT_DELAY):
    """px: adjusted close reindexed to `calendar` (NaN = no quote). Returns a frame
    indexed like signal_dates with entry/exit dates, prices and the raw target."""
    calendar = pd.DatetimeIndex(calendar)
    values = px.to_numpy()
    observed = ~np.isnan(values)
    quote_dates = calendar[observed]
    quote_values = values[observed]

    sig_pos = calendar.get_indexer(signal_dates)
    assert (sig_pos >= 0).all(), "signal dates must be market sessions"
    entry_pos = sig_pos + 1
    has_next = entry_pos < len(calendar)
    n = len(signal_dates)
    entry_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    entry_price = np.full(n, np.nan)
    entry_date[has_next] = calendar[entry_pos[has_next]].to_numpy()
    entry_price[has_next] = values[entry_pos[has_next]]

    scheduled = pd.DatetimeIndex(entry_date) + pd.Timedelta(days=holding_days)
    ok = ~np.isnan(entry_price) & (scheduled <= calendar[-1])
    exit_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_price = np.full(n, np.nan)
    if ok.any():
        due = scheduled[ok]
        ix = quote_dates.searchsorted(due)                    # first quote on/after due
        has_quote = ix < len(quote_dates)
        rows = np.flatnonzero(ok)[has_quote]
        exits = quote_dates[ix[has_quote]]
        delay = calendar.get_indexer(exits) - calendar.searchsorted(due[has_quote])
        keep = delay <= max_exit_delay
        exit_date[rows[keep]] = exits[keep].to_numpy()
        exit_price[rows[keep]] = quote_values[ix[has_quote][keep]]

    out = pd.DataFrame({"entry_date": entry_date, "entry_price": entry_price,
                        "label_end": exit_date, "exit_price": exit_price},
                       index=pd.DatetimeIndex(signal_dates, name="date"))
    out["target"] = out.exit_price / out.entry_price - 1
    # Timing evidence: entry strictly after signal; exit >= entry + 90 calendar days.
    has = out.target.notna()
    assert (out.loc[has, "entry_date"] > out.index[has]).all()
    assert (out.loc[has, "label_end"] >= out.loc[has, "entry_date"] + pd.Timedelta(days=holding_days)).all()
    return out
