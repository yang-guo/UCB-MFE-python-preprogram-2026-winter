"""
Part 2: Build features and the 90-day target for every signal date.

Design choices
- Signal dates: the last market session of each month. Features are measured
  after that close date. The trade would fill at the next session's close. Monthly
  signals keep the dataset small (~ 2000 stocks x 320 months) and match the
  monthly rebalancing I plan to use in the backtest.
- Target: gross return from the next session entry close compared to the first observed
  close on or after entry + 90 calendar days.

Reused code: 
-- ``load_data``
-- ``price_features``
-- ``financial_events``,
-- ``join_financials``
-- ``apply_membership``
--  ``make_labels`` and the feature lists come from ``Lectures/modeling/course_data.py``. 
My additions are the batching, the monthly sampling, three extra price features, and the peer ranking features.
"""

import time
import warnings
import duckdb
import numpy as np
import pandas as pd
from data import DATABASE, OUTPUT_DIR, experiment, load_membership
from course_data import (FINANCIAL_FEATURES, PRICE_FEATURES, apply_membership, financial_events, 
                         join_financials, load_data, make_labels, price_features)

# My extra price features computed below per stock
EXTRA_PRICE_FEATURES = ["momentum_126", "log_dollar_volume_63", "log_universe_rank"]
NUMERIC_FEATURES = PRICE_FEATURES + EXTRA_PRICE_FEATURES + FINANCIAL_FEATURES + ["filing_age_days"]
# Same-date percentile ranking of each numeric feature (0 = lowest, 1 = highest).
RANK_FEATURES = [f"rank_{name}" for name in NUMERIC_FEATURES]

PANEL_PATH = OUTPUT_DIR / "panel.parquet"
BATCH_SIZE = 400


def month_end_sessions(calendar):
    """Last market session of each calendar month."""
    sessions = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(sessions.groupby(calendar.to_period("M")).max().to_numpy())


def extra_price_features(prices, calendar):
    """
    Extra features

    - momentum_126: 6-month return.
    - log_dollar_volume_63: log of the median (raw close x volume) over the last
      63 sessions used as a measure for liquidity. Not adjusted raw close is used to represent 
      the price actually traded that day.
    """
    frames = []
    # iterate through each ticker
    for symbol, group in prices.groupby("symbol", sort=True):
        # fill in non-trading days
        p = group.set_index("date").reindex(calendar)
        out = pd.DataFrame(index=calendar)
        # 6-month return
        out["momentum_126"] = p.adj_close.pct_change(126, fill_method=None)
        # raw cap and then take log of the median
        dollar_volume = (p.close * p.volume).where(lambda x: x > 0)
        out["log_dollar_volume_63"] = np.log(dollar_volume.rolling(63, min_periods=42).median())
        out["symbol"] = symbol
        out.index.name = "date"
        frames.append(out.loc[p.adj_close.notna()].reset_index())
    return pd.concat(frames, ignore_index=True)


def build_batch(symbols, calendar, signal_dates, membership, end):
    """Features + labels for one batch of stocks and keep only on signal dates."""
    prices, fundamentals, universe, _ = load_data(DATABASE, end=end, symbols=symbols)
    # course helper
    panel = price_features(prices, calendar)
    panel = panel.merge(extra_price_features(prices, calendar), on=["date", "symbol"], how="left", validate="one_to_one")
    # monthly signals only
    panel = panel.loc[panel.date.isin(signal_dates)]
    # filing date
    panel = join_financials(panel, financial_events(fundamentals), universe)
    # quarter of next-session entry
    panel = apply_membership(panel, membership, calendar)
    # size proxy known at selection
    panel["log_universe_rank"] = np.log(panel.universe_rank)
    # only use future prices here
    panel = make_labels(panel, prices, calendar, holding_days=experiment["holding_calendar_days"])
    
    return panel.loc[panel.eligible].reset_index(drop=True)


def add_peer_features(panel):
    """
    Same date percentile ranking among eligible stocks.
    Ranks each feature balanced and are robust to extreme values
    """
    by_date = panel.groupby("date")
    for name in NUMERIC_FEATURES:
        panel[f"rank_{name}"] = by_date[name].rank(pct=True)
    # target relative to the average on the same date.
    panel["target_excess"] = panel.target - by_date.target.transform("mean")
    return panel


def build_panel(end=experiment["test_end"]):
    """Build the full feature panel: all stocks in batches and same date peer ranking."""
    # read the database 
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()
        # get market calendar
        calendar = pd.DatetimeIndex(pd.to_datetime(
            con.execute("SELECT date FROM calendar WHERE date <= ? ORDER BY date", [end]).fetchdf().date
        ).astype("datetime64[ns]"))
    signal_dates = month_end_sessions(calendar)
    # quarterly top 2000 lists. Decides which stocks are eligible to buy.
    membership = load_membership(DATABASE, end=end)

    pieces, quarantine = [], []
    # build the batch
    for start in range(0, len(symbols), BATCH_SIZE):
        t0 = time.time()
        batch = symbols[start:start + BATCH_SIZE]
        # catch warnings from course loader
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pieces.append(build_batch(batch, calendar, signal_dates, membership, end))
        # only report quarantine messages after the loop 
        quarantine += [str(w.message) for w in caught if "Quarantining" in str(w.message)]
        print(f"  stocks {start + 1:>4}-{start + len(batch):<4}: {len(pieces[-1]):>7,} eligible rows "
              f"({time.time() - t0:.0f}s)")
    for message in quarantine:
        print("  course loader:", message.split(";")[0])

    # combine the batches
    panel = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    # safty check to ensure each (date, stock) pair only appear once
    assert not panel.duplicated(["date", "symbol"]).any()
    return add_peer_features(panel), calendar


def check_timing(panel, calendar):
    """Look-ahead checks. Each one raises if a rule is broken. Provides evidence for timing check"""
    # only rows with finished 90-day return
    labeled = panel.loc[panel.target.notna()]
    next_session = pd.Series(calendar[1:], index=calendar[:-1])
    # features use the signal close, so the trade can only fill at next session date
    assert panel.entry_date.eq(panel.date.map(next_session)).all(), "entry must be the next session"
    # only use existing financial data
    assert (panel.available_date.isna() | panel.available_date.le(panel.date)).all(), "financials from the future"
    # top-2000 list must be available
    assert panel.universe_selection_date.le(panel.date).all(), "universe list not yet known"
    # hold at least 90 calendar days
    assert (labeled.label_end >= labeled.entry_date + pd.Timedelta(days=90)).all(), "exit before 90 days"
    # no exit price comes from after the last date 
    assert labeled.label_end.le(experiment["test_end"]).all(), "exit after the data end"
    print("Timing checks passed: next-session entry, filed financials, known universe, >=90-day exits.")


if __name__ == "__main__":
    # start the clock for total runtime
    t0 = time.time()
    # build the full panel
    panel, calendar = build_panel()
    # timing check before next step
    check_timing(panel, calendar)
    # outputs
    OUTPUT_DIR.mkdir(exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    # summarize
    print(f"Saved {len(panel):,} rows x {panel.shape[1]} columns to {PANEL_PATH}  in {time.time() - t0:.0f}s")
