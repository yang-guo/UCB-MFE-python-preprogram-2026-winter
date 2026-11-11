"""Load the HW4 database in ticker batches and build the monthly research panel."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from backtest import HOLDING_DAYS


# Appended, not inserted, so this folder's own modules keep priority.
MODEL_DIR = Path(__file__).resolve().parents[3] / "Lectures" / "modeling"
sys.path.append(str(MODEL_DIR))
import course_data as course  # reuse the course_data module from the Lectures/modeling folder


DATABASE = MODEL_DIR / "data" / "equities_top2000.duckdb"
PRICE_EVENTS = Path(__file__).with_name("price_events.csv")
BATCH_SIZE = 150

# Given setup.
DEV_START = "2010-01-01"
DEV_END = "2015-12-31"
TEST_START = "2016-04-01"
TEST_END = "2026-09-30"

# Manually chosen parameters, fixed on development data.
MAX_SESSION_MOVE = 4.0
EVENT_TOLERANCE = 0.25
MIN_DOLLAR_VOLUME = 5_000_000
FEATURES = course.PRICE_FEATURES + course.FINANCIAL_FEATURES
LABELS = ["entry_date", "label_end", "target"]


def monthly_signals(calendar):
    """Use the close before the first market session of each month."""
    months = calendar.to_period("M")
    return calendar[:-1][months[:-1] != months[1:]]


def load_price_events(path=PRICE_EVENTS):
    """Read the reviewed price events. Only documented splits and confirmed real moves change the data."""
    return pd.read_csv(path, parse_dates=["date"])


def apply_splits(prices, events):
    """Rescale prices before each documented split that the vendor left out of its history.

    The factor must explain the vendor's jump on that date to within
    EVENT_TOLERANCE, otherwise the event table is wrong and the load stops.
    """
    prices = prices.astype({"volume": float})
    splits = events.loc[events.type.eq("split") & events.symbol.isin(prices.symbol.unique())]
    for event in splits.itertuples():
        stock = prices.loc[prices.symbol.eq(event.symbol)]
        earlier = stock.loc[stock.date.lt(event.date)]
        on_day = stock.loc[stock.date.eq(event.date)]
        if earlier.empty or on_day.empty:
            continue
        jump = on_day.adj_close.iloc[0] / earlier.adj_close.iloc[-1]
        if abs(jump / event.factor - 1) > EVENT_TOLERANCE:
            raise ValueError(f"{event.symbol} moves {jump:.4g}x on {event.date:%Y-%m-%d}, not {event.factor:.4g}x.")
        prices.loc[earlier.index, ["close", "adj_close"]] *= event.factor
        prices.loc[earlier.index, "volume"] /= event.factor
    return prices


def quarantine_breaks(prices, events):
    """Drop a stock's quotes from its first unexplained move between observed closes.

    A close more than MAX_SESSION_MOVE times the previous one, or below that
    fraction of it, is treated as a break in the price basis unless the event
    table confirms it as a real move. Each row is judged only by quotes up to
    its own date. Also return one row per quarantined stock with its break
    date and the vendor's last quote.
    """
    ratio = prices.groupby("symbol").adj_close.pct_change(fill_method=None) + 1
    keys = pd.MultiIndex.from_frame(prices[["symbol", "date"]])
    real = pd.MultiIndex.from_frame(events.loc[events.type.eq("real"), ["symbol", "date"]])
    broken = (ratio.gt(MAX_SESSION_MOVE) | ratio.lt(1 / MAX_SESSION_MOVE)) & ~keys.isin(real)
    dropped = prices.loc[broken.groupby(prices.symbol).cummax()]
    breaks = dropped.groupby("symbol").agg(break_date=("date", "min"), vendor_last_date=("date", "max"),
                                           vendor_last_close=("adj_close", "last")).reset_index()
    return prices.drop(dropped.index), breaks


def _dollar_volume(prices, calendar):
    # Close and volume share one split basis, so their product is a proxy for dollars traded.
    traded = (prices.close * prices.volume).set_axis(pd.MultiIndex.from_frame(prices[["symbol", "date"]]))
    traded = traded.unstack("symbol").reindex(calendar)
    return traded.rolling(63, min_periods=63).median().stack().rename("dollar_volume_63")


def _batch_panel(prices, fundamentals, universe, membership, calendar, signals):
    # Trailing features come first, then the dated universe for the next-session entry.
    features = course.price_features(prices, calendar)
    features = features.loc[features.date.ge(DEV_START) & features.date.isin(signals)]
    features = features.join(_dollar_volume(prices, calendar), on=["date", "symbol"])
    features["eligible"] = features.eligible & features.dollar_volume_63.ge(MIN_DOLLAR_VOLUME)
    features = course.apply_membership(features, membership, calendar)
    features = features.loc[features.eligible]
    if features.empty:
        return None
    features = course.join_financials(features, course.financial_events(fundamentals), universe)
    # A late exit is still the first quoted close, so the label matches the backtester's sale.
    labeled = course.make_labels(features, prices, calendar, holding_days=HOLDING_DAYS,
                                 max_exit_delay=len(calendar))
    return labeled[["date", "symbol", *FEATURES, *LABELS]].replace([np.inf, -np.inf], np.nan)


def load_research_data(database, end, quotes_start, symbols=None, events=None):
    """Return the monthly panel, benchmark closes, daily quotes from quotes_start, and price breaks.

    Nothing after `end` is read. The panel keeps every eligible stock on each
    signal date. Rows whose 90-day outcome is not observed keep a missing target.
    By default every stock that was ever a universe member is loaded.
    """
    events = load_price_events() if events is None else events
    membership = course.load_membership(database, end=end)
    symbols = sorted(membership.symbol.unique()) if symbols is None else sorted(symbols)
    market, panels, quotes, breaks = None, [], [], []
    for offset in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[offset:offset + BATCH_SIZE]
        prices, fundamentals, universe, benchmark = course.load_data(database, end=end, symbols=batch)
        if market is None:
            market = benchmark.set_index("date").adj_close
            if market.index[-1] != pd.Timestamp(end):
                raise ValueError(f"Database benchmark must cover the requested end date {end}.")
            signals = monthly_signals(market.index)
        if prices.empty:
            continue
        prices, batch_breaks = quarantine_breaks(apply_splits(prices, events), events)
        breaks.append(batch_breaks)
        panel = _batch_panel(prices, fundamentals, universe,
                             membership.loc[membership.symbol.isin(batch)], market.index, signals)
        if panel is not None:
            panels.append(panel)
        # Grab the quotes from the prices table: dates as rows, stocks as columns.
        recent = prices.loc[prices.date.ge(quotes_start)]
        quotes.append(recent.pivot(index="date", columns="symbol", values="adj_close"))
    if not panels:
        raise ValueError("No eligible monthly signals in this database.")
    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    # After the batches, line every stock up on the trading calendar.
    quotes = pd.concat(quotes, axis=1).reindex(market.loc[quotes_start:].index)
    return panel, market, quotes, pd.concat(breaks, ignore_index=True)
