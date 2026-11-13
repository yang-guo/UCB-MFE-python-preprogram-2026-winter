"""Load prices and build model inputs. See hw4.ipynb for the course sources."""
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Lectures" / "modeling"))
from course_data import (  # noqa: E402
    PRICE_FEATURES, apply_membership, load_data, load_membership,
    make_labels, price_features,
)

FEATURES = PRICE_FEATURES + ["relative_momentum_63", "momentum_acceleration"]


def database_path():
    for path in [ROOT / "Lectures/modeling/data/equities_top2000.duckdb",
                 ROOT / "Homeworks/HW4/data/equities_top2000.duckdb"]:
        if path.is_file():
            return path
    raise FileNotFoundError("Download equities_top2000.duckdb from bCourses; see README.md.")


def market_data(database, end):
    with duckdb.connect(str(database), read_only=True) as con:
        frame = con.execute("SELECT date,adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
    frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
    if frame.date.duplicated().any() or not (np.isfinite(frame.adj_close) & frame.adj_close.gt(0)).all():
        raise ValueError("The index has repeated dates or invalid prices.")
    return frame.set_index("date").adj_close


def prepare_panel(database, end, config, batch_size=100):
    """Build monthly stock rows, even when future returns are unknown."""
    benchmark = market_data(database, end)
    calendar = benchmark.index
    entry_positions = calendar.get_indexer(calendar.to_series().groupby(calendar.to_period("M")).first())
    signal_dates = calendar[entry_positions[entry_positions > 0] - 1]
    membership = load_membership(database, end=end)
    symbols = sorted(membership.symbol.unique())
    pieces = []
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        prices, _, _, _ = load_data(database, end=end, symbols=batch)
        if prices.empty:
            continue
        frame = price_features(prices, calendar)
        # Use raw close times volume to measure trading activity.
        liquidity = []
        for symbol, group in prices.groupby("symbol", sort=False):
            daily = group.set_index("date").reindex(calendar)
            dollar_volume = (daily.close * daily.volume).where(daily.close.gt(0) & daily.volume.gt(0))
            liquidity.append(pd.DataFrame({
                "date": calendar, "symbol": symbol,
                "dollar_volume_21": dollar_volume.rolling(21, min_periods=21).mean().to_numpy(),
            }))
        frame = frame.loc[frame.date.ge("2010-01-01") & frame.date.isin(signal_dates)].copy()
        if frame.empty:
            continue
        frame = frame.merge(pd.concat(liquidity), on=["date", "symbol"], validate="one_to_one")
        frame["relative_momentum_63"] = frame.momentum_63 - frame.date.map(benchmark.pct_change(63, fill_method=None))
        frame["momentum_acceleration"] = frame.momentum_21 - frame.momentum_63 / 3
        frame = apply_membership(frame, membership.loc[membership.symbol.isin(batch)], calendar)
        frame = frame.loc[
            frame.eligible & frame.close.ge(config["minimum_price"])
            & frame.dollar_volume_21.ge(config["minimum_dollar_volume_21"])
            & frame.volatility_63.le(config["maximum_volatility_63"])
        ].copy()
        if frame.empty:
            continue
        frame = make_labels(frame, prices, calendar, holding_days=90, max_exit_delay=0)
        pieces.append(frame[["date", "symbol", *FEATURES, "entry_date", "scheduled_exit",
                             "label_end", "entry_price", "exit_price", "target",
                             "dollar_volume_21", "universe_rank", "universe_selection_date"]])
        if start % 500 == 0:
            print(f"Prepared {min(start + batch_size, len(symbols))}/{len(symbols)} stocks through {end}", flush=True)
    if not pieces:
        raise ValueError("No stocks passed the filters.")
    panel = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    assert not panel.duplicated(["date", "symbol"]).any()
    known = panel.target.notna()
    assert panel.loc[known, "label_end"].ge(panel.loc[known, "scheduled_exit"]).all()
    assert panel.entry_date.gt(panel.date).all()
    assert panel.universe_selection_date.le(panel.date).all()
    assert np.allclose(panel.loc[known, "target"], panel.loc[known, "exit_price"] / panel.loc[known, "entry_price"] - 1)
    return panel


def load_quotes(database, symbols, start, end):
    """Keep prices for stocks we own, even after they leave the stock list."""
    calendar = market_data(database, end).loc[start:end].index
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute("""SELECT date,symbol,adj_close FROM prices
            WHERE date BETWEEN ? AND ? AND symbol IN (SELECT unnest(?))
            AND isfinite(adj_close) AND adj_close>0 ORDER BY date,symbol""",
            [start, end, list(symbols)]).fetchdf()
    prices["date"] = pd.to_datetime(prices.date).astype("datetime64[ns]")
    prices = prices.loc[prices.date.isin(calendar)]
    if prices.duplicated(["date", "symbol"]).any():
        raise ValueError("A stock has two prices on the same date.")
    quotes = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    # One replacement stock traded under a temporary ticker absent from the database.
    deals = pd.read_csv(Path(__file__).with_name("corporate_actions.csv"))
    for deal in deals.loc[deals.new_close.notna()].itertuples():
        date = pd.Timestamp(deal.available_date)
        if date in quotes.index and deal.new_symbol in symbols:
            if deal.new_symbol not in quotes:
                quotes[deal.new_symbol] = np.nan
            if pd.isna(quotes.loc[date, deal.new_symbol]):
                quotes.loc[date, deal.new_symbol] = deal.new_close
    return quotes


def load_actions(database, end):
    """Read verified deals and convert cash or shares to adjusted-price units."""
    path = Path(__file__).with_name("corporate_actions.csv")
    deals = pd.read_csv(path, parse_dates=["effective_date", "available_date"])
    deals = deals.loc[deals.available_date.le(pd.Timestamp(end))].copy()
    actions = []
    with duckdb.connect(str(database), read_only=True) as con:
        for deal in deals.to_dict("records"):
            if deal["available_date"] < deal["effective_date"]:
                raise ValueError("A merger cannot be applied before it happens.")
            quote = con.execute("""SELECT date,close,adj_close FROM prices
                WHERE symbol=? AND date<=? AND close>0 AND adj_close>0
                ORDER BY date DESC LIMIT 1""", [deal["symbol"], deal["effective_date"]]).fetchone()
            if quote is None:
                raise ValueError(f"No price basis for {deal['symbol']}")
            old_factor = quote[2] / quote[1]
            if deal.get("kind", "cash") == "cash":
                deal["kind"] = "cash"
                deal["cash_price"] = deal["cash_per_share"] * old_factor
            else:
                new_quote = con.execute("""SELECT close,adj_close FROM prices
                    WHERE symbol=? AND date=(SELECT MIN(date) FROM calendar WHERE date>=?)""",
                    [deal["new_symbol"], deal["available_date"]]).fetchone()
                if new_quote is None and pd.notna(deal.get("new_close")):
                    new_quote = (deal["new_close"], deal["new_close"])
                if new_quote is None or min(new_quote) <= 0:
                    raise ValueError(f"No new-stock price basis for {deal['symbol']}")
                deal["unit_ratio"] = old_factor * deal["share_ratio"] / (new_quote[1] / new_quote[0])
                deal["cash_per_unit"] = deal["cash_per_share"] * old_factor if deal["kind"] == "mixed" else 0.
            deal["cvr_per_unit"] = old_factor if pd.notna(deal.get("cvr_max")) else 0.
            deal["basis_date"] = pd.Timestamp(quote[0])
            actions.append(deal)
    return actions



def save_ranking_flags(database, destination):
    with duckdb.connect(str(database), read_only=True) as con:
        flags = con.execute("""SELECT quarter_start,symbol,selection_close,selection_shares
            FROM universe_membership WHERE selection_close>1000000
            OR selection_shares>40000000000 ORDER BY quarter_start,symbol""").fetchdf()
    flags.to_csv(destination, index=False)
