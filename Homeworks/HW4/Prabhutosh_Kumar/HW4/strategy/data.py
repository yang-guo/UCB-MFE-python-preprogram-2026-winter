"""Bounded-memory feature building; the course's causal data helpers are reused.

Only monthly decision rows are retained, but every quote is kept for accounting.
Add a causal price feature to extra_features and list its name in config.json.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "Lectures/modeling"))
import course_data as course  # noqa: E402


def extra_features(panel):
    """No future rows, price filling, or learned parameters in these features."""
    panel["reversal_5"] = -panel.adj_close.pct_change(5, fill_method=None)
    panel["volatility_21"] = panel.adj_close.pct_change(fill_method=None).rolling(21).std() * np.sqrt(252)
    return panel


def build_dataset(config):
    database = ROOT / config["database"]
    with duckdb.connect(str(database), read_only=True) as con:
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").df().symbol.tolist()
        benchmark = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN '2008-01-01' AND ? ORDER BY date", [config["end"]]).df()
        ranking_audit = con.execute("""SELECT quarter_start,symbol,selection_close,selection_shares,market_cap_proxy
            FROM universe_membership WHERE selection_close>100000 OR selection_shares>10000000000
            ORDER BY quarter_start,symbol"""
        ).df()
    benchmark.date = pd.to_datetime(benchmark.date).astype("datetime64[ns]")
    calendar = pd.DatetimeIndex(benchmark.date)
    # Signal on the session BEFORE the month's first entry. This includes Mar 31, 2016.
    first = np.r_[True, calendar.to_period("M")[1:] != calendar.to_period("M")[:-1]]
    entries = np.flatnonzero(first)
    signals = calendar[entries[entries > 0] - 1]
    membership = course.load_membership(database, config["end"])
    pieces, quotes, extremes = [], [], []
    for offset in range(0, len(symbols), 100):
        batch = symbols[offset:offset + 100]
        prices, fundamentals, universe, _ = course.load_data(database, config["end"], batch)
        prices = prices.loc[prices.date.ge(calendar[0])].copy()
        wide = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
        quotes.append(wide)
        returns = wide.pct_change(fill_method=None)
        unusual = returns.where(returns.abs().gt(0.5)).stack().rename("daily_return").reset_index()
        extremes.append(unusual)
        panel = course.price_features(prices, calendar)
        extras = []
        for symbol in wide:
            p = extra_features(pd.DataFrame({"adj_close": wide[symbol]}))
            p["symbol"] = symbol
            p.index.name = "date"
            extras.append(p.loc[p.index.isin(signals), ["symbol", "reversal_5", "volatility_21"]].reset_index())
        panel = panel.loc[panel.date.isin(signals) & panel.date.ge("2009-12-01")]
        panel = panel.merge(pd.concat(extras), on=["date", "symbol"], validate="one_to_one")
        panel = course.apply_membership(panel, membership, calendar)
        panel = panel.loc[panel.eligible].copy()
        if not panel.empty:
            panel = course.join_financials(panel, course.financial_events(fundamentals), universe)
            panel = course.make_labels(panel, prices, calendar, config["holding_days"])
            pieces.append(panel)
        print(f"Prepared {min(offset + 100, len(symbols))}/{len(symbols)} securities", flush=True)
    return (pd.concat(pieces, ignore_index=True), pd.concat(quotes, axis=1),
            benchmark.set_index("date").adj_close, pd.concat(extremes, ignore_index=True), ranking_audit)
