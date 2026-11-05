"""Lecture 2 aggregation and Lecture 4 risk calculations, exposed as functions."""

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prices(config, ticker):
    """Read a fixed snapshot. Never substitute raw close for adjusted close."""
    start, end = config["start"], config["as_of"]
    path = Path(config["data_path"])
    if config["dataset"] == "fixture":
        all_prices = pd.read_csv(path)
        in_window = all_prices.date.between(start, end)
        calendar = sorted(all_prices.loc[in_window, "date"].unique())
        prices = all_prices.loc[in_window & all_prices.ticker.eq(ticker)].copy()
    else:
        # The table name is fixed in code. Ticker and dates are bound values.
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
            prices = pd.read_sql_query(
                "SELECT ts AS date, close, adj_close FROM ohlc "
                "WHERE ticker=? AND ts>=? AND ts<=? ORDER BY ts",
                connection, params=(ticker, start, end),
            )
            calendar = [row[0] for row in connection.execute(
                "SELECT DISTINCT ts FROM ohlc WHERE ts>=? AND ts<=? ORDER BY ts",
                (start, end),
            )]
    if prices.empty:
        raise ValueError(f"No prices for {ticker} in the requested window")
    prices["date"] = pd.to_datetime(prices.date, errors="raise")
    prices = prices.sort_values("date").reset_index(drop=True)
    if prices.date.duplicated().any():
        raise ValueError(f"Duplicate dates for {ticker}")
    for column in ["close", "adj_close"]:
        prices[column] = pd.to_numeric(prices[column], errors="raise")
        if not (np.isfinite(prices[column]) & prices[column].gt(0)).all():
            raise ValueError(f"Invalid {column} for {ticker}; inspect the source")
    expected = pd.DatetimeIndex(pd.to_datetime(calendar))
    internal = expected[(expected >= prices.date.min()) & (expected <= prices.date.max())]
    if len(internal.difference(prices.date)):
        raise ValueError(f"Internal calendar gaps for {ticker}; no filling is allowed")
    if len(prices) < 21:
        raise ValueError(f"{ticker} needs at least 21 prices for 20-return volatility")
    return prices


def risk_metrics(prices):
    """Keep fractions in storage and use the Lecture 4 log-return convention."""
    adjusted = prices.adj_close
    returns = np.log(adjusted / adjusted.shift(1)).dropna()
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    rolling = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    return {
        "first_date": prices.date.iloc[0].date().isoformat(),
        "last_date": prices.date.iloc[-1].date().isoformat(),
        "observations": len(prices),
        "total_return": float(adjusted.iloc[-1] / adjusted.iloc[0] - 1),
        "annualized_log_volatility": volatility,
        "mean_rolling_20_log_volatility": float(rolling.mean()),
        "max_drawdown": float((adjusted / adjusted.cummax() - 1).min()),
        "approximate_sharpe_rf_2pct": (
            float((252 * returns.mean() - 0.02) / volatility)
            if volatility > 1e-12 else None
        ),
        "last_raw_close": float(prices.close.iloc[-1]),
        "conventions": "Daily log returns; sample std; 252 sessions; 2% annual risk-free approximation",
    }


def portfolio_exposure(holdings_path, prices_by_ticker, cash):
    """Aggregate lots before calculating weights; actual shares use raw close."""
    holdings = pd.read_csv(holdings_path)
    required = {"ticker", "shares", "entry_price"}
    if not required.issubset(holdings.columns) or holdings.empty:
        raise ValueError("Holdings require ticker, shares, and entry_price")
    for column in ["shares", "entry_price"]:
        if not (np.isfinite(holdings[column]) & holdings[column].gt(0)).all():
            raise ValueError(f"Invalid holdings {column}")
    tickers = set(holdings.ticker)
    if not tickers.issubset(prices_by_ticker):
        raise ValueError("Every held ticker needs validated prices")
    dates = {prices_by_ticker[t]["last_date"] for t in tickers}
    if len(dates) != 1:
        raise ValueError("Portfolio marks must share one valuation date")
    holdings["cost"] = holdings.shares * holdings.entry_price
    grouped = holdings.groupby("ticker")[["shares", "cost"]].sum()
    grouped["weighted_entry_price"] = grouped.cost / grouped.shares
    grouped["raw_close"] = [prices_by_ticker[t]["last_raw_close"] for t in grouped.index]
    grouped["market_value"] = grouped.shares * grouped.raw_close
    total = float(grouped.market_value.sum() + cash)
    grouped["weight"] = grouped.market_value / total
    grouped["unrealized_pnl"] = grouped.market_value - grouped.cost
    return {"valuation_date": dates.pop(), "total_value": total, "cash": cash,
            "cash_weight": cash / total, "positions": grouped.reset_index().to_dict("records"),
            "assumption": "Illustrative actual-share holdings at the valuation date; no historical rebalance"}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
