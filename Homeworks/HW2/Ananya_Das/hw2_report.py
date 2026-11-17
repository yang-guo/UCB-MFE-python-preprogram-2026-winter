"""Risk metrics, data loading and plotting for the HW2 single-stock report (Part 2).

The report reads ``ohlc_hw_cleaned`` (written by ``01_data_cleaning.ipynb``)
through a read-only connection. All metrics are kept as fractions; percentages
are produced only when formatting for display.

Conventions (from the HW2 instructions):

* report returns, volatility, Sharpe and daily statistics use daily **log**
  returns of **adjusted close**;
* drawdown is measured on adjusted close against its running peak;
* undefined metrics (too few returns, zero volatility, empty subsets) are
  ``NaN`` and displayed as ``N/A``.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
MA_WINDOW = 50
TABLE = "ohlc_hw_cleaned"
COLUMNS = ["ts", "open", "high", "low", "close", "adj_close", "volume"]

COLORS = {"high": "#d62728", "low": "#2ca02c", "warmup": "#9e9e9e", "ma": "#1f3b73",
          "line": "#5b2a86", "neutral": "#9e9e9e", "drawdown": "#b2182b"}


# ------------------------------------------------------------------ metrics

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily log returns ``log(p_t / p_{t-1})``.

    The first value is ``NaN``; a missing price makes the returns on both
    sides of it ``NaN`` (nothing is filled).
    """
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate annualized rolling volatility.

    Sample standard deviation (``ddof=1``) of the last ``window`` daily log
    returns times ``sqrt(252)``. A value exists only once a full window of
    returns is available, so the first ``window`` prices have no estimate
    (the warm-up is left as ``NaN``, never filled with zero).
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def calculate_max_drawdown(prices: pd.Series) -> pd.Series:
    """Return per-date drawdown from the running price peak.

    ``(price - running_max) / running_max``: zero at a new high, negative
    below it, and it can recover. The maximum drawdown for the summary is
    ``drawdown.min()``; the running worst drawdown would be ``cummin()``.
    """
    running_max = prices.cummax()
    return (prices - running_max) / running_max


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate the annualized Sharpe ratio (course approximation).

    ``(mean_daily_log_return * 252 - rf) / (std_daily_log_return * sqrt(252))``
    with the sample standard deviation. Returns ``NaN`` with fewer than two
    returns or zero (or non-finite) volatility.
    """
    observed = returns.dropna()
    if len(observed) < 2:
        return np.nan
    sigma = observed.std()
    if not np.isfinite(sigma) or np.isclose(sigma, 0):
        return np.nan
    return float((observed.mean() * TRADING_DAYS - risk_free_rate) / (sigma * np.sqrt(TRADING_DAYS)))


def calculate_annualized_return(returns: pd.Series) -> float:
    """Compounded annualized return ``exp(sum(log_returns) * 252 / n) - 1``; ``NaN`` if ``n == 0``."""
    observed = returns.dropna()
    if observed.empty:
        return np.nan
    return float(np.expm1(observed.sum() * TRADING_DAYS / len(observed)))


def calculate_annualized_volatility(returns: pd.Series) -> float:
    """Whole-period volatility ``std(log_returns) * sqrt(252)``; ``NaN`` with fewer than two returns."""
    observed = returns.dropna()
    if len(observed) < 2:
        return np.nan
    return float(observed.std() * np.sqrt(TRADING_DAYS))


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def summary_statistics(prices: pd.Series, window: int = 20) -> dict:
    """All report metrics for one adjusted-close series, as fractions.

    ``positive_days`` divides positive returns by **all** non-missing returns
    (zero-return days included). Best/worst day and the up/down averages use
    log returns; an empty up or down subset gives ``NaN``.
    """
    returns = calculate_returns(prices)
    observed = returns.dropna()
    volatility = calculate_volatility(returns, window)
    drawdown = calculate_max_drawdown(prices)
    return {
        "total_return": float(prices.iloc[-1] / prices.iloc[0] - 1),
        "annualized_return": calculate_annualized_return(returns),
        "annualized_volatility": calculate_annualized_volatility(returns),
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown": float(drawdown.min()),
        "best_day": float(observed.max()) if len(observed) else np.nan,
        "worst_day": float(observed.min()) if len(observed) else np.nan,
        "positive_days": float((observed > 0).mean()) if len(observed) else np.nan,
        "avg_up_day": _mean_or_nan(observed[observed > 0]),
        "avg_down_day": _mean_or_nan(observed[observed < 0]),
        "avg_daily_return": _mean_or_nan(observed),
        "avg_volatility": float(volatility.mean()) if volatility.notna().any() else np.nan,
        "n_prices": int(prices.notna().sum()),
        "n_returns": int(len(observed)),
    }


SUMMARY_ROWS = [
    ("Total Return", "total_return", "{:+.2%}"),
    ("Annualized Return", "annualized_return", "{:+.2%}"),
    ("Annualized Volatility", "annualized_volatility", "{:.2%}"),
    ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
    ("Maximum Drawdown", "max_drawdown", "{:+.2%}"),
    ("Best Day", "best_day", "{:+.2%}"),
    ("Worst Day", "worst_day", "{:+.2%}"),
    ("Positive Days", "positive_days", "{:.1%}"),
    ("Average Up Day", "avg_up_day", "{:+.2%}"),
    ("Average Down Day", "avg_down_day", "{:+.2%}"),
]


def format_value(value: float, spec: str) -> str:
    """Format a fraction for display; ``NaN``/``None`` becomes ``N/A``."""
    if value is None or not np.isfinite(value):
        return "N/A"
    text = spec.format(value)
    return "+0.00%" if text == "-0.00%" else text


def format_summary_table(stats: dict) -> pd.DataFrame:
    """The ten-row ``Metric | Value`` summary table, formatted for display only."""
    return pd.DataFrame([(label, format_value(stats[key], spec)) for label, key, spec in SUMMARY_ROWS],
                        columns=["Metric", "Value"])


def to_glue_value(value):
    """Plain Python value for Scrapbook; ``NaN`` becomes ``None`` so the notebook JSON stays valid."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ------------------------------------------------------------- data access

def validate_parameters(ticker, start_date, end_date, volatility_window=20) -> tuple[str, str, str]:
    """Normalize and check the report parameters.

    Returns ``(TICKER, 'YYYY-MM-DD', 'YYYY-MM-DD')``. Raises ``ValueError`` for an
    empty ticker, unparseable or reversed dates, or a window below 2.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")
    try:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"dates must be YYYY-MM-DD strings: {start_date!r}, {end_date!r}") from exc
    if start > end:
        raise ValueError(f"start_date {start.date()} is after end_date {end.date()}")
    if isinstance(volatility_window, bool) or not isinstance(volatility_window, int) or volatility_window < 2:
        raise ValueError(f"volatility_window must be an integer >= 2, got {volatility_window!r}")
    return ticker.strip().upper(), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _connect(db_path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path} (copy the course database to data/data.db)")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_report_data(db_path, ticker: str, start_date: str, end_date: str,
                     min_prices: int = 21) -> pd.DataFrame:
    """Load one ticker's cleaned rows with ``start_date <= ts <= end_date`` (inclusive).

    Returns a frame indexed by date. Raises ``LookupError`` if the cleaned table
    or the ticker is missing, and ``ValueError`` if the window holds fewer than
    ``min_prices`` observations (20 returns need 21 prices); the messages give
    the ticker's available history so the cause is clear.
    """
    with closing(_connect(db_path)) as con:
        if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone() is None:
            raise LookupError(f"table {TABLE} not found; run 01_data_cleaning.ipynb first")
        span = con.execute(f"SELECT MIN(ts), MAX(ts), COUNT(*) FROM {TABLE} WHERE ticker = ?", (ticker,)).fetchone()
        if span[2] == 0:
            known = [r[0] for r in con.execute(f"SELECT DISTINCT ticker FROM {TABLE} ORDER BY ticker")]
            raise LookupError(f"ticker {ticker!r} not in {TABLE}; available: {', '.join(known)}")
        frame = pd.read_sql_query(
            f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE ticker = ? AND ts BETWEEN ? AND ? ORDER BY ts",
            con, params=(ticker, start_date, end_date))
    if len(frame) < min_prices:
        raise ValueError(
            f"{ticker}: insufficient history for {start_date} to {end_date}: {len(frame)} prices, "
            f"need at least {min_prices} ({min_prices - 1} returns); the ticker's data covers "
            f"{span[0]} to {span[1]}")
    frame["ts"] = pd.to_datetime(frame["ts"], format="%Y-%m-%d")
    return frame.set_index("ts")


def check_report_data(frame: pd.DataFrame) -> pd.Series:
    """Count rows failing each input check (all zero means the data is usable)."""
    prices = frame[["open", "high", "low", "close", "adj_close"]]
    return pd.Series({
        "missing or non-finite value": int((~np.isfinite(frame[prices.columns.tolist() + ["volume"]])).any(axis=1).sum()),
        "nonpositive price": int((prices <= 0).any(axis=1).sum()),
        "OHLC bounds violated": int(((frame["high"] < frame[["open", "close"]].max(axis=1))
                                     | (frame["low"] > frame[["open", "close"]].min(axis=1))).sum()),
        "negative volume": int((frame["volume"] < 0).sum()),
        "duplicate date": int(frame.index.duplicated(keep=False).sum()),
        "dates not increasing": int((~frame.index.to_series().diff().dropna().gt(pd.Timedelta(0))).sum()),
    }, name="failing rows")


# ---------------------------------------------------------------- plotting

def volatility_colors(volatility: pd.Series, threshold: float) -> np.ndarray:
    """Colour per date: red above ``threshold``, green at or below it, gray where no estimate exists."""
    return np.where(volatility.isna(), COLORS["warmup"],
                    np.where(volatility > threshold, COLORS["high"], COLORS["low"]))


def _date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_risk_report(frame: pd.DataFrame, ticker: str, start_date: str, end_date: str,
                     window: int = 20) -> plt.Figure:
    """Draw the 2x2 risk report (16x12 inches) for one ticker.

    ``frame`` needs a date index and an ``adj_close`` column. Panels: price
    with 50-day MA coloured by rolling volatility, daily log returns, rolling
    volatility against its mean, and drawdown with its maximum marked.
    """
    prices = frame["adj_close"]
    returns = calculate_returns(prices)
    volatility = calculate_volatility(returns, window)
    drawdown = calculate_max_drawdown(prices)
    moving_average = prices.rolling(MA_WINDOW).mean()
    mean_vol = volatility.mean()
    dates = mdates.date2num(prices.index.to_pydatetime())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{ticker} Risk Analysis Report - {start_date} to {end_date}", fontsize=18, fontweight="bold")
    ax_price, ax_ret, ax_vol, ax_dd = axes.ravel()

    # price: each segment takes the colour of its ending date's volatility
    points = np.column_stack([dates, prices.to_numpy()])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    ax_price.add_collection(LineCollection(segments, colors=volatility_colors(volatility, mean_vol)[1:], linewidths=2))
    ax_price.plot(prices.index, moving_average, color=COLORS["ma"], linewidth=1.6, linestyle="--")
    pad =(prices.max() - prices.min()) * 0.05 or prices.max() * 0.05
    ax_price.set_ylim(prices.min() - pad, prices.max() + pad)
    handles = [Line2D([], [], color=COLORS["high"], lw=2, label="Price, volatility above mean"),
               Line2D([], [], color=COLORS["low"], lw=2, label="Price, volatility at/below mean"),
               Line2D([], [], color=COLORS["warmup"], lw=2, label=f"Price, {window}-day warm-up (no estimate)"),
               Line2D([], [], color=COLORS["ma"], lw=1.6, ls="--", label="50-day moving average")]
    ax_price.legend(handles=handles, loc="best", fontsize=9)
    if moving_average.notna().sum() == 0:
        ax_price.text(0.02, 0.04, f"50-day MA unavailable: only {len(prices)} prices in the report",
                      transform=ax_price.transAxes, fontsize=10, color=COLORS["ma"])
    ax_price.set(title=f"{ticker} Price and 50-day MA", xlabel="Date", ylabel="Adjusted close ($)")

    # daily returns
    bar_colors = np.where(returns > 0, COLORS["low"], np.where(returns < 0, COLORS["high"], COLORS["neutral"]))
    ax_ret.bar(prices.index, returns * 100, color=bar_colors, width=1.0)
    ax_ret.axhline(0, color="black", linewidth=0.8)
    ax_ret.set(title=f"{ticker} Daily Returns", xlabel="Date", ylabel="Daily log return (%)")

    # rolling volatility
    vol_pct = volatility * 100
    ax_vol.plot(prices.index, vol_pct, color=COLORS["line"], linewidth=1.6, label=f"{window}-day volatility")
    if np.isfinite(mean_vol):
        ax_vol.axhline(mean_vol * 100, color="black", linestyle="--", linewidth=1,
                       label=f"Mean {mean_vol:.1%}")
        ax_vol.fill_between(prices.index, vol_pct, mean_vol * 100, where=(volatility > mean_vol).to_numpy(),
                            color=COLORS["high"], alpha=0.25, interpolate=True, label="High volatility (> mean)")
        ax_vol.fill_between(prices.index, vol_pct, mean_vol * 100, where=(volatility <= mean_vol).to_numpy(),
                            color=COLORS["low"], alpha=0.25, interpolate=True, label="Low volatility (<= mean)")
    ax_vol.legend(loc="best", fontsize=9)
    ax_vol.set(title=f"{ticker} Rolling Volatility ({window}-day)", xlabel="Date",
               ylabel="Annualized volatility (%)")

    # drawdown: layered transparent fills darken towards the trough
    dd_pct = drawdown * 100
    for level in (0.0, 0.25, 0.5, 0.75):
        ax_dd.fill_between(prices.index, dd_pct * level, dd_pct, color=COLORS["drawdown"], alpha=0.15, linewidth=0)
    ax_dd.plot(prices.index, dd_pct, color=COLORS["drawdown"], linewidth=1.2)
    trough = drawdown.idxmin()
    ax_dd.scatter([trough], [drawdown.min() * 100], color="black", zorder=5, s=40)
    ax_dd.annotate(f"Max drawdown {drawdown.min():.2%}\n{trough:%Y-%m-%d}", (trough, drawdown.min() * 100),
                   xytext=(10, -5), textcoords="offset points", fontsize=10, va="top")
    ax_dd.axhline(0, color="black", linewidth=0.8)
    ax_dd.set_ylim(min(drawdown.min() * 100 * 1.25, -1), 1)
    ax_dd.set(title=f"{ticker} Maximum Drawdown", xlabel="Date", ylabel="Drawdown from running peak (%)")

    for ax in axes.ravel():
        ax.set_xlim(dates.min() - 1, dates.max() + 1)   # same span everywhere, so the warm-up gap stays visible
        ax.grid(True, alpha=0.3)
        _date_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
