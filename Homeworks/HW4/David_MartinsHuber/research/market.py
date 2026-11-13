"""Causal market series used by the HMM research.

The benchmark loader is reused verbatim from ``data_loading.py``.  The
benchmark table stores only ``date`` and ``adj_close``, so the S&P 500 volume
proxy is built from the point-in-time universe membership:

``benchmark``
    ^SP500TR total-return index and the sum of volume for every name in the
    membership snapshot that is effective on that date.

``cap_weighted``
    A top-2000 market-cap-weighted price index and a cap-weighted volume
    series.  This is the documented fallback for the case where a genuine
    S&P 500 volume series is unavailable.

Both definitions are causal: a snapshot becomes usable on its
``effective_start`` and its weights come from ``market_cap_proxy``, which is
fixed at the selection date.  No future snapshot or price is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketSeries:
    """A daily price level and its volume proxy."""

    name: str
    price: pd.Series
    volume: pd.Series
    notes: str = ""

    def aligned(self) -> tuple[pd.Series, pd.Series]:
        """Return price and volume on their common, sorted dates."""
        frame = pd.concat(
            [self.price.rename("price"), self.volume.rename("volume")], axis=1
        ).dropna()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if frame.empty:
            raise ValueError(f"{self.name}: no overlapping price and volume dates")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"{self.name}: price and volume must be finite")
        if (frame.price <= 0).any() or (frame.volume < 0).any():
            raise ValueError(f"{self.name}: prices must be positive and volume nonnegative")
        return frame.price, frame.volume


def load_benchmark(database: str | Path, end: str) -> pd.Series:
    """Load the benchmark through the verbatim course loader.

    ``symbols`` is restricted to one ticker so the loader returns the full
    benchmark without materializing the entire price panel.
    """
    from data_loading import load_data

    _, _, _, benchmark = load_data(database, end=end, symbols=["AAPL"])
    price = benchmark.set_index("date")["adj_close"].astype(float).sort_index()
    price.index = pd.to_datetime(price.index)
    if price.empty or not (np.isfinite(price) & price.gt(0)).all():
        raise ValueError("Benchmark prices must be finite and positive")
    return price.rename("price")


def summed_universe_volume(database: str | Path, end: str) -> pd.Series:
    """Daily volume summed over the effective point-in-time universe.

    A date uses the latest membership snapshot whose ``effective_start`` is
    not after that date.  This is the S&P 500 volume proxy requested for the
    research: total traded volume of the members, not a single-name volume.
    """
    query = """
        WITH snaps AS (
            SELECT quarter_start, effective_start
            FROM universe_membership
            GROUP BY 1, 2
        )
        SELECT p.date AS date, SUM(p.volume) AS volume, COUNT(*) AS names
        FROM prices p
        JOIN benchmark calendar ON calendar.date = p.date
        ASOF JOIN snaps s ON p.date >= s.effective_start
        JOIN universe_membership m
          ON m.quarter_start = s.quarter_start AND m.symbol = p.symbol
        WHERE p.date <= ? AND isfinite(p.adj_close) AND p.adj_close > 0
          AND p.volume IS NOT NULL AND p.volume >= 0
        GROUP BY p.date
        ORDER BY p.date
    """
    with duckdb.connect(str(database), read_only=True) as con:
        frame = con.execute(query, [end]).fetchdf()
    if frame.empty:
        raise ValueError("No universe volume rows were returned")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    if frame["volume"].isna().any() or (frame["volume"] < 0).any():
        raise ValueError("Universe volume must be nonnegative and nonmissing")
    return frame["volume"].astype(float).rename("volume")


def cap_weighted_market(database: str | Path, end: str) -> pd.DataFrame:
    """Build the top-2000 cap-weighted index return and volume proxy.

    Weights are ``market_cap_proxy`` normalized within each quarterly
    snapshot, winsorized at the cross-sectional 99th percentile because the
    source has a few extreme market-cap outliers (for example, share counts
    that look split-adjusted).  A symbol contributes only when it has a
    previous close, so the index does not jump on listings.  Daily moves
    larger than 50% in absolute value are treated as source outliers and
    excluded from the weighted return, then the remaining weights are
    renormalized.  The returned ``volume`` is the cap-weighted average daily
    volume (weights sum to one), matching the requested "weight each asset's
    volume by its market-cap weight" rule.
    """
    query = """
        WITH snaps AS (
            SELECT quarter_start, effective_start
            FROM universe_membership
            GROUP BY 1, 2
        ),
        caps AS (
            SELECT quarter_start,
                   quantile_cont(market_cap_proxy, 0.99) AS cap99
            FROM universe_membership
            GROUP BY quarter_start
        ),
        weights AS (
            SELECT m.quarter_start, m.symbol,
                   LEAST(m.market_cap_proxy, c.cap99)
                   / SUM(LEAST(m.market_cap_proxy, c.cap99))
                         OVER (PARTITION BY m.quarter_start) AS weight
            FROM universe_membership m
            JOIN caps c USING (quarter_start)
        ),
        sessions AS (
            SELECT date, LAG(date) OVER (ORDER BY date) AS previous_session
            FROM benchmark
        ),
        valid_prices AS (
            SELECT p.*, c.previous_session
            FROM prices p JOIN sessions c ON p.date = c.date
            WHERE p.date <= ? AND isfinite(p.adj_close) AND p.adj_close > 0
              AND p.volume IS NOT NULL AND p.volume >= 0
        ),
        rets AS (
            SELECT symbol, date, adj_close, volume, previous_session,
                   LAG(date) OVER (PARTITION BY symbol ORDER BY date) AS previous_quote,
                   adj_close / LAG(adj_close) OVER (
                       PARTITION BY symbol ORDER BY date
                   ) - 1 AS raw_ret
            FROM valid_prices
        ),
        joined AS (
            SELECT r.date,
                   CASE
                       WHEN r.previous_quote = r.previous_session
                            AND isfinite(r.raw_ret) AND abs(r.raw_ret) <= 0.5
                       THEN r.raw_ret
                   END AS ret,
                   r.volume,
                   w.weight
            FROM rets r
            ASOF JOIN snaps s ON r.date >= s.effective_start
            JOIN weights w
              ON w.quarter_start = s.quarter_start AND w.symbol = r.symbol
        )
        SELECT date,
               SUM(weight * ret)
               / NULLIF(SUM(CASE WHEN ret IS NOT NULL THEN weight ELSE 0 END), 0)
                   AS index_return,
               SUM(weight * volume) AS volume,
               COUNT(*) AS name_count,
               SUM(CASE WHEN ret IS NULL THEN 1 ELSE 0 END) AS filtered,
               MAX(weight) AS max_weight
        FROM joined
        GROUP BY date
        ORDER BY date
    """
    with duckdb.connect(str(database), read_only=True) as con:
        frame = con.execute(query, [end]).fetchdf()
    if frame.empty:
        raise ValueError("No cap-weighted market rows were returned")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame["index_return"] = frame["index_return"].astype(float)
    # Anchor the index before any return exists; do not fabricate zero returns
    # for later sessions with no usable constituent return observations.
    if pd.isna(frame["index_return"].iloc[0]):
        frame.iloc[0, frame.columns.get_loc("index_return")] = 0.0
    if not np.isfinite(frame[["index_return", "volume"]].to_numpy()).all():
        raise ValueError("Cap-weighted index return and volume must be finite")
    frame["price"] = (1.0 + frame["index_return"]).cumprod()
    if frame["price"].le(0).any():
        raise ValueError("Cap-weighted index level must stay positive")
    return frame


def load_market_cap_frame(database: str | Path, end: str) -> pd.DataFrame:
    """Load point-in-time membership market caps without changing the loader.

    ``load_membership`` is deliberately kept verbatim and does not select
    ``market_cap_proxy``. The HMM cap-weighted strategy needs that column, so
    this read-only helper queries the same table separately.
    """
    query = """
        SELECT symbol, quarter_start, selection_date, market_cap_proxy
        FROM universe_membership
        WHERE quarter_start <= ? AND selection_date <= ?
        ORDER BY quarter_start, selection_date, symbol
    """
    with duckdb.connect(str(database), read_only=True) as con:
        frame = con.execute(query, [end, end]).fetchdf()
    if frame.empty:
        raise ValueError("No membership market-cap rows were returned")
    for column in ("quarter_start", "selection_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame["market_cap_proxy"] = pd.to_numeric(
        frame["market_cap_proxy"], errors="coerce"
    )
    if frame[["symbol", "quarter_start", "selection_date"]].isna().any().any():
        raise ValueError("Membership market-cap keys must be nonmissing")
    return frame


def build_market_series(
    database: str | Path, end: str
) -> dict[str, MarketSeries]:
    """Return the benchmark and cap-weighted market definitions."""
    database = Path(database)
    benchmark_price = load_benchmark(database, end)
    benchmark_volume = summed_universe_volume(database, end)
    cap = cap_weighted_market(database, end)

    benchmark = MarketSeries(
        "benchmark",
        benchmark_price,
        benchmark_volume,
        notes=(
            "^SP500TR total-return price; volume is the sum of the effective "
            "point-in-time universe members."
        ),
    )
    cap_weighted = MarketSeries(
        "cap_weighted",
        cap["price"].rename("price"),
        cap["volume"].rename("volume"),
        notes=(
            "Top-2000 cap-weighted price index and cap-weighted average "
            "volume; the fallback when a genuine S&P 500 volume series is "
            "unavailable."
        ),
    )
    return {"benchmark": benchmark, "cap_weighted": cap_weighted}
