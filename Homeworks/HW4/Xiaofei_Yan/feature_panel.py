"""Build an availability-aware HW4 feature panel in bounded symbol batches.

This module adapts the course helpers in ``Lectures/modeling/course_data.py``.
The adapted course functions are imported for price features, universe
membership, and 90-calendar-day labels; the additional HW4 features are built
here. The course source files and source DuckDB database are read-only.
"""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from price_quality import screen_prices, apply_quality_flags
from common_stock_screen import load_security_review, POLICY_VERSION


COURSE_ROOT = Path(__file__).resolve().parents[3]
MODELING_DIR = COURSE_ROOT / "Lectures" / "modeling"
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from course_data import (  # noqa: E402
    apply_membership,
    financial_events as course_financial_events,
    load_membership,
    make_labels,
    price_features,
)


DEFAULT_DATABASE = MODELING_DIR / "data" / "equities_top2000.duckdb"
DEFAULT_END_DATE = "2026-09-30"
DEFAULT_SIGNAL_START = "2010-01-01"
FUNDAMENTAL_FEATURES = [
    "operating_roa",
    "gross_margin",
    "revenue_yoy",
    "asset_growth_yoy",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "current_ratio",
    "filing_age_days",
]
PRICE_VOLUME_FEATURES = [
    "return_5",
    "momentum_21",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "relative_momentum_63",
    "beta_252",
    "range_21",
    "volume_ratio_21_252",
    "log_dollar_volume_21",
]
FEATURES = PRICE_VOLUME_FEATURES + FUNDAMENTAL_FEATURES


def load_reference_data(database=DEFAULT_DATABASE, end=DEFAULT_END_DATE):
    """Load the shared calendar, dated membership, and symbol list read-only."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Place the HW4 database at {database}")

    with duckdb.connect(str(database), read_only=True) as connection:
        benchmark = connection.execute(
            "SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date", [end]
        ).fetchdf()
        symbols = connection.execute(
            "SELECT DISTINCT symbol FROM universe_membership "
            "WHERE quarter_start <= ? ORDER BY symbol",
            [end],
        ).fetchdf()["symbol"].tolist()

    benchmark["date"] = pd.to_datetime(benchmark["date"])
    calendar = pd.DatetimeIndex(benchmark["date"])
    membership = load_membership(database=database, end=end)
    review = load_security_review(database)
    allowed = set(review.loc[review.common_stock_candidate, "symbol"])
    symbols = [symbol for symbol in symbols if symbol in allowed]
    return calendar, benchmark, membership, symbols


def load_symbol_batch(database, symbols, end=DEFAULT_END_DATE):
    """Read one symbol batch, retaining pre-2010 history for lookback windows."""
    database = Path(database)
    symbols = list(symbols)
    if not symbols:
        raise ValueError("symbols cannot be empty")

    with duckdb.connect(str(database), read_only=True) as connection:
        prices = connection.execute(
            """SELECT symbol, date, open, high, low, close, adj_close, volume
               FROM prices
               WHERE date <= ? AND symbol IN (SELECT UNNEST(?))
               ORDER BY symbol, date""",
            [end, symbols],
        ).fetchdf()
        fundamentals = connection.execute(
            """SELECT symbol, fiscal_date, period, filing_date,
                      revenue, gross_profit, operating_income, net_income,
                      operating_cash_flow, free_cash_flow, total_assets,
                      total_debt, cash_and_equivalents,
                      total_current_assets, total_current_liabilities
               FROM fundamentals
               WHERE fiscal_date <= ?
                 AND (filing_date <= ? OR filing_date IS NULL)
                 AND symbol IN (SELECT UNNEST(?))
               ORDER BY symbol, fiscal_date""",
            [end, end, symbols],
        ).fetchdf()
        universe = connection.execute(
            """SELECT symbol, sector FROM universe
               WHERE symbol IN (SELECT UNNEST(?)) ORDER BY symbol""",
            [symbols],
        ).fetchdf()

    prices["date"] = pd.to_datetime(prices["date"])
    fundamentals["fiscal_date"] = pd.to_datetime(fundamentals["fiscal_date"])
    fundamentals["filing_date"] = pd.to_datetime(fundamentals["filing_date"])
    return prices, fundamentals, universe


def _add_price_volume_features(prices, calendar, benchmark):
    """Add the selected causal price/volume features to the lecture panel."""
    base = price_features(prices, calendar)
    benchmark_close = benchmark.set_index("date")["adj_close"].reindex(calendar)
    benchmark_return = benchmark_close.pct_change(fill_method=None)
    benchmark_momentum_63 = benchmark_close.pct_change(63, fill_method=None)
    market_variance = benchmark_return.rolling(252, min_periods=252).var()

    additions = []
    for symbol, group in prices.groupby("symbol", sort=True):
        quotes = group.set_index("date").reindex(calendar)
        adjusted_close = quotes["adj_close"]
        daily_return = adjusted_close.pct_change(fill_method=None)

        # Use the supplied trading-session calendar; never fill quote gaps.
        quotes["return_5"] = adjusted_close.pct_change(5, fill_method=None)
        quotes["relative_momentum_63"] = (
            adjusted_close.pct_change(63, fill_method=None) - benchmark_momentum_63
        )
        rolling_covariance = daily_return.rolling(252, min_periods=252).cov(benchmark_return)
        quotes["beta_252"] = rolling_covariance / market_variance.where(market_variance.ne(0))

        valid_ohlc = (
            quotes["high"].gt(0)
            & quotes["low"].gt(0)
            & quotes["close"].gt(0)
            & quotes["high"].ge(quotes["low"])
        )
        daily_range = ((quotes["high"] - quotes["low"]) / quotes["close"]).where(valid_ohlc)
        quotes["range_21"] = daily_range.rolling(21, min_periods=21).mean()

        volume = quotes["volume"].astype(float)
        mean_volume_21 = volume.rolling(21, min_periods=21).mean()
        mean_volume_252 = volume.rolling(252, min_periods=252).mean()
        quotes["volume_ratio_21_252"] = mean_volume_21 / mean_volume_252.where(mean_volume_252.gt(0))

        dollar_volume = (quotes["close"] * volume).where(quotes["close"].gt(0) & volume.ge(0))
        mean_dollar_volume_21 = dollar_volume.rolling(21, min_periods=21).mean()
        quotes["log_dollar_volume_21"] = np.log(mean_dollar_volume_21.where(mean_dollar_volume_21.gt(0)))

        quotes["symbol"] = symbol
        quotes.index.name = "date"
        additions.append(
            quotes.loc[
                adjusted_close.notna(),
                ["symbol", *PRICE_VOLUME_FEATURES[:1], "relative_momentum_63", "beta_252",
                 "range_21", "volume_ratio_21_252", "log_dollar_volume_21"],
            ].reset_index()
        )

    extra = pd.concat(additions, ignore_index=True)
    return base.merge(extra, on=["symbol", "date"], how="left", validate="one_to_one")


def _financial_events(fundamentals):
    """Extend the course's availability-dated quarterly events with HW4 fields.

    Same-fiscal-period growth uses the matching record from one year earlier,
    and only when that prior record was also available by the current filing.
    """
    # Reuse the course implementation for its six established ratio features
    # and its filing-lag validation. The extra fields below are HW4 additions.
    course_events = course_financial_events(fundamentals)
    f = fundamentals.loc[
        fundamentals["period"].eq("quarter")
        & fundamentals["filing_date"].gt(fundamentals["fiscal_date"])
    ].copy()
    if f.empty:
        for feature in ["gross_margin", "revenue_yoy", "asset_growth_yoy", "current_ratio"]:
            course_events[feature] = np.nan
        return course_events

    f["available_date"] = f["filing_date"] + pd.Timedelta(days=1)
    f["prior_fiscal_date"] = f["fiscal_date"] - pd.DateOffset(years=1)
    prior = f[["symbol", "fiscal_date", "filing_date", "revenue", "total_assets"]].copy()
    prior = prior.rename(
        columns={
            "fiscal_date": "prior_fiscal_date",
            "filing_date": "prior_filing_date",
            "revenue": "prior_revenue",
            "total_assets": "prior_assets",
        }
    )
    f = f.merge(prior, on=["symbol", "prior_fiscal_date"], how="left", validate="many_to_one")

    prior_is_available = (
        f["prior_filing_date"].notna()
        & (f["prior_filing_date"] + pd.Timedelta(days=1) <= f["available_date"])
    )
    valid_revenue_pair = prior_is_available & f["revenue"].gt(0) & f["prior_revenue"].gt(0)
    valid_asset_pair = prior_is_available & f["total_assets"].gt(0) & f["prior_assets"].gt(0)
    f["revenue_yoy"] = (f["revenue"] / f["prior_revenue"] - 1).where(valid_revenue_pair)
    f["asset_growth_yoy"] = (f["total_assets"] / f["prior_assets"] - 1).where(valid_asset_pair)

    revenue = f["revenue"].where(f["revenue"].gt(0))
    f["gross_margin"] = f["gross_profit"] / revenue
    f["current_ratio"] = f["total_current_assets"] / f["total_current_liabilities"].where(
        f["total_current_liabilities"].gt(0)
    )
    extra_features = ["gross_margin", "revenue_yoy", "asset_growth_yoy", "current_ratio"]
    f[extra_features] = f[extra_features].replace([np.inf, -np.inf], np.nan)

    # Keep the newest fiscal period known at each availability date, matching
    # the course helper's protection against late-filed older statements.
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    f["latest_fiscal"] = f.groupby("symbol")["fiscal_date"].cummax()
    f = f.loc[f["fiscal_date"].eq(f["latest_fiscal"])]
    f = f.drop_duplicates(["symbol", "available_date"], keep="last")
    extra = f[["symbol", "fiscal_date", "filing_date", "available_date", *extra_features]]
    return course_events.merge(
        extra,
        on=["symbol", "fiscal_date", "filing_date", "available_date"],
        how="left",
        validate="one_to_one",
    )


def _join_financials(panel, events, universe):
    """Backward as-of join financial events and preserve stale-data gaps."""
    frames = []
    ratio_features = [name for name in FUNDAMENTAL_FEATURES if name != "filing_age_days"]
    for symbol, group in panel.groupby("symbol", sort=True):
        right = events.loc[events["symbol"].eq(symbol)].drop(columns="symbol").sort_values("available_date")
        joined = pd.merge_asof(
            group.sort_values("date"),
            right,
            left_on="date",
            right_on="available_date",
            direction="backward",
        )
        joined["filing_age_days"] = (joined["date"] - joined["available_date"]).dt.days
        stale = (joined["date"] - joined["fiscal_date"]).dt.days.gt(550)
        joined.loc[stale, ratio_features] = np.nan
        frames.append(joined)

    result = pd.concat(frames, ignore_index=True).merge(
        universe, on="symbol", how="left", validate="many_to_one"
    )
    result["sector"] = result["sector"].fillna("Unknown")
    if not (result["available_date"].isna() | result["available_date"].le(result["date"])).all():
        raise ValueError("A financial feature is being used before its availability date")
    return result


def build_feature_batch(
    database,
    symbols,
    calendar,
    benchmark,
    membership,
    end=DEFAULT_END_DATE,
    signal_start=DEFAULT_SIGNAL_START,
    audit_path=None,
):
    """Build features and labels for one symbol batch; source reads are read-only."""
    prices, fundamentals, universe = load_symbol_batch(database, symbols, end=end)
    prices = prices.loc[prices["date"].isin(calendar)].copy()
    prices, audit = screen_prices(prices)
    if audit_path is not None:
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)

    panel = _add_price_volume_features(prices, calendar, benchmark)
    panel = _join_financials(panel, _financial_events(fundamentals), universe)
    panel = apply_membership(panel, membership, calendar)
    panel = panel.loc[panel["date"].ge(pd.Timestamp(signal_start))].copy()
    panel = make_labels(panel, prices[["symbol", "date", "adj_close"]], calendar)
    panel = apply_quality_flags(panel, audit, calendar)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def iter_feature_batches(
    database=DEFAULT_DATABASE,
    end=DEFAULT_END_DATE,
    signal_start=DEFAULT_SIGNAL_START,
    batch_size=100,
    audit_dir=Path(__file__).parent / "outputs" / "data_quality" / POLICY_VERSION,
):
    """Yield feature panels by symbol batch to keep full-universe memory bounded."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    database = Path(database)
    calendar, benchmark, membership, symbols = load_reference_data(database, end=end)
    for start in range(0, len(symbols), batch_size):
        batch_symbols = symbols[start : start + batch_size]
        yield build_feature_batch(
            database=database,
            symbols=batch_symbols,
            calendar=calendar,
            benchmark=benchmark,
            membership=membership,
            end=end,
            signal_start=signal_start,
            audit_path=(Path(audit_dir) / f"quality_{end}_{start // batch_size:03d}.csv") if audit_dir is not None else None,
        )
