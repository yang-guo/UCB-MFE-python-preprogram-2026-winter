"""HW4: Build a Stock Prediction and Trading Strategy
see README.md for details
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
import warnings

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# directory 
sys.path.insert(0, str(Path(__file__).resolve().parent))

import course_data as cd  # noqa: E402
from backtest import (  # noqa: E402
    BacktestAnalytics,
    BacktestResult,
    OrderManagementSystem,
    capped_budgets,
)

#paths
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL_DB = REPO_ROOT / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
DEFAULT_CLASS_DB = REPO_ROOT / "Lectures" / "Lecture 8" / "equities_class.duckdb"
LOT_FIELDS = [
    "lot_id",
    "symbol",
    "units",
    "signal_date",
    "entry_date",
    "entry_price",
    "entry_fee",
    "due_date",
]

#price feature list
RAW_EXTRA_PRICE_FEATURES = [
    "momentum_10",
    "momentum_5_10_5",
    "momentum_5_21_5",
    "momentum_5_63_5",
    "momentum_5_252_5",
    "volatility_10",
    "volatility_21",
    "relative_volume_10",
    "relative_volume_21",
    "relative_volume_63",
    "amihud_illiquidity_10",
    "amihud_illiquidity_21",
    "amihud_illiquidity_63",
    "return_volume_corr_10",
    "return_volume_corr_21",
    "return_volume_corr_63",
    "turnover_10",
    "turnover_21",
    "turnover_63",
]
SECTOR_NEUTRAL_BASE_FEATURES = cd.NUMERIC_FEATURES + RAW_EXTRA_PRICE_FEATURES
SECTOR_RELATIVE_FEATURES = [f"sector_relative_{feature}" for feature in SECTOR_NEUTRAL_BASE_FEATURES]
EXTRA_PRICE_FEATURES = RAW_EXTRA_PRICE_FEATURES + SECTOR_RELATIVE_FEATURES + ["sector_profitability_rank"]
NUMERIC_FEATURES = cd.NUMERIC_FEATURES + EXTRA_PRICE_FEATURES
FEATURES = NUMERIC_FEATURES + ["sector"]


@dataclass
class Config:
    database: Path
    output_dir: Path
    train_start: str = "2010-01-01"
    development_end: str = "2015-12-31"
    buffer_end: str = "2016-03-31"
    test_start: str = "2016-04-01"
    test_end: str = "2026-09-30"
    validation_years: tuple[int, ...] = (2013, 2014)
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0)
    top_n: int = 20
    cohort_fraction: float = 1 / 3
    holding_days: int = 90
    fee_bps: float = 1.0
    name_cap: float = 0.05
    sector_cap: float = 0.25
    max_stale_sessions: int = 126
    reappearing_price_ratio_threshold: float = 20.0
    max_exit_delay: int = 5
    annual_rf: float = 0.02
    initial_cash: float = 1_000_000.0
    fit_frequency: str = "quarterly"
    target_clip_lower: float = -0.9
    target_clip_upper: float = 5.0


def clip_target(target: pd.Series, lower: float = -0.9, upper: float = 5.0) -> pd.Series:
    """Winsorize extreme 90-day returns for model fitting.

    The database contains a handful of implausible adjusted-price jumps. They are
    still valid labels in the audit tables, but letting them dominate a squared-error
    linear model would make the forecasts useless for ranking stocks.
    """
    return target.clip(lower=lower, upper=upper)


def month_end_dates(calendar: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """last market session of every calendar month in [start, end]."""
    cal = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if not len(cal):
        return pd.DatetimeIndex([], name="date")
    return pd.DatetimeIndex(
        pd.Series(cal).groupby(pd.Series(cal).dt.to_period("M")).apply(lambda s: s.iloc[-1])
    )


def quarter_end_dates(calendar: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Last market session of each calendar quarter in [start, end]."""
    cal = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if not len(cal):
        return pd.DatetimeIndex([], name="date")
    return pd.DatetimeIndex(
        pd.Series(cal).groupby(pd.Series(cal).dt.to_period("Q")).apply(lambda s: s.iloc[-1])
    )


def load_metadata(database: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load benchmark, universe, quarterly membership, and fundamentals"""
    with duckdb.connect(str(database), read_only=True) as con:
        benchmark = con.execute(
            "SELECT date, adj_close FROM benchmark ORDER BY date"
        ).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
        membership = con.execute(
            """SELECT symbol, quarter_start, quarter_end, selection_date, universe_rank
               FROM universe_membership ORDER BY quarter_start, universe_rank"""
        ).fetchdf()
        fundamentals = con.execute(
            """SELECT symbol, fiscal_date, period, filing_date, reported_currency,
                      revenue, net_income, operating_income, total_assets, total_debt,
                      cash_and_equivalents, free_cash_flow, operating_cash_flow,
                      shares_outstanding
               FROM fundamentals ORDER BY symbol, fiscal_date"""
        ).fetchdf()
    for frame, cols in [
        (benchmark, ["date"]),
        (membership, ["quarter_start", "quarter_end", "selection_date"]),
        (fundamentals, ["fiscal_date", "filing_date"]),
    ]:
        for col in cols:
            frame[col] = pd.to_datetime(frame[col]).astype("datetime64[ns]")
    benchmark = benchmark.drop_duplicates("date").set_index("date").sort_index()
    return benchmark, universe, membership, fundamentals


def load_price_features(database: Path, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """compute trailing price features"""
    if len(signal_dates) == 0:
        return pd.DataFrame(
            columns=["date", "symbol", "adj_close", "momentum_21", "momentum_63",
                     "momentum_252_21", "momentum_10", "momentum_5_10_5",
                     "momentum_5_21_5", "momentum_5_63_5", "momentum_5_252_5",
                     "volatility_10",
                     "volatility_21", "volatility_63", "drawdown_252",
                     "dollar_volume_10", "dollar_volume_21", "dollar_volume_63",
                     "relative_volume_10", "relative_volume_21", "relative_volume_63",
                     "amihud_illiquidity_10", "amihud_illiquidity_21", "amihud_illiquidity_63",
                     "return_volume_corr_10", "return_volume_corr_21", "return_volume_corr_63",
                     "eligible"]
        )
    with duckdb.connect(str(database), read_only=True) as con:
        con.register("signal_dates", pd.DataFrame({"date": pd.to_datetime(signal_dates)}))
        sql = """
        WITH cal AS (SELECT date FROM calendar),
        sym AS (SELECT symbol FROM universe),
        base AS (
          SELECT s.symbol, c.date, p.adj_close, p.volume
          FROM sym s
          CROSS JOIN cal c
          LEFT JOIN prices p
            ON p.symbol = s.symbol AND p.date = c.date
          WHERE p.adj_close IS NULL OR (p.adj_close > 0 AND isfinite(p.adj_close))
        ),
        px AS (
          SELECT
            symbol,
            date,
            adj_close,
            CASE WHEN volume > 0 AND isfinite(volume) THEN volume END AS volume_clean,
            adj_close * CASE WHEN volume > 0 AND isfinite(volume) THEN volume END AS dollar_volume,
            CASE WHEN adj_close IS NOT NULL
                 THEN adj_close / LAG(adj_close, 1) OVER w - 1 END AS daily_return,
            CASE WHEN adj_close IS NOT NULL
                 THEN adj_close / LAG(adj_close, 10) OVER w - 1 END AS momentum_10,
            CASE WHEN adj_close IS NOT NULL
                 THEN adj_close / LAG(adj_close, 21) OVER w - 1 END AS momentum_21,
            CASE WHEN adj_close IS NOT NULL
                 THEN adj_close / LAG(adj_close, 63) OVER w - 1 END AS momentum_63,
            CASE WHEN adj_close IS NOT NULL
                 THEN LAG(adj_close, 21) OVER w / LAG(adj_close, 252) OVER w - 1 END AS momentum_252_21,
            CASE WHEN adj_close IS NOT NULL
                 THEN AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                      )
                      / NULLIF(AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 14 PRECEDING AND 10 PRECEDING
                      ), 0) - 1
            END AS momentum_5_10_5,
            CASE WHEN adj_close IS NOT NULL
                 THEN AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                      )
                      / NULLIF(AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 25 PRECEDING AND 21 PRECEDING
                      ), 0) - 1
            END AS momentum_5_21_5,
            CASE WHEN adj_close IS NOT NULL
                 THEN AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                      )
                      / NULLIF(AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 67 PRECEDING AND 63 PRECEDING
                      ), 0) - 1
            END AS momentum_5_63_5,
            CASE WHEN adj_close IS NOT NULL
                 THEN AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                      )
                      / NULLIF(AVG(adj_close) OVER (
                        PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 256 PRECEDING AND 252 PRECEDING
                      ), 0) - 1
            END AS momentum_5_252_5,
            CASE WHEN adj_close IS NOT NULL
                 THEN adj_close / MAX(adj_close) OVER (
                      PARTITION BY symbol ORDER BY date
                      ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
                    ) - 1 END AS drawdown_252
          FROM base
          WINDOW w AS (PARTITION BY symbol ORDER BY date)
        ),
        stats AS (
          SELECT
            *,
            LEAST(GREATEST(daily_return, -0.999), 100.0) AS daily_return_capped
          FROM px
        ),
        windows AS (
          SELECT
            *,
            COUNT(daily_return_capped) OVER w63 AS ret_count,
            STDDEV_SAMP(daily_return_capped) OVER w10 * sqrt(252) AS volatility_10,
            STDDEV_SAMP(daily_return_capped) OVER w21 * sqrt(252) AS volatility_21,
            STDDEV_SAMP(daily_return_capped) OVER w63 * sqrt(252) AS volatility_63,
            AVG(dollar_volume) OVER w10 AS dollar_volume_10,
            AVG(dollar_volume) OVER w21 AS dollar_volume_21,
            AVG(dollar_volume) OVER w63 AS dollar_volume_63,
            CASE
              WHEN AVG(volume_clean) OVER w10 > 0
              THEN volume_clean / AVG(volume_clean) OVER w10 - 1
            END AS relative_volume_10,
            CASE
              WHEN AVG(volume_clean) OVER w21 > 0
              THEN volume_clean / AVG(volume_clean) OVER w21 - 1
            END AS relative_volume_21,
            CASE
              WHEN AVG(volume_clean) OVER w63 > 0
              THEN volume_clean / AVG(volume_clean) OVER w63 - 1
            END AS relative_volume_63,
            AVG(
              CASE WHEN dollar_volume > 0 THEN ABS(daily_return_capped) / dollar_volume END
            ) OVER w10 * 1000000.0 AS amihud_illiquidity_10,
            AVG(
              CASE WHEN dollar_volume > 0 THEN ABS(daily_return_capped) / dollar_volume END
            ) OVER w21 * 1000000.0 AS amihud_illiquidity_21,
            AVG(
              CASE WHEN dollar_volume > 0 THEN ABS(daily_return_capped) / dollar_volume END
            ) OVER w63 * 1000000.0 AS amihud_illiquidity_63,
            CORR(daily_return_capped, LN(1 + COALESCE(volume_clean, 0))) OVER w10 AS return_volume_corr_10,
            CORR(daily_return_capped, LN(1 + COALESCE(volume_clean, 0))) OVER w21 AS return_volume_corr_21,
            CORR(daily_return_capped, LN(1 + COALESCE(volume_clean, 0))) OVER (
              PARTITION BY symbol ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
            ) AS return_volume_corr_63
          FROM stats
          WINDOW
            w10 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
            w21 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW),
            w63 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW)
        )
        SELECT
          symbol,
          date,
          adj_close,
          momentum_21,
          momentum_63,
          momentum_252_21,
          momentum_10,
          momentum_5_10_5,
          momentum_5_21_5,
          momentum_5_63_5,
          momentum_5_252_5,
          volatility_10,
          volatility_21,
          volatility_63,
          drawdown_252,
          dollar_volume_10,
          dollar_volume_21,
          dollar_volume_63,
          relative_volume_10,
          relative_volume_21,
          relative_volume_63,
          amihud_illiquidity_10,
          amihud_illiquidity_21,
          amihud_illiquidity_63,
          return_volume_corr_10,
          return_volume_corr_21,
          return_volume_corr_63,
          (adj_close IS NOT NULL AND ret_count >= 63 AND volatility_63 IS NOT NULL) AS eligible
        FROM windows
        WHERE date IN (SELECT date FROM signal_dates)
          AND adj_close IS NOT NULL
        ORDER BY symbol, date
        """
        features = con.execute(sql).fetchdf()
    features["date"] = pd.to_datetime(features.date).astype("datetime64[ns]")
    return features.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_turnover_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add turnover proxies using the latest already-available share count."""
    market_value_proxy = panel.adj_close * panel.shares_outstanding
    for horizon in [10, 21, 63]:
        panel[f"turnover_{horizon}"] = (
            panel[f"dollar_volume_{horizon}"] / market_value_proxy.where(market_value_proxy.gt(0))
        )
    return panel


def add_sector_neutral_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Add same-date, same-sector peer-relative versions of all numeric factors."""
    eligible = panel.loc[panel.eligible].copy()
    if eligible.empty:
        for column in SECTOR_RELATIVE_FEATURES + ["sector_profitability_rank"]:
            panel[column] = np.nan
        return panel

    sector_groups = eligible.groupby(["date", "sector"], sort=False)
    for feature in SECTOR_NEUTRAL_BASE_FEATURES:
        peer_count = sector_groups[feature].transform("count")
        panel[f"sector_relative_{feature}"] = np.nan
        panel.loc[eligible.index, f"sector_relative_{feature}"] = (
            eligible[feature] - sector_groups[feature].transform("median")
        ).where(peer_count >= 3)

    profit_count = sector_groups.operating_roa.transform("count")
    panel["sector_profitability_rank"] = np.nan
    panel.loc[eligible.index, "sector_profitability_rank"] = (
        sector_groups.operating_roa.rank(pct=True).where(profit_count >= 3)
    )
    return panel


def load_quote_matrix(database: Path, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """a wide adjusted-close matrix indexed by the benchmark calendar."""
    with duckdb.connect(str(database), read_only=True) as con:
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()
        values = ", ".join(repr(s) for s in symbols)
        sql = f"""
        WITH clean AS (
          SELECT date, symbol, adj_close
          FROM prices
          WHERE adj_close > 0
            AND isfinite(adj_close)
            AND date IN (SELECT date FROM calendar)
        )
        SELECT * FROM (
          SELECT date, symbol, adj_close FROM clean
        )
        PIVOT (first(adj_close) FOR symbol IN ({values}))
        ORDER BY date
        """
        wide = con.execute(sql).fetchdf()
    wide["date"] = pd.to_datetime(wide.date).astype("datetime64[ns]")
    wide = wide.set_index("date").sort_index().reindex(calendar)
    wide.columns.name = "symbol"
    return wide


def build_panel(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.Series]:
    """Build the signal-date feature panel, labels, price matrix, and benchmark."""
    t0 = time.time()
    benchmark, universe, membership, fundamentals = load_metadata(config.database)
    calendar = pd.DatetimeIndex(benchmark.index)

    # Signals are the last market session before each entry month. We need them
    # through August 2026 for the final entry on September 1, 2026.
    signal_dates = month_end_dates(calendar, config.train_start, "2026-08-31")
    print(f"loaded metadata and {len(signal_dates)} monthly signal dates in {time.time()-t0:.1f}s", flush=True)

    price_panel = load_price_features(config.database, signal_dates)
    print(f"computed price features for {len(price_panel)} signal rows in {time.time()-t0:.1f}s", flush=True)

    events = cd.financial_events(fundamentals)
    panel = cd.join_financials(price_panel, events, universe)
    panel = add_turnover_features(panel)
    panel = cd.apply_membership(panel, membership, calendar)
    panel["eligible"] = panel["eligible"] & panel["is_universe_member"]
    panel = add_sector_neutral_factors(panel)

    quotes = load_quote_matrix(config.database, calendar)
    panel = make_labels_from_quotes(
        panel, quotes, calendar,
        holding_days=config.holding_days, max_exit_delay=config.max_exit_delay,
    )
    print(f"built panel with labels in {time.time()-t0:.1f}s", flush=True)
    return panel, quotes, calendar, benchmark.adj_close


def make_labels_from_quotes(
    panel: pd.DataFrame,
    quotes: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    holding_days: int = 90,
    max_exit_delay: int = 5,
) -> pd.DataFrame:
    """Next-session entry to first observed close on/after 90 calendar days.

    This is adapted from course_data.make_labels, but reads from the wide quote
    matrix rather than a long price DataFrame.
    """
    calendar = pd.DatetimeIndex(calendar)
    pieces = []
    for symbol, group in panel.groupby("symbol", sort=True):
        g = group.copy()
        positions = calendar.get_indexer(g.date) + 1
        entry = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns]")
        valid_entry = (positions > 0) & (positions < len(calendar))
        entry.loc[valid_entry] = calendar[positions[valid_entry]].to_numpy()
        g["entry_date"] = entry
        g["entry_price"] = quotes[symbol].reindex(entry).to_numpy() if valid_entry.any() else np.nan
        g["scheduled_exit"] = entry + pd.Timedelta(days=holding_days)
        g["label_end"] = pd.NaT
        g["exit_price"] = np.nan

        valid = g.entry_price.notna() & g.scheduled_exit.le(calendar[-1])
        if valid.any():
            due = g.loc[valid, "scheduled_exit"]
            col = quotes[symbol].dropna()
            ix = col.index.searchsorted(due)
            has_quote = ix < len(col)
            rows = due.index[has_quote]
            exits = col.index[ix[has_quote]]
            delay = calendar.get_indexer(exits) - calendar.searchsorted(due.loc[rows])
            rows = rows[delay <= max_exit_delay]
            exits = exits[delay <= max_exit_delay]
            g.loc[rows, "label_end"] = exits.to_numpy()
            g.loc[rows, "exit_price"] = col.reindex(exits).to_numpy()
        g["target"] = g.exit_price / g.entry_price - 1
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def development_folds(panel: pd.DataFrame, config: Config):
    """Chronological, purged validation folds using whole signal dates."""
    dev = panel.loc[
        panel.eligible
        & panel.date.ge(config.train_start)
        & panel.date.le(config.development_end)
        & panel.target.notna()
    ].copy()
    dev["model_target"] = clip_target(
        dev.target, config.target_clip_lower, config.target_clip_upper
    )
    folds = []
    for year in config.validation_years:
        valid = dev.date.between(f"{year}-01-01", f"{year}-12-31")
        if not valid.any():
            raise ValueError(f"No validation rows for {year}")
        first_signal = dev.loc[valid, "date"].min()
        train = (
            dev.date.ge(config.train_start)
            & dev.date.lt(first_signal)
            & dev.label_end.lt(first_signal)
            & dev.target.notna()
        )
        tr, va = np.flatnonzero(train), np.flatnonzero(valid)
        if not len(tr):
            raise ValueError(f"No mature training labels before {year}")
        if dev.iloc[tr].label_end.max() >= dev.iloc[va].date.min():
            raise AssertionError("Training label overlap with validation")
        folds.append((tr, va))
    return dev, folds


def fit_score_quarter(
    panel: pd.DataFrame,
    fit_date: pd.Timestamp,
    next_fit: pd.Timestamp,
    model_template,
    config: Config,
    features: list[str] | None = None,
):
    """Fit on outcomes completed before fit_date, score every signal in the interval."""
    features = FEATURES if features is None else features
    available = panel.loc[
        panel.eligible
        & panel.date.ge(config.train_start)
        & panel.date.le(fit_date)
        & panel.label_end.lt(fit_date)
        & panel.target.notna()
    ].copy()
    to_score = panel.loc[
        panel.eligible
        & panel.date.ge(fit_date)
        & panel.date.lt(next_fit)
    ].copy()
    if to_score.empty:
        return (
            pd.DataFrame(columns=["date", "symbol", "score", "fit_date"]),
            {
                "fit_date": fit_date,
                "rows": len(available),
                "last_training_signal": available.date.max() if len(available) else pd.NaT,
                "last_label_end": available.label_end.max() if len(available) else pd.NaT,
                "first_scored_signal": pd.NaT,
                "last_scored_signal": pd.NaT,
            },
        )
    model = clone(model_template).fit(
        available[features],
        clip_target(available.target, config.target_clip_lower, config.target_clip_upper),
    )
    predicted = to_score[["date", "symbol"]].copy()
    predicted["score"] = model.predict(to_score[features])
    predicted["fit_date"] = fit_date
    audit = {
        "fit_date": fit_date,
        "rows": len(available),
        "last_training_signal": available.date.max() if len(available) else pd.NaT,
        "last_label_end": available.label_end.max() if len(available) else pd.NaT,
        "first_scored_signal": to_score.date.min() if len(to_score) else pd.NaT,
        "last_scored_signal": to_score.date.max() if len(to_score) else pd.NaT,
    }
    return predicted, audit


class MonthlySignalStrategy:
    """A monthly long-only strategy using precomputed scores and concentration caps."""

    def __init__(self, scores: pd.DataFrame, sectors: pd.Series, config: Config):
        scores = scores[["date", "symbol", "score"]].copy()
        scores["date"] = pd.to_datetime(scores.date)
        if scores[["date", "symbol"]].isna().any().any():
            raise ValueError("Scores need dates and symbols")
        if scores.duplicated(["date", "symbol"]).any() or not np.isfinite(scores.score).all():
            raise ValueError("Scores must be finite with one row per date and symbol")
        self.scores = {
            date: group.set_index("symbol").score
            for date, group in scores.groupby("date")
        }
        self.sectors = sectors.fillna("Unknown")
        self.config = config
        self.holding_days = config.holding_days

    def orders(self, signal_date: pd.Timestamp, execution_date: pd.Timestamp, portfolio):
        # The monthly schedule buys only at the first session of a new month.
        if execution_date.to_period("M") == signal_date.to_period("M"):
            return pd.Series(dtype=float)
        scores = self.scores.get(signal_date)
        if scores is None or scores.empty:
            return pd.Series(dtype=float)
        scores = scores[np.isfinite(scores)]
        # A long-only book should not deliberately buy names the model thinks will fall.
        scores = scores[scores > 0]
        if scores.empty:
            return pd.Series(dtype=float)
        return capped_budgets(
            scores,
            portfolio.holdings,
            self.sectors,
            portfolio.cash,
            portfolio.nav,
            top_n=self.config.top_n,
            cohort_fraction=self.config.cohort_fraction,
            name_cap=self.config.name_cap,
            sector_cap=self.config.sector_cap,
            fee_bps=self.config.fee_bps,
        )


class SignalBacktester:
    """Daily close-to-close backtester.
    The daily loop and execution schedule are implemented here so the full-data
    run does not need to copy a 3,849 x 6,726 price history on every iteration.
    """

    def __init__(self, quotes: pd.DataFrame, benchmark: pd.Series, strategy, config: Config):
        self.quotes = quotes
        self.calendar = pd.DatetimeIndex(quotes.index)
        self.benchmark = benchmark
        self.strategy = strategy
        self.config = config
        self.oms = OrderManagementSystem(
            initial_cash=config.initial_cash, fee_bps=config.fee_bps
        )
        self.entry_dates = set(
            pd.DatetimeIndex(
                pd.Series(self.calendar).groupby(pd.Series(self.calendar).dt.to_period("M")).apply(
                    lambda s: s.iloc[0]
                )
            )
        )

    @staticmethod
    def _validate_orders(orders: pd.Series, cash: float) -> None:
        if not isinstance(orders, pd.Series) or not orders.index.is_unique or orders.index.hasnans:
            raise ValueError("Orders must be a Series of budgets with unique symbols")
        if not np.isfinite(orders).all() or orders.lt(0).any() or orders.sum() > cash + 1e-8:
            raise ValueError("Order budgets are invalid or exceed cash")

    def _check_held_marks(self, day_index: int, observed: set, last_seen, diagnostics):
        stale = 0
        for symbol in sorted({lot.symbol for lot in self.oms.lots.values()}):
            if symbol not in observed:
                stale += 1
                diagnostics.append(
                    {"date": self.calendar[day_index], "symbol": symbol, "event": "stale_mark"}
                )
                if day_index - last_seen.get(symbol, -10_000) > self.config.max_stale_sessions:
                    diagnostics.append(
                        {
                            "date": self.calendar[day_index],
                            "symbol": symbol,
                            "event": "stale_mark_over_limit",
                        }
                    )
        return stale

    def run(self) -> BacktestResult:
        start, end = pd.Timestamp(self.config.test_start), pd.Timestamp(self.config.test_end)
        days = self.calendar[(self.calendar >= start) & (self.calendar <= end)]
        if not len(days) or self.calendar.get_loc(days[0]) == 0:
            raise ValueError("Need test sessions and one preceding signal session")

        last_marks, last_seen, diagnostics = {}, {}, []
        previous_day = self.calendar[self.calendar.get_loc(days[0]) - 1]
        history = [
            {
                "date": previous_day,
                "cash": self.config.initial_cash,
                "holdings_value": 0.0,
                "nav": self.config.initial_cash,
                "open_lots": 0,
                "fees": 0.0,
                "gross_traded": 0.0,
                "stale_marks": 0,
            }
        ]

        for day in days:
            day_index = self.calendar.get_loc(day)
            signal_date = self.calendar[day_index - 1]
            portfolio = self.oms.snapshot(last_marks)
            orders = self.strategy.orders(signal_date, day, portfolio) if day in self.entry_dates else pd.Series(dtype=float)
            self._validate_orders(orders, portfolio.cash)

            observed = self.quotes.loc[day].dropna()
            held_symbols = {lot.symbol for lot in self.oms.lots.values()}
            for symbol, price in observed.items():
                gap = day_index - last_seen.get(symbol, day_index)
                last_price = last_marks.get(symbol, np.nan)
                large_price_change = (
                    not np.isfinite(last_price)
                    or last_price <= 0
                    or price / last_price > self.config.reappearing_price_ratio_threshold
                    or last_price / price > self.config.reappearing_price_ratio_threshold
                )
                if (
                    symbol in held_symbols
                    and gap > self.config.max_stale_sessions
                    and large_price_change
                ):
                    diagnostics.append(
                        {
                            "date": day,
                            "symbol": symbol,
                            "event": "ignored_reappearing_quote_after_long_gap",
                        }
                    )
                    continue
                last_marks[symbol] = price
                last_seen[symbol] = day_index
            stale = self._check_held_marks(day_index, set(observed.index), last_seen, diagnostics)

            fill_start = len(self.oms.fills)
            for lot in list(self.oms.lots.values()):
                within_stale_window = (
                    day_index - last_seen.get(lot.symbol, day_index)
                    <= self.config.max_stale_sessions
                )
                if lot.due_date <= day and lot.symbol in observed and within_stale_window:
                    self.oms.sell(
                        lot.units,
                        observed[lot.symbol],
                        lot_id=lot.lot_id,
                        date=day,
                    )

            for symbol, budget in orders.items():
                if budget <= 1e-8:
                    continue
                if symbol not in observed:
                    diagnostics.append(
                        {"date": day, "symbol": symbol, "event": "rejected_missing_quote"}
                    )
                    continue
                price = observed[symbol]
                size = self.oms.size_for_budget(budget, price)
                self.oms.buy(
                    size,
                    price,
                    symbol=symbol,
                    date=day,
                    signal_date=signal_date,
                    holding_days=self.config.holding_days,
                )

            portfolio = self.oms.snapshot(last_marks)
            daily_fills = self.oms.fills[fill_start:]
            history.append(
                {
                    "date": day,
                    "cash": portfolio.cash,
                    "holdings_value": portfolio.holdings.sum(),
                    "nav": portfolio.nav,
                    "open_lots": len(self.oms.lots),
                    "fees": sum(f["fee"] for f in daily_fills),
                    "gross_traded": sum(f["notional"] for f in daily_fills),
                    "stale_marks": stale,
                }
            )

        history = pd.DataFrame(history).set_index("date")
        history["benchmark_nav"] = self.config.initial_cash
        if len(days):
            history.loc[days, "benchmark_nav"] = (
                self.config.initial_cash
                * self.benchmark.reindex(days)
                / self.benchmark.loc[days[0]]
            )
        fills = pd.DataFrame(self.oms.fills, columns=self.oms.FILL_COLUMNS)
        open_lots = pd.DataFrame(
            [asdict(lot) for lot in self.oms.lots.values()],
            columns=LOT_FIELDS,
        )
        diagnostics_df = pd.DataFrame(diagnostics, columns=["date", "symbol", "event"])
        return BacktestResult(
            history=history,
            fills=fills,
            open_lots=open_lots,
            diagnostics=diagnostics_df,
            pnl=self.oms.pnl(last_marks),
        )

def performance_table(result: BacktestResult, config: Config) -> pd.DataFrame:
    return BacktestAnalytics(result).summary(config.annual_rf)


def validate_accounting(result: BacktestResult, quotes: pd.DataFrame, config: Config) -> dict:
    """Independent checks of cash, timing, and one verifiable buy/sell pair."""
    h = result.history
    checks = {}
    checks["final_nav_matches_cash_plus_open_marks"] = bool(
        np.isclose(h.nav.iloc[-1], result.pnl.nav, atol=1e-6)
    )
    checks["no_negative_cash"] = bool((h.cash >= -1e-8).all())

    reconstructed = (
        result.fills.groupby("date").cash_change.sum().reindex(h.index).fillna(0.0).cumsum()
        + config.initial_cash
    )
    checks["cash_reconstruction_max_abs_error"] = float(
        (reconstructed - h.cash).abs().max()
    )
    checks["cash_reconstruction_passed"] = bool(
        np.allclose(reconstructed, h.cash, atol=1e-7)
    )

    if not result.fills.empty:
        buys = result.fills[result.fills.side.eq("BUY")]
        if len(buys):
            first = buys.iloc[0]
            checks["first_buy_due_date_is_90_calendar_days"] = bool(
                pd.Timestamp(first.due_date) == pd.Timestamp(first.date) + pd.Timedelta(days=config.holding_days)
            )
            checks["first_buy_signal_precedes_fill"] = bool(
                pd.Timestamp(first.signal_date) < pd.Timestamp(first.date)
            )
            checks["first_buy_fill_price_matches_quote"] = bool(
                np.isclose(float(first.price), quotes.at[pd.Timestamp(first.date), first.symbol], rtol=0, atol=1e-9)
            )
        sells = result.fills[result.fills.side.eq("SELL")]
        if len(sells):
            sell = sells.iloc[0]
            matching_buys = result.fills[
                result.fills.lot_id.eq(sell.lot_id) & result.fills.side.eq("BUY")
            ]
            if len(matching_buys):
                buy = matching_buys.iloc[0]
                expected_pnl = (
                    sell["units"] * (sell["price"] - buy["price"])
                    - buy["fee"] * sell["units"] / buy["units"]
                    - sell["fee"]
                )
                checks["first_sell_pnl_matches_hand_calc"] = bool(
                    np.isclose(expected_pnl, sell["realized_pnl"], atol=1e-6)
                )
            checks["all_sells_on_or_after_due_date"] = bool(
                (
                    pd.to_datetime(sells.date)
                    >= pd.to_datetime(sells.due_date)
                ).all()
            )
    return checks


def save_outputs(output_dir: Path, result: BacktestResult, scores: pd.DataFrame,
                 fit_audit: pd.DataFrame, validation: pd.DataFrame, config: Config,
                 checks: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(output_dir / "nav_history.csv")
    result.fills.to_csv(output_dir / "fills.csv", index=False)
    result.open_lots.to_csv(output_dir / "open_lots.csv", index=False)
    result.diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    result.pnl.to_csv(output_dir / "pnl.csv")
    scores.to_csv(output_dir / "predictions.csv", index=False)
    fit_audit.to_csv(output_dir / "fit_audit.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    (output_dir / "accounting_checks.json").write_text(json.dumps(checks, indent=2, default=str))

    config_out = output_dir / "config.json"
    config_out.write_text(
        json.dumps(
            {
                "database": str(config.database),
                "train_start": config.train_start,
                "development_end": config.development_end,
                "buffer_end": config.buffer_end,
                "test_start": config.test_start,
                "test_end": config.test_end,
                "validation_years": list(config.validation_years),
                "alpha_grid": list(config.alpha_grid),
                "top_n": config.top_n,
                "cohort_fraction": config.cohort_fraction,
                "holding_days": config.holding_days,
                "fee_bps": config.fee_bps,
                "name_cap": config.name_cap,
                "sector_cap": config.sector_cap,
                "max_stale_sessions": config.max_stale_sessions,
                "reappearing_price_ratio_threshold": config.reappearing_price_ratio_threshold,
                "max_exit_delay": config.max_exit_delay,
                "annual_rf": config.annual_rf,
                "initial_cash": config.initial_cash,
                "fit_frequency": config.fit_frequency,
                "target_clip_lower": config.target_clip_lower,
                "target_clip_upper": config.target_clip_upper,
            },
            indent=2,
            default=str,
        )
    )


def plot_results(output_dir: Path, result: BacktestResult, config: Config) -> None:
    h = result.history
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(h.index, h.nav / config.initial_cash - 1, label="Strategy", linewidth=2)
    ax.plot(
        h.index,
        h.benchmark_nav / config.initial_cash - 1,
        label="S&P 500 total return",
        linewidth=2,
        linestyle="--",
    )
    ax.set_title("HW4 cumulative return vs S&P 500 total return")
    ax.set_ylabel("Cumulative net return")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_return.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="DuckDB path. Defaults to the full HW4 database when present.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Use the classroom database.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    args = parser.parse_args()

    if args.quick or (args.database is None and not DEFAULT_FULL_DB.exists()):
        database = DEFAULT_CLASS_DB
    elif args.database is None:
        database = DEFAULT_FULL_DB
    else:
        database = args.database
    if not Path(database).is_file():
        raise FileNotFoundError(f"Database not found: {database}")

    output_dir = args.output_dir or (Path(__file__).resolve().parent / "outputs")
    config = Config(
        database=Path(database).resolve(),
        output_dir=output_dir.resolve(),
        top_n=args.top_n,
        fee_bps=args.fee_bps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building panel...", flush=True)
    panel, quotes, calendar, benchmark_series = build_panel(config)

    # The classroom extract ends in 2015 and is only used for a quick sanity run.
    if calendar.max() < pd.Timestamp(config.test_start):
        quick_start = calendar[calendar >= pd.Timestamp("2015-01-01")]
        if not len(quick_start):
            raise ValueError("Classroom database has no 2015 data for a quick run")
        config.test_start = str(quick_start.min().date())
        config.test_end = str(calendar.max().date())

    print("Selecting and validating the Ridge model...", flush=True)
    development, folds = development_folds(panel, config)
    search = GridSearchCV(
        cd.make_pipeline(Ridge(), numeric_features=NUMERIC_FEATURES),
        {"model__alpha": list(config.alpha_grid)},
        cv=folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(development[FEATURES], development.model_target)
    chosen_model = search.best_estimator_
    validation = pd.DataFrame(
        {
            "alpha": search.cv_results_["param_model__alpha"],
            "mean_test_neg_rmse": search.cv_results_["mean_test_score"],
            "rank_test_neg_rmse": search.cv_results_["rank_test_score"],
        }
    )
    print(validation.to_string(index=False), flush=True)

    # A small, honest holdout evaluation on 2015 using data only through 2014.
    holdout_fit_date = calendar[calendar <= pd.Timestamp("2014-12-31")].max()
    holdout_train = panel.loc[
        panel.eligible
        & panel.date.ge(config.train_start)
        & panel.date.le(holdout_fit_date)
        & panel.label_end.lt(holdout_fit_date)
        & panel.target.notna()
    ].copy()
    holdout_score = panel.loc[
        panel.eligible
        & panel.date.between("2015-01-01", config.development_end)
    ].copy()
    holdout_model = clone(chosen_model).fit(
        holdout_train[FEATURES],
        clip_target(holdout_train.target, config.target_clip_lower, config.target_clip_upper),
    )
    holdout_pred = holdout_model.predict(holdout_score[FEATURES])
    holdout_eval = holdout_score.loc[holdout_score.target.notna()].copy()
    holdout_eval["target"] = clip_target(
        holdout_eval.target, config.target_clip_lower, config.target_clip_upper
    )
    holdout_metrics = cd.prediction_metrics(
        holdout_eval,
        holdout_pred[holdout_score.target.notna().to_numpy()],
    )
    print("2015 holdout metrics:", holdout_metrics.to_dict(), flush=True)

    print("Fitting walk-forward models and scoring test signals...", flush=True)
    first_test_signal = calendar[calendar < pd.Timestamp(config.test_start)].max()
    last_test_signal = month_end_dates(calendar, config.test_start, "2026-08-31").max()
    fit_dates = quarter_end_dates(calendar, first_test_signal, last_test_signal)
    if first_test_signal not in fit_dates:
        # First fit should be the final session before the first possible test entry.
        fit_dates = pd.DatetimeIndex([first_test_signal, *fit_dates[fit_dates > first_test_signal]])

    score_parts, audit_parts = [], []
    for i, fit_date in enumerate(fit_dates):
        next_fit = fit_dates[i + 1] if i + 1 < len(fit_dates) else pd.Timestamp("2026-09-01")
        predicted, audit = fit_score_quarter(panel, fit_date, next_fit, chosen_model, config)
        score_parts.append(predicted)
        audit_parts.append(audit)
        if len(predicted):
            assert predicted.date.min() >= fit_date
            assert predicted.date.max() < next_fit
    nonempty_score_parts = [part for part in score_parts if not part.empty]
    test_scores = (
        pd.concat(nonempty_score_parts, ignore_index=True)
        if nonempty_score_parts
        else pd.DataFrame(columns=["date", "symbol", "score", "fit_date"])
    )
    fit_audit = pd.DataFrame(audit_parts)
    test_scores = test_scores.loc[test_scores.date.ge(first_test_signal)]
    assert not test_scores.duplicated(["date", "symbol"]).any()

    sectors = pd.Series(
        panel.drop_duplicates("symbol").set_index("symbol").sector.fillna("Unknown")
    )
    strategy = MonthlySignalStrategy(test_scores, sectors, config)
    print("Running the backtester...", flush=True)
    result = SignalBacktester(quotes, benchmark_series, strategy, config).run()
    summary = performance_table(result, config)
    print(summary.to_string(), flush=True)

    checks = validate_accounting(result, quotes, config)
    print(json.dumps(checks, indent=2, default=str), flush=True)
    save_outputs(
        output_dir, result, test_scores, fit_audit, validation, config, checks
    )
    plot_results(output_dir, result, config)
    print(f"Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
