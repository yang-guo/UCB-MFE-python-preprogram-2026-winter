"""Reusable features and timing helpers for the HW4 Ridge model.

Flow: cleaned prices -> historical features -> dated eligibility -> predictions.
Build future targets separately. Never pass target audit columns to a model.

The final model (run D) uses PRICE_FEATURES, the four D_RATIOS ranked within
sector, and filing_age_days. Its training targets are clipped per fit
(clip_training_target), candidates pass a volatility screen
(signal_volatility_screen), and rows touched by an implausible one-day price
jump are dropped (suspect_price_flags). run_final_test.py holds the settings.
Earlier feature sets (MARKET_FEATURES, ADDITIONAL_PRICE_FEATURES) remain for
the development notebooks that tested and rejected them.

Adapted from Lectures/modeling/course_data.py: price_features, apply_membership,
make_labels, QuantileClipper, and prediction_metrics. The course authors supply
the timing and preprocessing conventions. This module adds a cleaned-data
loader, explicit feature selection, and a split that retains unlabeled forecasts.
The module is self-contained. It does not execute either notebook on import.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

PRICE_FEATURES = [
    "momentum_21", "momentum_63", "momentum_252_21",
    "volatility_63", "drawdown_252",
]

# Keep the original list unchanged so notebooks 02 and 03 remain the baseline.
ADDITIONAL_PRICE_FEATURES = ["volatility_monthly_21", "rsi_14"]

MARKET_FEATURES = ["momentum_63", "momentum_252_21", "relative_momentum_63",
                   "volatility_63", "volatility_252", "beta_252"]


def load_development_inputs(database, end="2015-12-31", lookback_start="2008-01-01"):
    """Read only bounded development observations from either DuckDB schema.

    Supports the cleaned class database and the supplied equities_top2000.duckdb
    schema. Every dated query has an upper cutoff. Never read future price,
    membership, or benchmark observations. Snapshot sector labels remain a
    source-data limitation. No financial statements are needed for these runs.

    For the raw database, remove exact duplicate price rows, off-calendar rows,
    and nonpositive/nonfinite prices. Keep gaps; never invent a quote. Conflicting
    duplicate keys stop the run. Report exclusions. This is a bounded price-data
    adapter, not a replacement for the full cleaning/audit notebook. It loads
    the selected period into memory. Larger datasets still need adequate RAM.
    """
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(database)
    if pd.Timestamp(end) > pd.Timestamp("2015-12-31"):
        raise ValueError("This development loader must not read data after 2015-12-31.")
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute("""SELECT symbol, date, adj_close FROM prices
                                WHERE date BETWEEN ? AND ? ORDER BY symbol, date""",
                             [lookback_start, end]).fetchdf()
        sessions = con.execute("SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date",
                               [lookback_start, end]).fetchdf()
        membership = con.execute("""SELECT symbol, quarter_start, quarter_end, selection_date, universe_rank
                                    FROM universe_membership WHERE quarter_start <= ? AND selection_date <= ?
                                    ORDER BY quarter_start, universe_rank""", [end, end]).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe").fetchdf()
        benchmark = con.execute("""SELECT date, adj_close FROM benchmark
                                   WHERE symbol='^SP500TR' AND date BETWEEN ? AND ? ORDER BY date""",
                                [lookback_start, end]).fetchdf()
    prices["date"] = pd.to_datetime(prices.date).astype("datetime64[ns]")
    benchmark["date"] = pd.to_datetime(benchmark.date).astype("datetime64[ns]")
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    calendar = pd.DatetimeIndex(pd.to_datetime(sessions.date)).astype("datetime64[ns]")
    if calendar.empty or calendar.has_duplicates or calendar.hasnans:
        raise ValueError("Require unique valid market sessions.")
    read_rows = len(prices)
    prices = prices.drop_duplicates()
    exact_duplicates = read_rows - len(prices)
    valid = (prices.symbol.notna() & prices.date.isin(calendar)
             & np.isfinite(prices.adj_close) & prices.adj_close.gt(0))
    excluded = int((~valid).sum())
    prices = prices.loc[valid].reset_index(drop=True)
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("Conflicting price keys: run the data cleaner before this experiment.")
    if universe.symbol.duplicated().any() or membership.duplicated(["symbol", "quarter_start"]).any():
        raise ValueError("Duplicate sector or membership keys.")
    if benchmark.date.duplicated().any():
        raise ValueError("Duplicate benchmark dates.")
    index_prices = benchmark.set_index("date").adj_close.reindex(calendar)
    if not (np.isfinite(index_prices) & index_prices.gt(0)).all():
        raise ValueError("A valid observed benchmark price is required on every selected session.")
    audit = pd.Series({"price_rows_read": read_rows, "exact_duplicates_removed": exact_duplicates,
                       "invalid_price_rows_excluded": excluded, "price_rows_used": len(prices)}, name="Input audit")
    return prices, calendar, membership, universe.set_index("symbol").sector.to_dict(), benchmark, audit


def rolling_beta(stock_returns, market_returns, window=252, min_periods=126):
    """Trailing paired-return covariance / paired market variance.

    Inputs: daily simple-return Series on the SAME ordered market-session index.
    Use at most window sessions, with at least min_periods valid pairs. Missing
    stock returns also remove the market return from the variance calculation.
    Never backfill missing prices or use observations after the signal date.
    Output is dimensionless; zero market variance yields NaN.
    """
    if not stock_returns.index.equals(market_returns.index):
        raise ValueError("Beta inputs must have identical session indexes.")
    if not 2 <= min_periods <= window:
        raise ValueError("Require 2 <= min_periods <= window.")
    valid = np.isfinite(stock_returns) & np.isfinite(market_returns)
    stock = stock_returns.where(valid)
    market = market_returns.where(valid)
    covariance = stock.rolling(window, min_periods=min_periods).cov(market, ddof=1)
    variance = market.rolling(window, min_periods=min_periods).var(ddof=1)
    return (covariance / variance.where(variance.gt(0))).replace([np.inf, -np.inf], np.nan)


def market_relative_features(prices, calendar, benchmark):
    """Additional Run B columns: relative momentum, long volatility, and beta.

    Join by symbol/date to price_features. Existing momentum_63,
    momentum_252_21, and volatility_63 remain unchanged. Volatility requires
    252 valid consecutive daily returns and uses sample std * sqrt(252).
    Beta permits 126 paired observations within the last 252 market sessions.
    ^SP500TR supplies the market return, including its total-return adjustments.
    """
    calendar = pd.DatetimeIndex(calendar)
    market_price = benchmark.set_index("date").adj_close.reindex(calendar)
    market_return = market_price.pct_change(fill_method=None)
    market_momentum = market_price.pct_change(63, fill_method=None)
    frames = []
    for symbol, group in prices.groupby("symbol", sort=True):
        px = group.set_index("date").adj_close.reindex(calendar)
        returns = px.pct_change(fill_method=None)
        frame = pd.DataFrame({"date": calendar, "symbol": symbol,
            "relative_momentum_63": (px.pct_change(63, fill_method=None) - market_momentum).to_numpy(),
            "volatility_252": (returns.rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252)).to_numpy(),
            "beta_252": rolling_beta(returns, market_return).to_numpy()})
        frames.append(frame.loc[px.notna().to_numpy()])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def cross_sectional_ranks(panel, feature_names):
    """Add *_rank columns using only eligible stocks on the same signal date.

    pandas rank(pct=True, method='average') preserves missing values, averages
    ties, and divides by the count of nonmissing eligible values per feature.
    Noneligible rows keep missing ranks. Keep raw columns for allocation/audits.
    This operation learns no statistics across dates. It needs no future data.
    """
    result = panel.copy()
    eligible = result.eligible
    ranks = result.loc[eligible].groupby("date")[feature_names].rank(pct=True, method="average")
    for name in feature_names:
        result[f"{name}_rank"] = np.nan
        result.loc[eligible, f"{name}_rank"] = ranks[name]
    return result


def add_benchmark_targets(panel, benchmark):
    """Add benchmark and excess targets on each row's EXACT entry/exit dates.

    This is a label-only operation. Do not send these columns to the model or
    allocator. Keep stock_target separate. Use label_end for the completed
    outcome cutoff in annual_split. A missing stock or index outcome stays NaN.
    """
    result = panel.copy()
    quotes = benchmark.set_index("date").adj_close
    entry = result.entry_date.map(quotes)
    exit_price = result.label_end.map(quotes)
    result["stock_target"] = result.target
    result["benchmark_target"] = exit_price / entry - 1
    result["excess_target"] = result.stock_target - result.benchmark_target
    return result


def monthly_volatility(adjusted_close, window=21):
    """Estimate one-month volatility from the latest 21 market-session returns.

    Input: a price Series indexed by every market session, including missing
    quotes as NaN. Output: rolling sample std(daily simple returns) * sqrt(21).
    The result is in monthly decimal units, not annual units. window changes
    the estimation lookback; the scaling horizon stays 21 sessions. Require
    all window returns. Do not fill a missing price or jump across its session.
    This is a trailing estimate, not volatility realized in a future month.
    """
    if not isinstance(window, int) or window < 2:
        raise ValueError("The volatility window must be an integer of at least two.")
    returns = adjusted_close.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(21)


def relative_strength_index(adjusted_close, window=14):
    """Return Wilder RSI on [0, 100], using past/current adjusted closes only.

    Input: a price Series aligned to every market session. Seed average gain
    and loss with the first window price changes. Then update each average as
    ((window - 1) * old_average + current_change) / window. Calculate RSI as
    100 * average_gain / (average_gain + average_loss).

    Require window + 1 consecutive valid prices. Reset after a missing quote.
    Return 100 for gains without losses, 0 for losses without gains, and 50
    for a flat window (our explicit neutral-value convention). No 30/70 trade
    rule is applied. This is a continuous model input, not an order signal.
    Formula reference: https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError("The RSI window must be a positive integer.")
    result = np.full(len(adjusted_close), np.nan)
    previous = np.nan
    count = 0
    gain_sum = loss_sum = average_gain = average_loss = 0.0
    for i, price in enumerate(adjusted_close.to_numpy(dtype=float)):
        if not np.isfinite(price) or price <= 0:
            previous = np.nan
            count = 0
            gain_sum = loss_sum = average_gain = average_loss = 0.0
            continue
        if not np.isfinite(previous):
            previous = price
            continue
        change = price - previous
        previous = price
        gain, loss = max(change, 0.0), max(-change, 0.0)
        count += 1
        if count <= window:
            gain_sum += gain
            loss_sum += loss
            if count < window:
                continue
            average_gain, average_loss = gain_sum / window, loss_sum / window
        else:
            average_gain = ((window - 1) * average_gain + gain) / window
            average_loss = ((window - 1) * average_loss + loss) / window
        total = average_gain + average_loss
        result[i] = 50.0 if total == 0 else 100 * average_gain / total
    return pd.Series(result, index=adjusted_close.index, name=f"rsi_{window}")


def additional_price_features(prices, calendar):
    """Build monthly volatility and RSI without changing baseline features.

    Input: cleaned symbol/date/adj_close rows and the full session calendar.
    Output: one row per observed quote with two added price features. Join it
    to a baseline panel by symbol/date. Keep the same baseline eligibility so
    an experiment does not change its candidate pool by future availability.
    """
    frames = []
    calendar = pd.DatetimeIndex(calendar)
    for symbol, group in prices.groupby("symbol", sort=True):
        px = group.set_index("date").adj_close.reindex(calendar)
        features = pd.DataFrame({"symbol": symbol, "date": calendar,
                                 "volatility_monthly_21": monthly_volatility(px).to_numpy(),
                                 "rsi_14": relative_strength_index(px).to_numpy()})
        frames.append(features.loc[px.notna().to_numpy()])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def price_to_sales(market_cap, trailing_sales, market_cap_currency, sales_currency):
    """Calculate P/S from verified, time-aligned inputs. Return NaN if unsafe.

    Inputs are Series with the SAME index. Use contemporaneous total equity
    market capitalization and trailing twelve-month (TTM) company revenue.
    Currency labels must be known and equal. Monetary units must also match.
    The caller must verify units and use only reports published by the signal
    date. This arithmetic helper does not build TTM data or an as-of join.
    Do not pass one quarter's revenue as annual sales, or silently multiply
    adjusted prices by shares with an unverified adjustment/share-class basis.

    The class database lacks usable currency-verified P/S inputs. Notebook 04
    therefore documents this function but excludes P/S from the experiment.
    Missing or nonpositive market cap / TTM sales yields NaN, never infinity.
    P/S definition: https://scs.fidelity.com/webcontent/ap010098-etf-content/17.02/help/research/learn_er_glossary_3.shtml
    """
    series = [market_cap, trailing_sales, market_cap_currency, sales_currency]
    if not all(isinstance(s, pd.Series) and s.index.equals(market_cap.index) for s in series):
        raise ValueError("All P/S inputs must be Series with the same index.")
    cap = pd.to_numeric(market_cap, errors="coerce")
    sales = pd.to_numeric(trailing_sales, errors="coerce")
    cap_currency = market_cap_currency.astype("string").str.strip().str.upper()
    revenue_currency = sales_currency.astype("string").str.strip().str.upper()
    known_currency = cap_currency.str.fullmatch(r"[A-Z]{3}", na=False)
    same_currency = cap_currency.eq(revenue_currency).fillna(False)
    valid = np.isfinite(cap) & cap.gt(0) & np.isfinite(sales) & sales.gt(0) & known_currency & same_currency
    ratio = cap / sales.where(valid)
    return ratio.where(valid & np.isfinite(ratio)).rename("price_to_sales")


def load_cleaned_data(database, end="2015-12-31"):
    """Read required tables only. Preserve gaps and use a read-only connection.

    Input: a database exported by 01_data_preparation.ipynb.
    Output: prices, a market-session calendar, and dated universe membership.
    end is a hard data cutoff, including the prices used to build targets.
    """
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Run the cleaning notebook first: {database}")
    with duckdb.connect(str(database), read_only=True) as connection:
        prices = connection.execute(
            "SELECT symbol, date, adj_close FROM prices WHERE date <= ? ORDER BY symbol, date",
            [end],
        ).fetchdf()
        sessions = connection.execute(
            "SELECT date FROM calendar WHERE date <= ? ORDER BY date", [end]
        ).fetchdf()
        membership = connection.execute(
            """SELECT symbol, quarter_start, quarter_end, selection_date, universe_rank
               FROM universe_membership WHERE quarter_start <= ?
               ORDER BY quarter_start, universe_rank""", [end]
        ).fetchdf()
    prices["date"] = pd.to_datetime(prices.date).astype("datetime64[ns]")
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    calendar = pd.DatetimeIndex(pd.to_datetime(sessions.date)).astype("datetime64[ns]")
    if prices.empty or calendar.empty:
        raise ValueError("The selected date range has no prices or market sessions.")
    if calendar.has_duplicates or calendar.hasnans:
        raise ValueError("The market calendar must contain unique valid dates.")
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("Prices must have one row per stock and session.")
    if prices[["symbol", "date"]].isna().any().any():
        raise ValueError("Price keys must not be missing.")
    if not (np.isfinite(prices.adj_close) & prices.adj_close.gt(0)).all():
        raise ValueError("Use cleaned finite positive adjusted prices.")
    if not prices.date.isin(calendar).all():
        raise ValueError("Price dates must belong to the market calendar.")
    return prices, calendar, membership



def price_features(prices, calendar):
    """Reindex each stock to market sessions; gaps remain missing, never padded."""
    frames = []
    calendar = pd.DatetimeIndex(calendar)
    for symbol, group in prices.groupby("symbol", sort=True):
        p = group.set_index("date").reindex(calendar)
        px = p.adj_close
        daily_return = px.pct_change(fill_method=None)
        p["momentum_21"] = px.pct_change(21, fill_method=None)
        p["momentum_63"] = px.pct_change(63, fill_method=None)
        p["momentum_252_21"] = px.shift(21) / px.shift(252) - 1
        p["volatility_63"] = daily_return.rolling(63, min_periods=63).std() * np.sqrt(252)
        p["drawdown_252"] = px / px.rolling(252, min_periods=252).max() - 1
        p["symbol"] = symbol
        p.index.name = "date"
        # Predict only after 63 complete returns; longer features may still be missing.
        p["eligible"] = px.notna() & p.volatility_63.notna()
        frames.append(p.loc[px.notna()].reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def apply_membership(panel, membership, calendar):
    """Use membership for the next-session entry; keep all rows for history/marks.

    A quarter's list is known after the preceding session's close. Thus a June
    30 signal for a July 1 entry uses Q3, not Q2. Future entrants never become
    retrospectively eligible. This function does not alter prices or open lots.
    """
    if membership.duplicated(["symbol", "quarter_start"]).any():
        raise ValueError("Duplicate quarterly membership key")
    calendar = pd.DatetimeIndex(calendar)
    result = panel.copy()
    positions = calendar.get_indexer(result.date)
    next_dates = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    valid = (positions >= 0) & (positions + 1 < len(calendar))
    next_dates.loc[valid] = calendar[positions[valid] + 1].to_numpy()
    result["membership_quarter"] = next_dates.dt.to_period("Q").dt.start_time
    right = membership.rename(columns={"quarter_start": "membership_quarter",
                                        "selection_date": "universe_selection_date"})
    result = result.merge(right, on=["symbol", "membership_quarter"], how="left", validate="many_to_one")
    result["is_universe_member"] = result.universe_rank.notna()
    if not result.loc[result.is_universe_member, "universe_selection_date"].le(result.loc[result.is_universe_member, "date"]).all():
        raise ValueError("Universe selection was unavailable at the signal close")
    result["eligible"] = result.eligible & result.is_universe_member
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def make_labels(panel, prices, calendar, holding_days=90, max_exit_delay=0):
    """Next-session entry to first observed close on/after 90 calendar days.

    Entry must have a quote on the next *market* session. Missing exits may be
    delayed by max_exit_delay market sessions (zero by default). Censored rows keep missing targets;
    they remain in the feature/prediction panel and are not investment filters.
    """
    calendar = pd.DatetimeIndex(calendar)
    pieces = []
    price_groups = prices.groupby("symbol", sort=False)
    for symbol, group in panel.groupby("symbol", sort=True):
        g = group.copy()
        quotes = price_groups.get_group(symbol).set_index("date").adj_close.sort_index()
        quotes = quotes.loc[quotes.index.isin(calendar)]
        positions = calendar.get_indexer(g.date) + 1
        entry = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns]")
        valid_entry = (positions > 0) & (positions < len(calendar))
        entry.loc[valid_entry] = calendar[positions[valid_entry]].to_numpy()
        g["entry_date"] = entry
        g["entry_price"] = entry.map(quotes)
        g["scheduled_exit"] = entry + pd.Timedelta(days=holding_days)
        g["label_end"] = pd.NaT
        g["exit_price"] = np.nan
        valid = g.entry_price.notna() & g.scheduled_exit.le(calendar[-1])
        due = g.loc[valid, "scheduled_exit"]
        ix = quotes.index.searchsorted(due)
        has_quote = ix < len(quotes)
        rows = due.index[has_quote]
        exits = quotes.index[ix[has_quote]]
        delay = calendar.get_indexer(exits) - calendar.searchsorted(due.loc[rows])
        rows = rows[delay <= max_exit_delay]
        exits = exits[delay <= max_exit_delay]
        g.loc[rows, "label_end"] = exits.to_numpy()
        g.loc[rows, "exit_price"] = quotes.reindex(exits).to_numpy()
        g["target"] = g.exit_price / g.entry_price - 1
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


class QuantileClipper(TransformerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Learn per-column tail cutoffs on training rows; preserve missing values."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Require 0 <= lower < upper <= 1")
        a = validate_data(self, X, reset=True, dtype=float, ensure_all_finite="allow-nan")
        # All-missing columns are left unbounded and handled by the imputer.
        self.lower_bounds_ = np.full(a.shape[1], -np.inf)
        self.upper_bounds_ = np.full(a.shape[1], np.inf)
        observed = np.isfinite(a).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(a[:, observed], self.lower, axis=0)
        self.upper_bounds_[observed] = np.nanquantile(a[:, observed], self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_bounds_")
        a = validate_data(self, X, reset=False, dtype=float, ensure_all_finite="allow-nan")
        return np.clip(a, self.lower_bounds_, self.upper_bounds_)


def make_ridge_pipeline(alpha=100.0, feature_names=None):
    """Select explicit inputs. Learn all preprocessing from fit rows only.

    Quantile clipping limits the influence of extreme feature values.
    Median imputation preserves rows with missing long-window features.
    Missing indicators identify these rows. Scaling puts inputs on comparable
    scales before Ridge applies its coefficient penalty.
    """
    feature_names = list(PRICE_FEATURES if feature_names is None else feature_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Choose a nonempty list of unique feature names.")
    numeric = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    return Pipeline([
        ("preprocess", ColumnTransformer([("numeric", numeric, feature_names)], remainder="drop")),
        ("model", Ridge(alpha=alpha)),
    ])


def annual_split(panel, year, train_start="2010-01-01"):
    """Return training rows and ALL eligible forecast rows for one year.

    Fit just before the first signal in the year. A training target must end
    strictly before that signal. Prediction eligibility never requires a future
    entry quote, an exit quote, or a completed target. Score completed targets
    later, outside this function.
    """
    signals = panel.loc[panel.eligible & panel.date.between(f"{year}-01-01", f"{year}-12-31")].copy()
    if signals.empty:
        raise ValueError(f"No eligible signals for {year}.")
    cutoff = signals.date.min()
    train = panel.loc[
        panel.eligible & panel.date.ge(train_start) & panel.date.lt(cutoff)
        & panel.label_end.lt(cutoff) & np.isfinite(panel.target)
    ].copy()
    if train.empty:
        raise ValueError(f"No completed training targets before {cutoff.date()}.")
    return train, signals


def walk_forward_forecasts(panel, feature_names, start="2013-01-01", end="2015-12-31",
                          train_start="2010-01-01", frequency="monthly", alpha=100.0):
    """Refit an expanding-window Ridge pipeline each month or year.

    Input: a feature/target panel from make_labels. Only eligible signal rows
    receive predictions. At each period's first signal, fit on ALL eligible
    historical rows since train_start whose target ends strictly before that
    signal. Do not train only on executed trades. The latest month's prices
    can update signal features, but its 90-day targets are not yet available.

    Refit clipping, imputation, scaling, and Ridge from scratch for each period.
    Keep that fitted pipeline fixed within the period. Forecast the same daily
    rows for both frequencies. Return forecasts and an auditable fit log with
    training counts, newly included rows, outcome cutoffs, and phase timings.
    Timings include pipeline preprocessing; they exclude data loading, feature
    construction, and portfolio execution. This helper changes no saved files.
    """
    from time import perf_counter

    if frequency not in {"monthly", "annual"}:
        raise ValueError("frequency must be 'monthly' or 'annual'.")
    signals = panel.loc[panel.eligible & panel.date.between(start, end)].copy()
    if signals.empty:
        raise ValueError("No eligible signals in the requested forecast period.")
    rule = "M" if frequency == "monthly" else "Y"
    history = panel.loc[panel.eligible & panel.date.ge(train_start)]
    parts, fit_log = [], []
    previous_count = 0
    for period, group in signals.groupby(signals.date.dt.to_period(rule), sort=True):
        cutoff = group.date.min()
        train = history.loc[history.date.lt(cutoff) & history.label_end.lt(cutoff)
                            & np.isfinite(history.target)]
        if train.empty:
            raise ValueError(f"No completed training targets before {cutoff.date()}.")
        model = make_ridge_pipeline(alpha=alpha, feature_names=feature_names)
        before_fit = perf_counter()
        model.fit(train[feature_names], train.target)
        fit_seconds = perf_counter() - before_fit
        forecast = group[["date", "symbol"]].copy()
        before_predict = perf_counter()
        forecast["prediction"] = model.predict(group[feature_names])
        predict_seconds = perf_counter() - before_predict
        forecast["fit_cutoff"] = cutoff
        parts.append(forecast)
        fit_log.append({"period": str(period), "fit_cutoff": cutoff,
                        "last_training_signal": train.date.max(),
                        "last_training_outcome": train.label_end.max(),
                        "training_rows": len(train), "new_training_rows": len(train) - previous_count,
                        "forecast_rows": len(group), "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds})
        previous_count = len(train)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True), pd.DataFrame(fit_log)



def prediction_metrics(frame, predictions):
    """Separate pooled error from same-date cross-sectional ranking."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    result = frame[["date", "symbol", "target"]].copy()
    result["prediction"] = np.asarray(predictions)
    ic = result.groupby("date").apply(
        lambda x: x.prediction.corr(x.target, method="spearman")
        if len(x) >= 5 and x.prediction.nunique() > 1 and x.target.nunique() > 1 else np.nan,
        include_groups=False,
    )
    return pd.Series({"RMSE": np.sqrt(mean_squared_error(result.target, result.prediction)),
                      "MAE": mean_absolute_error(result.target, result.prediction),
                      "R2": r2_score(result.target, result.prediction),
                      "mean_daily_rank_IC": ic.mean(), "IC_dates": ic.notna().sum()})


# Adapted verbatim from Lectures/modeling/course_data.py: financial_events and
# join_financials. Publication lag, latest fiscal period, and 550-day staleness
# follow the course. Vendor restatements cannot be undone by availability dates.
FINANCIAL_FEATURES = ["net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets", "accruals_to_assets"]
D_RATIOS = FINANCIAL_FEATURES[:4]
def financial_events(fundamentals, extra_lag_days=0):
    """Turn quarterly records into conservative availability-dated feature events.

    A filing date has no time of day: use it only from the following calendar
    day's close. Nonpositive filing lags are untrusted and excluded. This does
    not recover original vintages from a restated vendor snapshot.
    """
    f = fundamentals.loc[
        fundamentals.period.eq("quarter")
        & fundamentals.filing_date.gt(fundamentals.fiscal_date)
    ].copy()
    f["available_date"] = f.filing_date + pd.Timedelta(days=1 + extra_lag_days)
    assets = f.total_assets.where(f.total_assets.gt(0))
    revenue = f.revenue.where(f.revenue.gt(0))
    f["net_margin"] = f.net_income / revenue
    f["operating_roa"] = f.operating_income / assets
    f["debt_to_assets"] = f.total_debt / assets
    f["cash_to_assets"] = f.cash_and_equivalents / assets
    f["fcf_to_assets"] = f.free_cash_flow / assets
    f["accruals_to_assets"] = (f.net_income - f.operating_cash_flow) / assets
    f[FINANCIAL_FEATURES] = f[FINANCIAL_FEATURES].replace([np.inf, -np.inf], np.nan)
    f = f.sort_values(["symbol", "available_date", "fiscal_date"])
    # An old fiscal period filed late must not replace a newer known period.
    latest_fiscal = f.groupby("symbol").fiscal_date.cummax()
    f = f.loc[f.fiscal_date.eq(latest_fiscal)].drop_duplicates(["symbol", "available_date"], keep="last")
    return f[["symbol", "fiscal_date", "filing_date", "available_date"] + FINANCIAL_FEATURES]

def join_financials(panel, events, universe):
    """Backward as-of join within each stock, retaining unavailable features."""
    frames = []
    for symbol, group in panel.groupby("symbol", sort=True):
        right = events.loc[events.symbol.eq(symbol)].drop(columns="symbol").sort_values("available_date")
        joined = pd.merge_asof(group.sort_values("date"), right,
                               left_on="date", right_on="available_date", direction="backward")
        joined["filing_age_days"] = (joined.date - joined.available_date).dt.days
        stale = (joined.date - joined.fiscal_date).dt.days.gt(550)
        joined.loc[stale, FINANCIAL_FEATURES] = np.nan
        frames.append(joined)
    result = pd.concat(frames, ignore_index=True).merge(universe, on="symbol", validate="many_to_one")
    result["sector"] = result.sector.fillna("Unknown")
    assert (result.available_date.isna() | result.available_date.le(result.date)).all()
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def sector_financial_ranks(panel, minimum_sector_size=5):
    """Rank four ratios within date/sector among eligible stocks only.

    Fall back to the whole eligible date when sector size is below five or
    sector is unknown. Size counts eligible stocks, including missing ratios.
    Missing ratios remain missing. Preserve raw ratios in *_raw columns.
    """
    result = panel.copy()
    eligible = result.loc[result.eligible]
    sizes = eligible.groupby(["date", "sector"], dropna=False).symbol.transform("size")
    fallback = sizes.lt(minimum_sector_size) | eligible.sector.isna() | eligible.sector.eq("Unknown")
    result["financial_rank_fallback"] = False
    result.loc[eligible.index, "financial_rank_fallback"] = fallback
    for name in D_RATIOS:
        result[name + "_raw"] = result[name]
        local = eligible.groupby(["date", "sector"], dropna=False)[name].rank(pct=True)
        global_rank = eligible.groupby("date")[name].rank(pct=True)
        result[name] = np.nan
        result.loc[eligible.index, name] = local.where(~fallback, global_rank)
    return result


def build_financial_panel(panel, database, end="2015-12-31"):
    """Read published reports only, apply course helpers, then sector ranks.

    Respects the cleaned available_date when it is later than filing+1 day.
    No currency conversion is substituted for these same-statement ratios.
    """
    if pd.Timestamp(end) > pd.Timestamp("2015-12-31"):
        raise ValueError("Development cutoff is 2015-12-31.")
    with duckdb.connect(str(database), read_only=True) as con:
        reports = con.execute("SELECT * FROM fundamentals WHERE filing_date < ? AND available_date <= ?", [end, end]).fetchdf()
    for name in ["fiscal_date", "filing_date", "available_date"]:
        reports[name] = pd.to_datetime(reports[name]).astype("datetime64[ns]")
    events = financial_events(reports)
    availability = reports[["symbol", "fiscal_date", "available_date"]].rename(columns={"available_date": "cleaned_available_date"})
    events = events.merge(availability, on=["symbol", "fiscal_date"], validate="one_to_one")
    events["available_date"] = events[["available_date", "cleaned_available_date"]].max(axis=1)
    events = events.drop(columns="cleaned_available_date")
    with duckdb.connect(str(database), read_only=True) as con:
        universe = con.execute("SELECT symbol, sector FROM universe").fetchdf()
    joined = join_financials(panel, events, universe)
    return sector_financial_ranks(joined)


def process_peak_memory_mib():
    """Peak resident process memory, including native array and database memory."""
    import sys
    if sys.platform != 'win32':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
            ['peak','current','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak / 1024**2


def build_full_development_cache(database, cache_path, lookback_start='2000-01-01',
                                 train_start='2010-01-01', end='2015-12-31', group_size=50,
                                 allow_final_test=False):
    """Apply the existing A/D functions in stock groups, then rank by whole date.

    This changes storage only. Use full stock histories for rolling features.
    Keep the original zero-delay target. Retain only eligible model rows from
    train_start after computing features. Keep all entry-date volatility rows
    separately. Rank each full month's eligible cross-sections with the SAME
    sector_financial_ranks function; never rank within a stock group.
    All source queries are bounded before materialization. Snapshot sectors
    remain the same documented source limitation as the class experiments.

    The development guard stays on by default. Only run_final_test.py passes
    allow_final_test=True, after the strategy is fixed, to build the same
    features through the final-test end date (never later than 2026-09-30).
    """
    import gc
    from time import perf_counter
    if pd.Timestamp(end) > pd.Timestamp('2015-12-31') and not allow_final_test:
        raise ValueError('Development ends on 2015-12-31.')
    if pd.Timestamp(end) > pd.Timestamp('2026-09-30'):
        raise ValueError('The homework data ends on 2026-09-30.')
    if Path(database).resolve() == Path(cache_path).resolve():
        raise ValueError('The cache must not replace the source.')
    started = perf_counter()
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database), read_only=True) as src, duckdb.connect(str(cache_path)) as dst:
        dst.execute("SET memory_limit='384MB'")
        dst.execute('SET threads=2')
        calendar = pd.DatetimeIndex(src.execute('SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date', [lookback_start,end]).fetchdf().date).astype('datetime64[ns]')
        membership = src.execute('SELECT symbol,quarter_start,selection_date,universe_rank FROM universe_membership WHERE quarter_start<=? AND selection_date<=?', [end,end]).fetchdf()
        for name in ['quarter_start','selection_date']:
            membership[name] = pd.to_datetime(membership[name]).astype('datetime64[ns]')
        universe = src.execute('SELECT symbol,sector FROM universe').fetchdf()
        reports = src.execute('SELECT * FROM fundamentals WHERE fiscal_date<=? AND filing_date<? AND available_date<=?', [end,end,end]).fetchdf()
        for name in ['fiscal_date','filing_date','available_date']:
            reports[name] = pd.to_datetime(reports[name]).astype('datetime64[ns]')
        events = financial_events(reports)
        available = reports[['symbol','fiscal_date','available_date']].rename(columns={'available_date':'cleaned_available_date'})
        events = events.merge(available, on=['symbol','fiscal_date'], validate='one_to_one')
        events['available_date'] = events[['available_date','cleaned_available_date']].max(axis=1)
        events = events.drop(columns='cleaned_available_date')
        del reports, available
        symbols = src.execute('SELECT DISTINCT symbol FROM prices WHERE date BETWEEN ? AND ? ORDER BY symbol', [lookback_start,end]).fetchnumpy()['symbol'].tolist()
        for table in ['A','D','D_raw','entry_volatility','calendar']:
            dst.execute(f'DROP TABLE IF EXISTS {table}')
        created = set()
        def save(table, frame):
            dst.register('part', frame)
            if table not in created:
                dst.execute(f'CREATE TABLE {table} AS SELECT * FROM part')
                created.add(table)
            else:
                dst.execute(f'INSERT INTO {table} BY NAME SELECT * FROM part')
            dst.unregister('part')
        save('calendar', pd.DataFrame({'date':calendar}))
        read_rows = 0
        for start in range(0,len(symbols),group_size):
            group = symbols[start:start+group_size]
            prices = src.execute('SELECT symbol,date,adj_close FROM prices WHERE date BETWEEN ? AND ? AND symbol IN (SELECT unnest(?)) ORDER BY symbol,date', [lookback_start,end,group]).fetchdf()
            prices['date'] = pd.to_datetime(prices.date).astype('datetime64[ns]')
            read_rows += len(prices)
            features = price_features(prices, calendar)
            features = features.loc[features.date.ge(train_start)].copy()
            save('entry_volatility', features[['date','symbol','volatility_63']])
            panel = apply_membership(features, membership, calendar)
            panel = make_labels(panel.loc[panel.eligible], prices, calendar, holding_days=90, max_exit_delay=0)
            columns = ['date','symbol','eligible','target','label_end'] + PRICE_FEATURES
            panel = panel[columns]
            save('A', panel)
            financial = join_financials(panel, events, universe)
            save('D_raw', financial[columns + D_RATIOS + ['filing_age_days','sector','available_date']])
            del prices, features, panel, financial
            gc.collect()
        for month in pd.period_range(train_start,end,freq='M'):
            raw = dst.execute('SELECT * FROM D_raw WHERE date BETWEEN ? AND ? ORDER BY date,symbol', [month.start_time, min(month.end_time,pd.Timestamp(end))]).fetchdf()
            if raw.empty:
                continue
            ranked = sector_financial_ranks(raw)
            save('D', ranked.drop(columns=[name+'_raw' for name in D_RATIOS]))
            del raw, ranked
        dst.execute('DROP TABLE D_raw')
        counts = {name:dst.execute(f'SELECT count(*) FROM {name}').fetchone()[0] for name in ['A','D']}
        assert counts['A']==counts['D']
        assert dst.execute('SELECT max(date) FROM A').fetchone()[0] <= pd.Timestamp(end)
        dst.execute('CHECKPOINT')
    return {'price_rows_read':read_rows,'stocks_with_pre_cutoff_prices':len(symbols),
            'eligible_model_rows':counts['A'],'seconds':perf_counter()-started,
            'peak_process_memory_mib':process_peak_memory_mib()}


def fit_cached_development(cache_path, run, features, years=(2013,2014,2015),
                           train_start='2010-01-01', alpha=100.0, clip_target=False):
    """Fit the original annual pipeline on bounded, completed cache targets."""
    from time import perf_counter
    if run not in ('A','D') or any(year>2015 for year in years):
        raise ValueError('Use development runs A/D and years through 2015.')
    forecasts, scores, splits, coefficients = [], [], [], []
    with duckdb.connect(str(cache_path),read_only=True) as con:
        for year in years:
            started = perf_counter()
            signals = con.execute(f'SELECT * FROM {run} WHERE date BETWEEN ? AND ? ORDER BY date,symbol', [f'{year}-01-01',f'{year}-12-31']).fetchdf()
            cutoff = signals.date.min()
            train = con.execute(f'SELECT * FROM {run} WHERE date>=? AND date<? AND label_end<? AND isfinite(target) ORDER BY date,symbol', [train_start,cutoff,cutoff]).fetchdf()
            assert train.label_end.max()<cutoff
            model = make_ridge_pipeline(alpha=alpha,feature_names=features)
            training_target, clip_audit = clip_training_target(train.target) if clip_target else (train.target, {})
            model.fit(train[features],training_target)
            forecast = signals[['date','symbol','target','label_end','volatility_63']].copy()
            forecast['prediction'] = model.predict(signals[features])
            forecast['validation_year'] = year
            forecasts.append(forecast)
            scored = forecast.loc[forecast.target.notna()]
            scores.append({'model':run,'year':year,**prediction_metrics(scored,scored.prediction).to_dict(),
                           'scored_rows':len(scored),'forecast_rows':len(forecast)})
            splits.append({'model':run,'year':year,'training_rows':len(train),
                           'first_signal':cutoff,'last_training_outcome':train.label_end.max(),
                           'seconds':perf_counter()-started, **clip_audit})
            names = model.named_steps['preprocess'].get_feature_names_out()
            coefficients.extend({'model':run,'year':year,'term':name,'coefficient':float(value)}
                                for name,value in zip(names,model.named_steps['model'].coef_))
            coefficients.append({'model':run,'year':year,'term':'intercept','coefficient':float(model.named_steps['model'].intercept_)})
    return pd.concat(forecasts,ignore_index=True),pd.DataFrame(scores),pd.DataFrame(splits),pd.DataFrame(coefficients)


def development_coverage(cache_path, run, features, start='2010-01-01', end='2015-12-31'):
    """Coverage uses eligible stock-date rows, including unscored forecasts."""
    if run not in ('A','D') or pd.Timestamp(end)>pd.Timestamp('2015-12-31'):
        raise ValueError('Use bounded development cache rows.')
    missing = []
    with duckdb.connect(str(cache_path),read_only=True) as con:
        for feature in features:
            count, total = con.execute(f'SELECT count(*) FILTER(WHERE "{feature}" IS NULL),count(*) FROM {run} WHERE date BETWEEN ? AND ?', [start,end]).fetchone()
            missing.append({'feature':feature,'missing_rows':count,'eligible_rows':total,'missing_share':count/total})
        eligible = con.execute(f'SELECT date,count(*) eligible_stocks FROM {run} WHERE date BETWEEN ? AND ? GROUP BY date ORDER BY date', [start,end]).fetchdf()
        sessions = con.execute('SELECT date FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date', [start,end]).fetchdf()
        eligible = sessions.merge(eligible,on='date',how='left').fillna({'eligible_stocks':0})
        eligible['eligible_stocks'] = eligible.eligible_stocks.astype(int)
        fallback = con.execute('SELECT year(date) AS year,sum(CAST(financial_rank_fallback AS INT)) fallback_stock_dates,count(*) eligible_stock_dates,avg(CAST(financial_rank_fallback AS INT)) fallback_share FROM D WHERE date BETWEEN ? AND ? GROUP BY year(date) ORDER BY year(date)', [start,end]).fetchdf() if run=='D' else pd.DataFrame()
    summary = eligible.eligible_stocks.agg(['min','median','max']).rename('eligible_stocks').to_frame()
    return pd.DataFrame(missing),eligible,summary,fallback


def no_quote_diagnostics(account, stressed_account):
    """Flagged values are lot-session marks, not distinct capital lost.

    Stress reruns the same forecasts with zero recovery. Later allocations
    respond to changed cash/NAV. Also show a fixed-ledger net-proceeds deduction.
    """
    flags = account.valuation_flags
    exits = account.trades.loc[account.trades.side.eq('SELL') & account.trades.no_quote_exit]
    return pd.Series({'flagged_lot_valuations':len(flags),
        'flagged_stock_sessions':len(flags[['date','symbol']].drop_duplicates()),
        'flagged_sessions':flags.date.nunique(), 'flagged_mark_value_sum':flags.market_value.sum(),
        'maximum_flagged_value_on_one_session':flags.groupby('date').market_value.sum().max() if len(flags) else 0,
        'no_quote_exit_count':len(exits), 'no_quote_exit_gross_value':exits.notional.sum(),
        'no_quote_exit_net_proceeds':exits.cash_change.sum(),
        'zero_recovery_final_value':stressed_account.daily.nav.iloc[-1],
        'fixed_ledger_zero_recovery_value':account.daily.nav.iloc[-1]-exits.cash_change.sum()})


def rank_ic_stability(scores):
    """Define steadier before evaluation: smaller sample std of yearly mean IC."""
    return scores.groupby('model').mean_daily_rank_IC.agg(['mean','std','min','max'])


def cached_trade_errors(cache_path, account, forecasts, benchmark):
    """Load only signal/entry volatility keys needed by the closed-trade report."""
    from strategy_backtest import closed_trade_errors
    closed = account.trades.loc[account.trades.side.eq('SELL')]
    keys = pd.concat([closed[['symbol',c]].rename(columns={c:'date'})
                      for c in ['signal_date','entry_date']],ignore_index=True).drop_duplicates()
    with duckdb.connect(str(cache_path),read_only=True) as con:
        con.register('needed_keys',keys)
        panel = con.execute('SELECT v.* FROM entry_volatility v JOIN needed_keys k USING(symbol,date)').fetchdf()
    errors = closed_trade_errors(account,forecasts,panel,benchmark)
    return errors.merge(closed[['lot_id','no_quote_exit','price_date']],on='lot_id',validate='one_to_one')


def largest_completed_training_targets(cache_path, train_start='2010-01-01', year=2013):
    """Inspect existing targets before a fit. Do not filter or alter training rows."""
    if year > 2015:
        raise ValueError('This audit is restricted to development years.')
    with duckdb.connect(str(cache_path),read_only=True) as con:
        cutoff = con.execute("SELECT min(date) FROM A WHERE date BETWEEN ? AND ?",[f'{year}-01-01',f'{year}-12-31']).fetchone()[0]
        return con.execute('SELECT symbol,date,label_end,target FROM A WHERE date>=? AND date<? AND label_end<? AND isfinite(target) ORDER BY target DESC LIMIT 20', [train_start,cutoff,cutoff]).fetchdf()


def clip_training_target(training_target):
    """Fit 1%/99% bounds on this fit's completed training targets only.

    Return a new Series. Never overwrite raw targets, validation outcomes, or
    execution prices. Use linear-interpolated empirical quantiles (pandas default).
    """
    if len(training_target)==0 or not np.isfinite(training_target).all():
        raise ValueError('Require nonempty finite completed training targets.')
    lower, upper = training_target.quantile([.01,.99]).to_numpy()
    below, above = training_target.lt(lower), training_target.gt(upper)
    return training_target.clip(lower,upper), {
        'target_clip_lower':float(lower),'target_clip_upper':float(upper),
        'target_rows_clipped_low':int(below.sum()),'target_rows_clipped_high':int(above.sum()),
        'target_rows_clipped':int((below|above).sum())}


def signal_volatility_screen(eligible_forecasts, quantile=.80):
    """Annotate only the supplied eligible cross-section on each signal date.

    Keep values equal to the threshold. Do not use future dates or outcomes.
    Training rows and held lots are not screened. Nonfinite volatility cannot
    enter a purchase. Call after entry-quarter membership eligibility is known.
    """
    if not 0 < quantile <= 1:
        raise ValueError('Use a quantile in (0,1].')
    result = eligible_forecasts.copy()
    volatility = result.volatility_63.where(np.isfinite(result.volatility_63))
    result['volatility_threshold'] = volatility.groupby(result.date).transform('quantile',q=quantile)
    result['passes_volatility_screen'] = volatility.notna() & volatility.le(result.volatility_threshold)
    return result


def forecast_decile_report(forecasts, screened=False):
    """Rank forecasts on each date before checking realized target availability.

    Decile 1 is lowest; decile 10 is highest. Break forecast ties by symbol.
    Use qcut on first ranks for ten near-equal groups. Return equal-date-weighted
    yearly averages of within-date means, pooled stock-date means, and the mean
    matched-date 10-minus-1 spread. All realized targets remain UNCLIPPED.
    Daily signals overlap over their 90-day horizons; these are not independent
    trades. The candidate view applies the same signal-volatility screen first.
    """
    frame = signal_volatility_screen(forecasts)
    if screened:
        frame = frame.loc[frame.passes_volatility_screen].copy()
    sizes = frame.groupby('date').symbol.transform('size')
    frame = frame.loc[sizes.ge(10)].sort_values(['date','prediction','symbol']).copy()
    frame['decile'] = frame.groupby('date').prediction.transform(
        lambda x: pd.qcut(x.rank(method='first'),10,labels=False)+1).astype(int)
    scored = frame.loc[np.isfinite(frame.target)].copy()
    scored['year'] = scored.date.dt.year
    daily = scored.groupby(['year','date','decile']).target.agg(['mean','count']).reset_index()
    yearly = daily.groupby(['year','decile']).agg(mean_realized_return=('mean','mean'),
        scored_dates=('date','size'),scored_rows=('count','sum')).reset_index()
    pooled = scored.groupby(['year','decile']).target.mean().rename('pooled_mean_realized_return').reset_index()
    yearly = yearly.merge(pooled,on=['year','decile'],validate='one_to_one')
    wide = daily.pivot(index=['year','date'],columns='decile',values='mean').reindex(columns=range(1,11))
    spreads = (wide[10]-wide[1]).dropna().rename('top_minus_bottom').reset_index()
    spread = spreads.groupby('year').agg(top_minus_bottom=('top_minus_bottom','mean'),
        matched_dates=('date','size')).reset_index()
    return yearly, spread


def forecast_range_report(forecasts):
    """Audit raw model outputs without imposing output bounds or hiding extremes."""
    annotated = signal_volatility_screen(forecasts)
    ranges = annotated.groupby('validation_year').prediction.agg(
        count='size',minimum='min',mean='mean',maximum='max',
        p01=lambda x:x.quantile(.01),median='median',p99=lambda x:x.quantile(.99),
        below_minus_100_percent=lambda x:int(x.lt(-1).sum()),
        above_plus_100_percent=lambda x:int(x.gt(1).sum())).reset_index()
    largest = annotated.sort_values(['prediction','date','symbol'],ascending=[False,True,True]).head(20)
    largest_candidates = annotated.loc[annotated.passes_volatility_screen].sort_values(
        ['prediction','date','symbol'],ascending=[False,True,True]).head(20)
    return ranges,largest,largest_candidates


def suspect_price_flags(prices, calendar, rows):
    """Flag extreme adjacent-session moves without changing prices or targets.

    A ratio strictly outside [0.2, 5] is suspect. Never bridge missing quotes.
    Exclude candidates on the jump session and the following 252 sessions.
    The feature window includes today and the preceding 252 sessions. Label
    flags inspect next-session entry through label_end, inclusive. They are
    outcome-quality flags ONLY: never use them to filter live candidates.
    Pass bounded prices/calendar; incomplete labels stay unscored as before.
    """
    calendar = pd.DatetimeIndex(calendar)
    groups = prices.groupby('symbol',sort=False)
    parts = []
    for symbol, group in rows.groupby('symbol',sort=False):
        quotes = groups.get_group(symbol).set_index('date').adj_close.reindex(calendar)
        ratio = quotes / quotes.shift(1)
        jumps = ratio.lt(.2) | ratio.gt(5)
        recent = jumps.rolling(253,min_periods=1).max().astype(bool)
        result = group[['date','symbol']].copy()
        result['suspect_lookback'] = result.date.map(recent).to_numpy()
        result['suspect_label'] = False
        completed = group.label_end.notna()
        entry_pos = calendar.get_indexer(group.loc[completed,'date']) + 1
        end_pos = calendar.searchsorted(group.loc[completed,'label_end'],side='right')
        cumulative = np.r_[0, jumps.to_numpy(dtype=np.int64).cumsum()]
        result.loc[completed,'suspect_label'] = cumulative[end_pos] - cumulative[entry_pos] > 0
        parts.append(result)
    return pd.concat(parts).sort_index()
