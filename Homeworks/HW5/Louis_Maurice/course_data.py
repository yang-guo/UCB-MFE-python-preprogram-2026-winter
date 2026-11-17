"""Small, inspectable data helpers shared by Lectures 7 and 8.

Historical features use only past/current observations. Future prices appear
only in make_labels; never pass its audit columns to a model or allocator.
"""

from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

PRICE_FEATURES = ["momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252"]
FINANCIAL_FEATURES = ["net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets", "accruals_to_assets"]
NUMERIC_FEATURES = PRICE_FEATURES + FINANCIAL_FEATURES + ["filing_age_days"]
FEATURES = NUMERIC_FEATURES + ["sector"]


def load_data(database=None, end="2015-12-31", symbols=None):
    """Select columns/dates and optional ticker batches in read-only SQL."""
    database = Path(database) if database else Path(__file__).parent / "data/equities_class.duckdb"
    if not database.is_file():
        raise FileNotFoundError(f"Download the modeling database from bCourses to {database}")
    symbol_filter = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    symbol_params = [list(symbols)] if symbols is not None else []
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute(f"""SELECT symbol,date,close,adj_close,volume FROM prices
            WHERE date<=? {symbol_filter} ORDER BY symbol,date""", [end] + symbol_params).fetchdf()
        fundamentals = con.execute(f"""SELECT symbol,fiscal_date,period,filing_date,
            reported_currency,revenue,net_income,operating_income,total_assets,
            total_debt,cash_and_equivalents,free_cash_flow,operating_cash_flow
            FROM fundamentals WHERE fiscal_date<=? AND (filing_date<=? OR filing_date IS NULL)
            {symbol_filter} ORDER BY symbol,fiscal_date""", [end, end] + symbol_params).fetchdf()
        universe = con.execute(f"SELECT symbol,sector FROM universe WHERE true {symbol_filter} ORDER BY symbol", symbol_params).fetchdf()
        benchmark = con.execute("SELECT date,adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
    for frame, columns in [(prices, ["date"]), (fundamentals, ["fiscal_date", "filing_date"]), (benchmark, ["date"])]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    assert not prices.duplicated(["symbol", "date"]).any()
    invalid_price = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
    if invalid_price.any():
        warnings.warn(f"Quarantining {int(invalid_price.sum())} invalid adjusted-price rows; inspect the raw database and delivery audit.", stacklevel=2)
        prices = prices.loc[~invalid_price].copy()
    assert not benchmark.date.duplicated().any()
    assert (np.isfinite(benchmark.adj_close) & benchmark.adj_close.gt(0)).all()
    off_calendar = ~prices.date.isin(benchmark.date)
    if off_calendar.any():
        warnings.warn(f"Quarantining {int(off_calendar.sum())} price rows outside the supplied market calendar; inspect the raw database and delivery audit.", stacklevel=2)
        prices = prices.loc[~off_calendar].copy()
    return prices, fundamentals, universe, benchmark


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


def load_membership(database=None, end="2015-12-31"):
    """Quarterly membership is a dated decision input, not the static stock pool."""
    database = Path(database) if database else Path(__file__).parent / "data/equities_class.duckdb"
    with duckdb.connect(str(database), read_only=True) as con:
        membership = con.execute("""SELECT symbol,quarter_start,quarter_end,selection_date,universe_rank
            FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start,universe_rank""", [end]).fetchdf()
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    return membership


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


def make_labels(panel, prices, calendar, holding_days=90, max_exit_delay=5):
    """Next-session entry to first observed close on/after 90 calendar days.

    Entry must have a quote on the next *market* session. Missing exits may be
    delayed at most five market sessions. Censored rows keep missing targets;
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


def prepare_panel(database=None, end="2015-12-31"):
    prices, fundamentals, universe, benchmark = load_data(database, end=end)
    panel = price_features(prices, benchmark.date)
    panel = join_financials(panel, financial_events(fundamentals), universe)
    panel = apply_membership(panel, load_membership(database, end=end), benchmark.date)
    return make_labels(panel, prices, benchmark.date)


def date_folds(frame, validation_years=(2013, 2014), train_start="2010-01-01"):
    """Return positional sklearn indices with whole dates and mature labels.

    Fold model is fitted immediately before the validation year's first signal.
    Thus every training outcome must end strictly before that signal date.
    """
    splits = []
    for year in validation_years:
        valid = frame.date.between(f"{year}-01-01", f"{year}-12-31") & frame.target.notna()
        if not valid.any():
            raise ValueError(f"No validation rows for {year}")
        first_signal = frame.loc[valid, "date"].min()
        train = (frame.date.ge(train_start) & frame.date.lt(first_signal)
                 & frame.label_end.lt(first_signal) & frame.target.notna())
        tr, va = np.flatnonzero(train), np.flatnonzero(valid)
        if not len(tr):
            raise ValueError(f"No matured training labels before {year}")
        assert frame.iloc[tr].label_end.max() < frame.iloc[va].date.min()
        assert not set(frame.iloc[tr].date).intersection(frame.iloc[va].date)
        splits.append((tr, va))
    return splits


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


def make_pipeline(model=None, numeric_features=None, include_sector=True):
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    numeric = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transforms = [("numeric", numeric, numeric_features)]
    if include_sector:
        transforms.append(("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]))
    return Pipeline([("preprocess", ColumnTransformer(transforms, remainder="drop")),
                     ("model", Ridge(alpha=100) if model is None else model)])


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
