"""HW4 section 2: predict the 90-calendar-day holding return.

Signal features use only information known at that session's close. The label
is the return from the next session's close to the first quoted close on or
after 90 calendar days. Training rows are monthly, and a fit may use a label
only after that holding period has ended.

Lecture helpers reused from Lectures/modeling/course_data.py:
QuantileClipper, make_pipeline, and the filing-date / membership timing rules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "Lectures" / "modeling"))

import course_data  # noqa: E402

DATA_DIR = REPO / "Lectures" / "modeling" / "data"
FULL_DB = DATA_DIR / "equities_top2000.duckdb"
CLASS_DB = DATA_DIR / "equities_class.duckdb"
OUTPUTS = HERE / "outputs"
RESULTS = HERE / "results"
PANEL_PATH = OUTPUTS / "panel.parquet"
CHOICE_PATH = RESULTS / "chosen_model.json"

TRAIN_START = pd.Timestamp("2010-01-01")
DEV_END = pd.Timestamp("2015-12-31")
BUFFER_END = pd.Timestamp("2016-03-31")
TEST_START = pd.Timestamp("2016-04-01")
TEST_END = pd.Timestamp("2026-09-30")
HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5
STALE_FILING_DAYS = 550
SELECTION_YEARS = (2013, 2014)
CONFIRM_YEAR = 2015

# Price and filing features are known at the signal close. Peer features are
# computed later from that same close's eligible cross-section.
BASE_NUMERIC = [
    "momentum_5",
    "momentum_21",
    "momentum_63",
    "momentum_126",
    "momentum_252_21",
    "volatility_21",
    "volatility_63",
    "drawdown_252",
    "volume_ratio_21",
    "large_cap_score",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
    "market_momentum_63",
    "market_volatility_63",
]
PEER_NUMERIC = ["sector_relative_momentum", "sector_profitability_rank"]
NUMERIC_FEATURES = BASE_NUMERIC + PEER_NUMERIC
MODEL_COLUMNS = NUMERIC_FEATURES + ["sector"]

# Tree counts are fixed here. sklearn's early stopping would hold out a random
# slice of the training rows, including later dates, so the year folds decide
# the size instead.
CONFIGS = {
    "shallow": dict(
        learning_rate=0.08,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=500,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=42,
    ),
    "medium": dict(
        learning_rate=0.06,
        max_iter=100,
        max_leaf_nodes=31,
        min_samples_leaf=250,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    ),
    "deep": dict(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=42,
    ),
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def connect(database):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{OUTPUTS / 'duckdb_tmp'}'")
    con.execute("SET memory_limit = '6GB'")
    con.execute("SET threads = 4")
    con.execute("ATTACH ? AS src (READ_ONLY)", [str(database)])
    return con


def build_panel(database, price_end, export_start, out_path=None, eligible_only=True):
    """Build the signal panel and optionally write eligible rows to parquet.

    Windows and filing joins are trailing. `price_end` drops later prices so a
    development check cannot see the final-test sample.
    """
    price_end = pd.Timestamp(price_end)
    export_start = pd.Timestamp(export_start)
    log(f"Building panel from {database.name} through {price_end.date()}")
    con = connect(database)
    con.execute(
        """
        CREATE TEMP TABLE cal AS
        SELECT CAST(date AS TIMESTAMP) AS date,
               row_number() OVER (ORDER BY date) - 1 AS cal_idx
        FROM src.benchmark
        WHERE date <= ?
        ORDER BY date
        """,
        [price_end],
    )
    con.execute(
        """
        CREATE TEMP TABLE px AS
        SELECT p.symbol, c.date, c.cal_idx, p.adj_close, p.volume
        FROM src.prices p
        JOIN cal c ON c.date = CAST(p.date AS TIMESTAMP)
        WHERE p.date <= ?
          AND p.adj_close > 0
          AND isfinite(p.adj_close)
        """,
        [price_end],
    )
    n_px = con.execute("SELECT count(*) FROM px").fetchone()[0]
    log(f"Clean price rows: {n_px:,}")

    con.execute(
        """
        CREATE TEMP TABLE dense AS
        WITH bounds AS (
            SELECT symbol, min(cal_idx) AS i0, max(cal_idx) AS i1
            FROM px GROUP BY symbol
        )
        SELECT b.symbol, c.cal_idx, c.date, p.adj_close, p.volume
        FROM bounds b
        JOIN cal c ON c.cal_idx BETWEEN b.i0 AND b.i1
        LEFT JOIN px p ON p.symbol = b.symbol AND p.cal_idx = c.cal_idx
        """
    )
    log(f"Calendar-aligned rows: {con.execute('SELECT count(*) FROM dense').fetchone()[0]:,}")

    con.execute(
        """
        CREATE TEMP TABLE feat AS
        WITH lagged AS (
            SELECT symbol, cal_idx, date, adj_close, volume,
                (adj_close / lag(adj_close, 1) OVER w) - 1 AS daily_ret,
                (adj_close / lag(adj_close, 5) OVER w) - 1 AS momentum_5,
                (adj_close / lag(adj_close, 21) OVER w) - 1 AS momentum_21,
                (adj_close / lag(adj_close, 63) OVER w) - 1 AS momentum_63,
                (adj_close / lag(adj_close, 126) OVER w) - 1 AS momentum_126,
                (lag(adj_close, 21) OVER w / lag(adj_close, 252) OVER w) - 1 AS momentum_252_21
            FROM dense
            WINDOW w AS (PARTITION BY symbol ORDER BY cal_idx)
        )
        SELECT *,
            CASE WHEN count(daily_ret) OVER w21 = 21
                 THEN stddev_samp(daily_ret) OVER w21 * sqrt(252) END AS volatility_21,
            CASE WHEN count(daily_ret) OVER w63 = 63
                 THEN stddev_samp(daily_ret) OVER w63 * sqrt(252) END AS volatility_63,
            CASE WHEN count(adj_close) OVER w252 = 252
                 THEN adj_close / (max(adj_close) OVER w252) - 1 END AS drawdown_252,
            CASE WHEN count(volume) OVER w21 = 21 AND avg(volume) OVER w21 > 0
                 THEN volume / avg(volume) OVER w21 - 1 END AS volume_ratio_21
        FROM lagged
        WINDOW
            w21 AS (PARTITION BY symbol ORDER BY cal_idx ROWS BETWEEN 20 PRECEDING AND CURRENT ROW),
            w63 AS (PARTITION BY symbol ORDER BY cal_idx ROWS BETWEEN 62 PRECEDING AND CURRENT ROW),
            w252 AS (PARTITION BY symbol ORDER BY cal_idx ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
        """
    )
    con.execute("DROP TABLE dense")

    con.execute(
        """
        CREATE TEMP TABLE bench_feat AS
        WITH aligned AS (
            SELECT c.date, c.cal_idx, b.adj_close
            FROM cal c
            LEFT JOIN src.benchmark b ON CAST(b.date AS TIMESTAMP) = c.date
        ),
        lagged AS (
            SELECT date,
                (adj_close / lag(adj_close, 1) OVER w) - 1 AS daily_ret,
                (adj_close / lag(adj_close, 63) OVER w) - 1 AS market_momentum_63
            FROM aligned
            WINDOW w AS (ORDER BY cal_idx)
        )
        SELECT date, market_momentum_63,
            CASE WHEN count(daily_ret) OVER w63 = 63
                 THEN stddev_samp(daily_ret) OVER w63 * sqrt(252) END AS market_volatility_63
        FROM lagged
        WINDOW w63 AS (ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW)
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE fin AS
        WITH raw AS (
            SELECT symbol,
                CAST(fiscal_date AS TIMESTAMP) AS fiscal_date,
                CAST(filing_date AS TIMESTAMP) AS filing_date,
                CAST(filing_date AS TIMESTAMP) + INTERVAL 1 DAY AS available_date,
                CASE WHEN revenue > 0 AND isfinite(revenue) AND isfinite(net_income)
                     THEN net_income / revenue END AS net_margin,
                CASE WHEN total_assets > 0 AND isfinite(total_assets) AND isfinite(operating_income)
                     THEN operating_income / total_assets END AS operating_roa,
                CASE WHEN total_assets > 0 AND isfinite(total_assets) AND isfinite(total_debt)
                     THEN total_debt / total_assets END AS debt_to_assets,
                CASE WHEN total_assets > 0 AND isfinite(total_assets) AND isfinite(cash_and_equivalents)
                     THEN cash_and_equivalents / total_assets END AS cash_to_assets,
                CASE WHEN total_assets > 0 AND isfinite(total_assets) AND isfinite(free_cash_flow)
                     THEN free_cash_flow / total_assets END AS fcf_to_assets,
                CASE WHEN total_assets > 0 AND isfinite(total_assets)
                      AND isfinite(net_income) AND isfinite(operating_cash_flow)
                     THEN (net_income - operating_cash_flow) / total_assets END AS accruals_to_assets
            FROM src.fundamentals
            WHERE period = 'quarter'
              AND filing_date IS NOT NULL
              AND fiscal_date IS NOT NULL
              AND filing_date > fiscal_date
              AND fiscal_date <= ?
              AND filing_date <= ?
        ),
        ordered AS (
            SELECT *,
                max(fiscal_date) OVER (
                    PARTITION BY symbol
                    ORDER BY available_date, fiscal_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS latest_fiscal
            FROM raw
        ),
        fresh AS (
            SELECT * EXCLUDE (latest_fiscal)
            FROM ordered
            WHERE fiscal_date = latest_fiscal
        )
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT *,
                row_number() OVER (
                    PARTITION BY symbol, available_date
                    ORDER BY fiscal_date DESC
                ) AS rn
            FROM fresh
        )
        WHERE rn = 1
        """,
        [price_end, price_end],
    )

    con.execute(
        """
        CREATE TEMP TABLE mem AS
        SELECT symbol,
               CAST(quarter_start AS TIMESTAMP) AS quarter_start,
               CAST(selection_date AS TIMESTAMP) AS selection_date,
               universe_rank,
               count(*) OVER (PARTITION BY quarter_start) AS quarter_n
        FROM src.universe_membership
        """
    )
    dupes = con.execute(
        """
        SELECT count(*) FROM (
            SELECT symbol, quarter_start FROM mem
            GROUP BY 1, 2 HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if dupes:
        raise ValueError(f"Duplicate quarterly membership keys: {dupes}")

    con.execute(
        """
        CREATE TEMP TABLE month_starts AS
        SELECT min(date) AS date
        FROM cal
        GROUP BY date_trunc('month', date)
        """
    )

    # Entry is the next market session. Membership is the quarter of that entry.
    # The exit is the first later quote on or after entry + 90 calendar days,
    # and only if that quote falls within five market sessions of the due date.
    con.execute(
        f"""
        CREATE TEMP TABLE labeled AS
        WITH quotes AS (
            SELECT symbol, cal_idx, date AS quote_date, adj_close AS quote_px
            FROM feat
            WHERE adj_close IS NOT NULL
        ),
        with_entry AS (
            SELECT f.symbol, f.date AS signal_date, f.cal_idx,
                f.momentum_5, f.momentum_21, f.momentum_63, f.momentum_126, f.momentum_252_21,
                f.volatility_21, f.volatility_63, f.drawdown_252, f.volume_ratio_21,
                c.date AS entry_date,
                q.quote_px AS entry_price,
                c.date + INTERVAL {HOLDING_DAYS} DAY AS scheduled_exit,
                CAST(date_trunc('quarter', c.date) AS TIMESTAMP) AS membership_quarter
            FROM feat f
            JOIN cal c ON c.cal_idx = f.cal_idx + 1
            LEFT JOIN quotes q ON q.symbol = f.symbol AND q.cal_idx = c.cal_idx
            WHERE f.adj_close IS NOT NULL
        ),
        with_exit AS (
            SELECT e.*, x.quote_date AS exit_date, x.quote_px AS exit_price, x.cal_idx AS exit_idx
            FROM with_entry e
            ASOF LEFT JOIN quotes x
                ON e.symbol = x.symbol AND e.scheduled_exit <= x.quote_date
        ),
        with_due AS (
            SELECT e.*, d.cal_idx AS due_idx
            FROM with_exit e
            ASOF LEFT JOIN cal d
                ON e.scheduled_exit <= d.date
        )
        SELECT symbol, signal_date AS date,
            momentum_5, momentum_21, momentum_63, momentum_126, momentum_252_21,
            volatility_21, volatility_63, drawdown_252, volume_ratio_21,
            entry_date, membership_quarter,
            CASE
                WHEN entry_price IS NOT NULL
                 AND scheduled_exit <= (SELECT max(date) FROM cal)
                 AND exit_price IS NOT NULL
                 AND exit_idx - due_idx <= {MAX_EXIT_DELAY}
                THEN exit_date
            END AS label_end,
            CASE
                WHEN entry_price IS NOT NULL
                 AND scheduled_exit <= (SELECT max(date) FROM cal)
                 AND exit_price IS NOT NULL
                 AND exit_idx - due_idx <= {MAX_EXIT_DELAY}
                THEN exit_price / entry_price - 1
            END AS target
        FROM with_due
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE panel AS
        WITH joined AS (
            SELECT s.*,
                m.selection_date,
                m.universe_rank,
                CASE WHEN m.universe_rank IS NOT NULL
                     THEN 1 - (m.universe_rank - 1) * 1.0 / m.quarter_n END AS large_cap_score,
                f.fiscal_date,
                f.available_date,
                f.net_margin, f.operating_roa, f.debt_to_assets, f.cash_to_assets,
                f.fcf_to_assets, f.accruals_to_assets,
                date_diff('day', CAST(f.available_date AS DATE), CAST(s.date AS DATE)) AS filing_age_days,
                date_diff('day', CAST(f.fiscal_date AS DATE), CAST(s.date AS DATE)) AS fiscal_age_days,
                b.market_momentum_63, b.market_volatility_63,
                coalesce(u.sector, 'Unknown') AS sector,
                ms.date IS NOT NULL AS is_month_start
            FROM labeled s
            LEFT JOIN mem m
                ON s.symbol = m.symbol AND s.membership_quarter = m.quarter_start
            ASOF LEFT JOIN fin f
                ON s.symbol = f.symbol AND s.date >= f.available_date
            LEFT JOIN bench_feat b ON b.date = s.date
            LEFT JOIN src.universe u ON u.symbol = s.symbol
            LEFT JOIN month_starts ms ON ms.date = s.date
        )
        SELECT symbol, date, sector, is_month_start,
            momentum_5, momentum_21, momentum_63, momentum_126, momentum_252_21,
            volatility_21, volatility_63, drawdown_252, volume_ratio_21, large_cap_score,
            CASE WHEN fiscal_age_days > {STALE_FILING_DAYS} THEN NULL ELSE net_margin END AS net_margin,
            CASE WHEN fiscal_age_days > {STALE_FILING_DAYS} THEN NULL ELSE operating_roa END AS operating_roa,
            CASE WHEN fiscal_age_days > {STALE_FILING_DAYS} THEN NULL ELSE debt_to_assets END AS debt_to_assets,
            CASE WHEN fiscal_age_days > {STALE_FILING_DAYS} THEN NULL ELSE cash_to_assets END AS cash_to_assets,
            CASE WHEN fiscal_age_days > {STALE_FILING_DAYS} THEN NULL ELSE fcf_to_assets END AS fcf_to_assets,
            CASE WHEN fiscal_age_days > {STALE_FILING_DAYS} THEN NULL ELSE accruals_to_assets END AS accruals_to_assets,
            filing_age_days, market_momentum_63, market_volatility_63,
            selection_date, available_date, fiscal_date, entry_date, label_end, target, universe_rank,
            volatility_63 IS NOT NULL AND universe_rank IS NOT NULL AS eligible
        FROM joined
        """
    )

    leaked = con.execute(
        """
        SELECT count(*) FROM panel
        WHERE (selection_date IS NOT NULL AND selection_date > date)
           OR (available_date IS NOT NULL AND available_date > date)
        """
    ).fetchone()[0]
    if leaked:
        raise ValueError(f"Features use information after the signal close: {leaked} rows")

    where = "date >= ?"
    params = [export_start]
    if eligible_only:
        where += " AND eligible"
    query = f"SELECT * FROM panel WHERE {where} ORDER BY date, symbol"
    n_out = con.execute(f"SELECT count(*) FROM panel WHERE {where}", params).fetchone()[0]
    log(f"Export rows: {n_out:,}")
    if out_path is None:
        frame = con.execute(query, params).fetchdf()
        con.close()
        return _normalize(frame)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    con.execute(f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)", params)
    con.close()
    log(f"Wrote {out_path}")
    return out_path


def _normalize(frame):
    for column in ["date", "selection_date", "available_date", "fiscal_date", "entry_date", "label_end"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    frame["sector"] = frame["sector"].fillna("Unknown").astype(str)
    frame["is_month_start"] = frame["is_month_start"].astype(bool)
    return frame


def read_panel(start, end, month_start_only=False):
    clauses = ["date >= ?", "date <= ?"]
    params = [pd.Timestamp(start), pd.Timestamp(end)]
    if month_start_only:
        clauses.append("is_month_start")
    query = f"SELECT * FROM read_parquet(?) WHERE {' AND '.join(clauses)} ORDER BY date, symbol"
    con = duckdb.connect()
    frame = con.execute(query, [str(PANEL_PATH), *params]).fetchdf()
    con.close()
    return _normalize(frame)


def assert_timing(frame, fit_boundary=None):
    """Fail if a feature, membership list, or training label arrived too late."""
    if frame.empty:
        raise ValueError("Empty frame")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol-date rows")
    if not frame["selection_date"].le(frame["date"]).all():
        raise ValueError("Universe selection date is after the signal")
    financials = frame["available_date"].isna() | frame["available_date"].le(frame["date"])
    if not financials.all():
        raise ValueError("A filing is used before its available date")
    if frame["entry_date"].notna().any() and not frame.loc[frame.entry_date.notna(), "entry_date"].gt(
        frame.loc[frame.entry_date.notna(), "date"]
    ).all():
        raise ValueError("Entry is not after the signal close")
    if fit_boundary is not None:
        fit_boundary = pd.Timestamp(fit_boundary)
        if not frame["label_end"].notna().all() or not frame["label_end"].lt(fit_boundary).all():
            raise ValueError("Training used a holding period that had not ended")
        if not frame["date"].lt(fit_boundary).all():
            raise ValueError("Training signal is on or after the fit boundary")


def add_peer_features(frame):
    """Same-day sector median and rank, using only stocks eligible on that date.

    Call this on the full cross-section of a date before dropping rows whose
    90-day outcome is still open. The outcome is not an input to these features.
    """
    out = frame.copy()
    grouped = out.groupby(["date", "sector"], sort=False)
    momentum_count = grouped["momentum_63"].transform("count")
    out["sector_relative_momentum"] = (
        out["momentum_63"] - grouped["momentum_63"].transform("median")
    ).where(momentum_count >= 3)
    profitability_count = grouped["operating_roa"].transform("count")
    out["sector_profitability_rank"] = grouped["operating_roa"].rank(method="average", pct=True).where(
        profitability_count >= 3
    )
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def make_tree(params):
    model = HistGradientBoostingRegressor(**params)
    numeric = course_data.Pipeline([("clip", course_data.QuantileClipper())])
    preprocess = course_data.ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("sector", course_data.OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
        ],
        remainder="drop",
    )
    return course_data.Pipeline([("preprocess", preprocess), ("model", model)])


def _date_weights(frame):
    return (1.0 / frame.groupby("date")["symbol"].transform("size")).to_numpy()


def fit_regressor(pipe, train):
    """Clip training targets at the training 1st/99th percentiles, then fit.

    The bounds come from matured training outcomes only. Reported errors use
    the original targets.
    """
    y = train["target"].to_numpy(dtype=float)
    lo, hi = np.quantile(y, [0.01, 0.99])
    y_fit = np.clip(y, lo, hi)
    weights = _date_weights(train)
    pipe.fit(train[MODEL_COLUMNS], y_fit, model__sample_weight=weights)
    return pipe, float(lo), float(hi)


def matured_training_rows(frame, fit_boundary):
    fit_boundary = pd.Timestamp(fit_boundary)
    # Peer stats above were computed with every eligible name. The fit keeps
    # only outcomes whose exit date is already in the past.
    keep = frame["label_end"].notna() & frame["label_end"].lt(fit_boundary) & frame["target"].notna()
    train = frame.loc[keep].copy()
    assert_timing(train, fit_boundary)
    if not np.isfinite(train["target"]).all():
        raise ValueError("Non-finite training target")
    return train


def score_block(train_raw, predict_raw, params, fit_boundary, label_end_max=None):
    train_raw = add_peer_features(train_raw)
    predict = add_peer_features(predict_raw)
    train = matured_training_rows(train_raw, fit_boundary)
    started = time.perf_counter()
    pipe, y_lo, y_hi = fit_regressor(make_tree(params), train)
    prediction = pipe.predict(predict[MODEL_COLUMNS])
    elapsed = time.perf_counter() - started
    predict = predict.copy()
    predict["prediction"] = prediction
    audit = {
        "fit_boundary": str(pd.Timestamp(fit_boundary).date()),
        "train_rows": int(len(train)),
        "train_dates": int(train["date"].nunique()),
        "train_signal_min": str(train["date"].min().date()),
        "train_signal_max": str(train["date"].max().date()),
        "train_label_end_max": str(train["label_end"].max().date()),
        "predict_rows": int(len(predict)),
        "y_clip_low": y_lo,
        "y_clip_high": y_hi,
        "fit_seconds": round(elapsed, 1),
    }
    log(
        f"Fit before {audit['fit_boundary']}: {audit['train_rows']:,} rows, "
        f"labels through {audit['train_label_end_max']}, {elapsed:.1f}s"
    )
    return predict, audit


def daily_rank_ic(scored):
    values = []
    for _, group in scored.groupby("date", sort=True):
        if len(group) < 5 or group["prediction"].nunique() <= 1 or group["target"].nunique() <= 1:
            values.append(np.nan)
            continue
        values.append(group["prediction"].corr(group["target"], method="spearman"))
    return pd.Series(values, index=scored["date"].drop_duplicates().sort_values(), dtype=float)


def ranking_stats(scored, score_column="prediction"):
    """Cross-sectional spread of realized 90-day returns. Overlapping, not a NAV."""
    spreads = []
    excesses = []
    for _, group in scored.groupby("date", sort=True):
        if len(group) < 30 or group[score_column].nunique() < 5:
            continue
        low, high = group[score_column].quantile([0.2, 0.8])
        top = group.loc[group[score_column] >= high, "target"].mean()
        bottom = group.loc[group[score_column] <= low, "target"].mean()
        spreads.append(top - bottom)
        winners = group.nlargest(30, score_column)["target"].mean()
        excesses.append(winners - group["target"].mean())
    return {
        "quintile_spread": float(np.mean(spreads)) if spreads else np.nan,
        "top30_excess": float(np.mean(excesses)) if excesses else np.nan,
        "ranking_dates": int(len(excesses)),
    }


def summarize(scored, score_sign=1):
    usable = scored.loc[scored["target"].notna() & np.isfinite(scored["target"])].copy()
    if usable.empty:
        raise ValueError("No completed outcomes to score")
    metrics = {
        "rows": int(len(usable)),
        "dates": int(usable["date"].nunique()),
        "RMSE": float(np.sqrt(mean_squared_error(usable["target"], usable["prediction"]))),
        "MAE": float(mean_absolute_error(usable["target"], usable["prediction"])),
        "R2": float(r2_score(usable["target"], usable["prediction"])),
        "mean_daily_rank_IC": float(daily_rank_ic(usable).mean()),
    }
    usable["strategy_score"] = score_sign * usable["prediction"]
    metrics.update(ranking_stats(usable, "strategy_score"))
    metrics["strategy_IC"] = float(score_sign * metrics["mean_daily_rank_IC"])
    return metrics, usable


def metrics_by_year(scored, score_sign, role):
    rows = []
    scored = scored.copy()
    scored["year"] = scored["date"].dt.year
    for year, group in scored.groupby("year"):
        metrics, _ = summarize(group, score_sign=score_sign)
        rows.append({"year": int(year), "role": role, **metrics})
    return pd.DataFrame(rows)


def first_on_or_after(start):
    start = pd.Timestamp(start)
    frame = read_panel(start, start + pd.Timedelta(days=14))
    if frame.empty:
        raise ValueError(f"No eligible signals on or after {start.date()}")
    return pd.Timestamp(frame["date"].min())


def window_frames(fit_boundary, predict_start, predict_end):
    fit_boundary = pd.Timestamp(fit_boundary)
    train_raw = read_panel(TRAIN_START, fit_boundary - pd.Timedelta(days=1), month_start_only=True)
    predict_raw = read_panel(predict_start, predict_end)
    if train_raw.empty or predict_raw.empty:
        raise ValueError(f"Missing rows for fit {fit_boundary.date()} predict {predict_start.date()}")
    assert_timing(train_raw)
    assert_timing(predict_raw)
    return train_raw, predict_raw


def evaluate_config(name, params, years, label_end_max=None):
    fold_rows = []
    audits = []
    scored_parts = []
    for year in years:
        fit_boundary = first_on_or_after(f"{year}-01-01")
        predict_end = min(pd.Timestamp(f"{year}-12-31"), DEV_END)
        train_raw, predict_raw = window_frames(fit_boundary, fit_boundary, predict_end)
        scored, audit = score_block(train_raw, predict_raw, params, fit_boundary)
        audit["config"] = name
        audit["predict_start"] = str(fit_boundary.date())
        audit["predict_end"] = str(predict_end.date())
        audits.append(audit)
        if label_end_max is not None:
            keep = scored["label_end"].isna() | scored["label_end"].le(label_end_max)
            scored = scored.loc[keep]
        metrics, usable = summarize(scored)
        fold_rows.append({"config": name, "year": year, "retrain": "annual", **metrics})
        usable["config"] = name
        usable["year"] = year
        scored_parts.append(usable)
        log(f"{name} {year}: rank IC {metrics['mean_daily_rank_IC']:.4f}, RMSE {metrics['RMSE']:.4f}")
    return pd.DataFrame(fold_rows), audits, scored_parts


def baseline_folds(years):
    rows = []
    for year in years:
        fit_boundary = first_on_or_after(f"{year}-01-01")
        predict_end = min(pd.Timestamp(f"{year}-12-31"), DEV_END)
        train_raw, predict_raw = window_frames(fit_boundary, fit_boundary, predict_end)
        train = matured_training_rows(add_peer_features(train_raw), fit_boundary)
        predict = add_peer_features(predict_raw)
        predict = predict.loc[predict["target"].notna()].copy()
        y_train = train["target"].to_numpy(dtype=float)
        lo, hi = np.quantile(y_train, [0.01, 0.99])
        # AAM's recorded 2012 holding return exceeds 100,000, so the raw mean is
        # not a usable forecast. Use the same training-label clip as the tree.
        mean_prediction = np.full(len(predict), np.clip(y_train, lo, hi).mean())
        predict["prediction"] = mean_prediction
        mean_metrics, _ = summarize(predict)
        rows.append({"model": "training mean", "retrain": "annual", "year": year, **mean_metrics})

        ridge = course_data.make_pipeline(
            Ridge(alpha=100), numeric_features=NUMERIC_FEATURES, include_sector=True
        )
        y = train["target"].to_numpy(dtype=float)
        lo, hi = np.quantile(y, [0.01, 0.99])
        ridge.fit(train[MODEL_COLUMNS], np.clip(y, lo, hi), model__sample_weight=_date_weights(train))
        predict["prediction"] = ridge.predict(predict[MODEL_COLUMNS])
        ridge_metrics, _ = summarize(predict)
        rows.append({"model": "Ridge alpha 100", "retrain": "annual", "year": year, **ridge_metrics})
        log(
            f"baseline {year}: ridge IC {ridge_metrics['mean_daily_rank_IC']:.4f}, "
            f"mean model IC {mean_metrics['mean_daily_rank_IC']:.4f}"
        )
    return pd.DataFrame(rows)


def static_years(params, years, fit_year, label_end_max=None):
    fit_boundary = first_on_or_after(f"{fit_year}-01-01")
    predict_end = min(pd.Timestamp(f"{max(years)}-12-31"), DEV_END)
    train_raw, predict_raw = window_frames(fit_boundary, first_on_or_after(f"{min(years)}-01-01"), predict_end)
    scored, audit = score_block(train_raw, predict_raw, params, fit_boundary)
    if label_end_max is not None:
        scored = scored.loc[scored["label_end"].isna() | scored["label_end"].le(label_end_max)]
    rows = []
    for year in years:
        part = scored.loc[scored["date"].dt.year.eq(year) & scored["target"].notna()]
        metrics, _ = summarize(part)
        rows.append({"config": "chosen", "year": year, "retrain": "static", **metrics})
        log(f"static {year}: rank IC {metrics['mean_daily_rank_IC']:.4f}")
    audit["config"] = "static"
    return pd.DataFrame(rows), audit


def plot_results(yearly, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"selection": "#1f4e79", "confirmation": "#c47b17", "final test": "#2e7d32"}
    for role, group in yearly.groupby("role", sort=False):
        axes[0].bar(group["year"], group["mean_daily_rank_IC"], color=colors[role], label=role, width=0.8)
        axes[1].bar(group["year"], group["top30_excess"], color=colors[role], width=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean daily Spearman IC")
    axes[1].set_ylabel("Top-30 excess 90-day return")
    axes[1].set_xlabel("Signal year")
    axes[0].set_title("90-day return model, frozen before the final test")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(yearly["year"].tolist())
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def verify():
    """Match the lecture feature and label timing on the class extract."""
    log("Checking features against course_data on the class database")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reference = course_data.prepare_panel(CLASS_DB, end="2015-12-31")
    mine = build_panel(
        CLASS_DB,
        price_end="2015-12-31",
        export_start="2008-01-01",
        eligible_only=False,
    )
    reference["date"] = pd.to_datetime(reference["date"]).astype("datetime64[ns]")
    merged = mine.merge(reference, on=["symbol", "date"], suffixes=("_mine", "_ref"))
    if merged.empty:
        raise ValueError("Class-database comparison matched no rows")
    columns = [
        "momentum_21",
        "momentum_63",
        "momentum_252_21",
        "volatility_63",
        "drawdown_252",
        "net_margin",
        "operating_roa",
        "debt_to_assets",
        "cash_to_assets",
        "fcf_to_assets",
        "accruals_to_assets",
        "filing_age_days",
        "target",
    ]
    worst = {}
    for column in columns:
        left = merged[f"{column}_mine"]
        right = merged[f"{column}_ref"]
        both_missing = left.isna() & right.isna()
        disagree_missing = left.isna() != right.isna()
        gap = (left - right).abs()
        gap = gap.mask(both_missing, 0)
        worst[column] = {
            "max_abs": float(gap.max(skipna=True) or 0),
            "missing_mismatches": int(disagree_missing.sum()),
        }
        if worst[column]["max_abs"] > 1e-8 or worst[column]["missing_mismatches"]:
            raise AssertionError(f"{column} does not match the lecture helper: {worst[column]}")
    eligible_gap = int((merged["eligible_mine"] != merged["eligible_ref"]).sum())
    if eligible_gap:
        raise AssertionError(f"Eligibility differs on {eligible_gap} rows")

    signal = pd.Timestamp("2012-08-31")
    aapl = mine.loc[mine["symbol"].eq("AAPL") & mine["date"].eq(signal)].iloc[0]
    if pd.Timestamp(aapl["entry_date"]) != pd.Timestamp("2012-09-04"):
        raise AssertionError(f"AAPL entry date is {aapl['entry_date']}, expected 2012-09-04")
    if pd.Timestamp(aapl["label_end"]) != pd.Timestamp("2012-12-03"):
        raise AssertionError(f"AAPL exit date is {aapl['label_end']}, expected 2012-12-03")
    RESULTS.mkdir(parents=True, exist_ok=True)
    check = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "signal_date": "2012-08-31",
                "entry_date": str(pd.Timestamp(aapl["entry_date"]).date()),
                "exit_date": str(pd.Timestamp(aapl["label_end"]).date()),
                "target_return": float(aapl["target"]),
                "rows_compared": int(len(merged)),
                "max_feature_gap": max(item["max_abs"] for item in worst.values()),
            }
        ]
    )
    check.to_csv(RESULTS / "label_timing_check.csv", index=False)
    log(f"Class timing check passed on {len(merged):,} rows. AAPL 90-day return {aapl['target']:.4f}")

    # The smoke fit only checks that the estimator accepts this matrix.
    sample = mine.loc[mine["eligible"] & mine["date"].between("2010-01-01", "2012-12-31")].copy()
    sample = add_peer_features(sample)
    train = matured_training_rows(sample.loc[sample["is_month_start"]], pd.Timestamp("2013-01-02"))
    pipe = make_tree(
        dict(max_iter=3, max_leaf_nodes=7, min_samples_leaf=20, learning_rate=0.1, early_stopping=False, random_state=42)
    )
    pipe, _, _ = fit_regressor(pipe, train)
    pred = pipe.predict(train[MODEL_COLUMNS])
    if not np.isfinite(pred).all():
        raise AssertionError("Smoke-fit predictions were not finite")
    ridge = course_data.make_pipeline(Ridge(alpha=100), numeric_features=NUMERIC_FEATURES, include_sector=True)
    y = train["target"].to_numpy(dtype=float)
    ridge.fit(train[MODEL_COLUMNS], y, model__sample_weight=_date_weights(train))
    if not np.isfinite(ridge.predict(train[MODEL_COLUMNS])).all():
        raise AssertionError("Ridge smoke-fit predictions were not finite")
    log("Tree and Ridge pipelines accept the feature matrix")


def develop():
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PANEL_PATH}; build the panel first")
    log("Selecting a tree size on 2013-2014. 2015 is held for confirmation.")
    fold_frames = []
    audits = []
    for name, params in CONFIGS.items():
        folds, fold_audits, _ = evaluate_config(name, params, SELECTION_YEARS)
        fold_frames.append(folds)
        audits.extend(fold_audits)
    folds = pd.concat(fold_frames, ignore_index=True)
    folds.to_csv(RESULTS / "validation_folds.csv", index=False)
    ranking = (
        folds.groupby("config")
        .agg(mean_daily_rank_IC=("mean_daily_rank_IC", "mean"), RMSE=("RMSE", "mean"))
        .sort_values(["mean_daily_rank_IC", "RMSE"], ascending=[False, True])
    )
    chosen_name = ranking.index[0]
    chosen_params = CONFIGS[chosen_name]
    config_search_ic = float(ranking.loc[chosen_name, "mean_daily_rank_IC"])
    log(f"Chosen config {chosen_name} by annual-refit IC {config_search_ic:.4f}")

    log("Comparing one fit against a refit at each year, on the same two years")
    annual = folds.loc[folds["config"].eq(chosen_name)].copy()
    static, static_audit = static_years(chosen_params, SELECTION_YEARS, fit_year=SELECTION_YEARS[0])
    audits.append(static_audit)
    annual_ic = float(annual["mean_daily_rank_IC"].mean())
    static_ic = float(static["mean_daily_rank_IC"].mean())
    retrain_rule = "annual" if annual_ic >= static_ic else "static"
    # The sign follows the rule we will actually trade, not the discarded refits.
    rule_ic = annual_ic if retrain_rule == "annual" else static_ic
    score_sign = 1 if rule_ic >= 0 else -1
    log(f"Retrain rule: {retrain_rule} (annual IC {annual_ic:.4f}, static IC {static_ic:.4f}), score sign {score_sign}")

    baselines = pd.concat(
        [baseline_folds(SELECTION_YEARS), static_ridge(SELECTION_YEARS, SELECTION_YEARS[0])],
        ignore_index=True,
    )
    baselines.to_csv(RESULTS / "baseline_folds.csv", index=False)

    log("Scoring 2015 with the frozen rule. This does not change the choice.")
    if retrain_rule == "annual":
        confirm, confirm_audits, _ = evaluate_config(
            chosen_name, chosen_params, (CONFIRM_YEAR,), label_end_max=BUFFER_END
        )
        audits.extend(confirm_audits)
    else:
        confirm, confirm_audit = static_years(
            chosen_params, (CONFIRM_YEAR,), fit_year=SELECTION_YEARS[0], label_end_max=BUFFER_END
        )
        confirm["config"] = chosen_name
        audits.append(confirm_audit)
    confirm.to_csv(RESULTS / "confirmation_2015.csv", index=False)
    chosen_dev = (static if retrain_rule == "static" else annual).copy()
    chosen_dev["role"] = "selection"
    confirm_rows = confirm.copy()
    confirm_rows["role"] = "confirmation"
    strategy_years = pd.concat([chosen_dev, confirm_rows], ignore_index=True)
    strategy_years.to_csv(RESULTS / "chosen_strategy_years.csv", index=False)
    pd.concat([annual, static], ignore_index=True).to_csv(RESULTS / "retrain_comparison.csv", index=False)

    choice = {
        "model": "HistGradientBoostingRegressor",
        "config_name": chosen_name,
        "params": chosen_params,
        "retrain_rule": retrain_rule,
        "score_sign": score_sign,
        "selection_years": list(SELECTION_YEARS),
        "selection_metric": "mean daily Spearman rank IC of the frozen retrain rule",
        "selection_ic": rule_ic,
        "config_search_ic": config_search_ic,
        "annual_ic": annual_ic,
        "static_ic": static_ic,
        "features": MODEL_COLUMNS,
        "training_sample": "first market session of each month",
        "target_clip": [0.01, 0.99],
        "feature_clip": [0.01, 0.99],
    }
    CHOICE_PATH.write_text(json.dumps(choice, indent=2))
    pd.DataFrame(audits).to_csv(RESULTS / "development_fit_audit.csv", index=False)
    ranking.to_csv(RESULTS / "config_ranking.csv")
    log(f"Froze {chosen_name} / {retrain_rule} before the final test")
    return choice


def date_level_metrics(scored, score_sign):
    """One row per signal date, small enough to keep for the full test."""
    scored = scored.copy()
    scored["strategy_score"] = score_sign * scored["prediction"]
    rows = []
    for date, group in scored.groupby("date", sort=True):
        y = group["target"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        row = {
            "date": pd.Timestamp(date),
            "n": int(len(group)),
            "sse": float(np.square(y - prediction).sum()),
            "sae": float(np.abs(y - prediction).sum()),
            "sum_y": float(y.sum()),
            "sum_y2": float(np.square(y).sum()),
            "ic": np.nan,
            "spread": np.nan,
            "excess": np.nan,
        }
        if len(group) >= 5 and group["prediction"].nunique() > 1 and group["target"].nunique() > 1:
            row["ic"] = float(group["prediction"].corr(group["target"], method="spearman"))
        if len(group) >= 30 and group["strategy_score"].nunique() >= 5:
            low, high = group["strategy_score"].quantile([0.2, 0.8])
            top = group.loc[group["strategy_score"] >= high, "target"].mean()
            bottom = group.loc[group["strategy_score"] <= low, "target"].mean()
            row["spread"] = float(top - bottom)
            row["excess"] = float(group.nlargest(30, "strategy_score")["target"].mean() - y.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_daily(daily, score_sign):
    n = int(daily["n"].sum())
    sse = float(daily["sse"].sum())
    sum_y = float(daily["sum_y"].sum())
    sum_y2 = float(daily["sum_y2"].sum())
    sst = sum_y2 - (sum_y ** 2) / n
    ic = float(daily["ic"].mean())
    return {
        "rows": n,
        "dates": int(daily["date"].nunique()),
        "RMSE": float(np.sqrt(sse / n)),
        "MAE": float(daily["sae"].sum() / n),
        "R2": float(1 - sse / sst) if sst > 0 else np.nan,
        "mean_daily_rank_IC": ic,
        "strategy_IC": float(score_sign * ic),
        "quintile_spread": float(daily["spread"].mean()),
        "top30_excess": float(daily["excess"].mean()),
        "ranking_dates": int(daily["excess"].notna().sum()),
    }


def test(choice=None):
    if choice is None:
        if not CHOICE_PATH.exists():
            raise FileNotFoundError("chosen_model.json is missing; run development before the test")
        choice = json.loads(CHOICE_PATH.read_text())
    params = choice["params"]
    score_sign = int(choice["score_sign"])
    rule = choice["retrain_rule"]
    log(f"Final test from {TEST_START.date()} using {choice['config_name']} ({rule})")

    if rule == "static":
        fit_points = [first_on_or_after(TEST_START)]
    else:
        fit_points = [first_on_or_after(TEST_START)]
        for year in range(2017, 2027):
            fit_points.append(first_on_or_after(pd.Timestamp(f"{year}-01-01")))
    fit_points = sorted(dict.fromkeys(fit_points))
    # Predict one calendar year at a time so a ten-year static model does not
    # hold the whole test cross-section in memory. The fit still changes only
    # on the retrain dates above.
    predict_starts = list(fit_points)
    for year in range(2017, 2027):
        predict_starts.append(first_on_or_after(pd.Timestamp(f"{year}-01-01")))
    predict_starts = sorted(dict.fromkeys(predict_starts))

    audits = []
    daily_parts = []
    open_holds = 0
    fitted = None
    active_fit = None
    for i, start in enumerate(predict_starts):
        predict_end = predict_starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(predict_starts) else TEST_END
        if predict_end < start:
            continue
        if start in fit_points or fitted is None:
            train_raw = read_panel(TRAIN_START, start - pd.Timedelta(days=1), month_start_only=True)
            assert_timing(train_raw)
            train = matured_training_rows(add_peer_features(train_raw), start)
            started = time.perf_counter()
            fitted, y_lo, y_hi = fit_regressor(make_tree(params), train)
            elapsed = time.perf_counter() - started
            active_fit = start
            audits.append({
                "fit_boundary": str(start.date()),
                "train_rows": int(len(train)),
                "train_dates": int(train["date"].nunique()),
                "train_signal_min": str(train["date"].min().date()),
                "train_signal_max": str(train["date"].max().date()),
                "train_label_end_max": str(train["label_end"].max().date()),
                "y_clip_low": y_lo,
                "y_clip_high": y_hi,
                "fit_seconds": round(elapsed, 1),
                "retrain_rule": rule,
            })
            log(
                f"Fit before {start.date()}: {len(train):,} rows, "
                f"labels through {train['label_end'].max().date()}, {elapsed:.1f}s"
            )
        predict_raw = read_panel(start, predict_end)
        assert_timing(predict_raw)
        predict = add_peer_features(predict_raw)
        open_holds += int(predict["target"].isna().sum())
        predict = predict.loc[predict["target"].notna() & np.isfinite(predict["target"])].copy()
        predict["prediction"] = fitted.predict(predict[MODEL_COLUMNS])
        daily_parts.append(date_level_metrics(predict, score_sign))
        log(f"Scored {start.date()} through {pd.Timestamp(predict_end).date()} with the {active_fit.date()} fit")

    daily = pd.concat(daily_parts, ignore_index=True)
    metrics = aggregate_daily(daily, score_sign)
    metrics["open_holds_excluded"] = open_holds
    yearly_rows = []
    for year, group in daily.groupby(daily["date"].dt.year):
        yearly_rows.append({"year": int(year), "role": "final test", **aggregate_daily(group, score_sign)})
    yearly = pd.DataFrame(yearly_rows)
    pd.DataFrame(audits).to_csv(RESULTS / "test_fit_audit.csv", index=False)
    yearly.to_csv(RESULTS / "test_metrics_by_year.csv", index=False)
    pd.Series(metrics).to_csv(RESULTS / "test_summary.csv", header=["value"])
    log(
        f"Test rank IC {metrics['mean_daily_rank_IC']:.4f}, "
        f"strategy IC {metrics['strategy_IC']:.4f}, "
        f"top-30 excess {metrics['top30_excess']:.4f}, "
        f"RMSE {metrics['RMSE']:.4f}"
    )
    return yearly, metrics


def static_ridge(years, fit_year):
    """Ridge fit once, matching the static tree comparison."""
    fit_boundary = first_on_or_after(f"{fit_year}-01-01")
    predict_end = min(pd.Timestamp(f"{max(years)}-12-31"), DEV_END)
    train_raw, predict_raw = window_frames(fit_boundary, first_on_or_after(f"{min(years)}-01-01"), predict_end)
    train = matured_training_rows(add_peer_features(train_raw), fit_boundary)
    predict = add_peer_features(predict_raw)
    ridge = course_data.make_pipeline(Ridge(alpha=100), numeric_features=NUMERIC_FEATURES, include_sector=True)
    y = train["target"].to_numpy(dtype=float)
    lo, hi = np.quantile(y, [0.01, 0.99])
    ridge.fit(train[MODEL_COLUMNS], np.clip(y, lo, hi), model__sample_weight=_date_weights(train))
    rows = []
    for year in years:
        part = predict.loc[predict["date"].dt.year.eq(year) & predict["target"].notna()].copy()
        part["prediction"] = ridge.predict(part[MODEL_COLUMNS])
        metrics, _ = summarize(part)
        rows.append({"model": "Ridge alpha 100", "retrain": "static", "year": int(year), **metrics})
        log(f"static ridge {year}: rank IC {metrics['mean_daily_rank_IC']:.4f}")
    return pd.DataFrame(rows)


def write_chart(choice):
    pre = pd.read_csv(RESULTS / "chosen_strategy_years.csv")
    tested = pd.read_csv(RESULTS / "test_metrics_by_year.csv")
    pre = pre[["year", "role", "mean_daily_rank_IC", "top30_excess"]].copy()
    tested = tested[["year", "mean_daily_rank_IC", "top30_excess"]].copy()
    tested["role"] = "final test"
    yearly = pd.concat([pre, tested], ignore_index=True)
    plot_results(yearly, RESULTS / "rank_ic_by_year.png")
    yearly.to_csv(RESULTS / "rank_ic_by_year.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Build and evaluate the HW4 90-day return model")
    parser.add_argument("--stage", choices=["verify", "build", "develop", "test", "all"], default="all")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.stage in ("verify", "all"):
        verify()
    if args.stage in ("build", "all"):
        build_panel(FULL_DB, price_end=TEST_END, export_start=TRAIN_START, out_path=PANEL_PATH, eligible_only=True)
    choice = None
    if args.stage in ("develop", "all"):
        choice = develop()
    if args.stage in ("test", "all"):
        test(choice)
        write_chart(choice if choice is not None else json.loads(CHOICE_PATH.read_text()))
    log("Done")


if __name__ == "__main__":
    main()
