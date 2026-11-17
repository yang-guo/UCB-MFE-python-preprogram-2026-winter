"""Load the modeling dataset and build look-ahead-safe expanding-window folds.

Reads the finalized feature panel and labels (read-only); it never changes the
feature or label methodology.
"""
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

FEATURE_PANEL = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/combined_feature_panel.parquet"
)

LABELS = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/training_labels.parquet"
)

# Fixed feature list, chosen before looking at any validation results.
# adj_close levels are excluded: back-adjusted prices embed later splits and
# dividends. Raw dollar fundamentals are excluded for unit/scale problems.
FEATURES = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "universe_rank",
    "profit_margin",
    "ebitda_margin",
    "debt_to_equity",
    "asset_turnover",
    "fcf_margin",
    "fundamental_age_days",  # derived row-wise: date - fiscal_date
]

TUNING_YEARS = (2012, 2013, 2014)
HOLDOUT_YEAR = 2015

# First final-test session. No label exiting on/after this date may be used,
# not even for development evaluation.
TEST_START = pd.Timestamp("2016-04-01")

# Tuning fits use every Nth market session (labels overlap for 90 days, so
# daily rows are highly redundant). Validation always uses every session.
TRAIN_EVERY_N_SESSIONS = 5


@dataclass
class Fold:
    year: int
    fit_date: pd.Timestamp
    train: pd.DataFrame
    valid: pd.DataFrame


def load_dataset():
    """Return (usable labelled rows with features, market-session index)."""
    con = duckdb.connect(str(DB), read_only=True)
    feature_cols = ", ".join(f"f.{c}" for c in FEATURES if c != "fundamental_age_days")
    frame = con.execute(f"""
        SELECT
            l.symbol,
            l.feature_date,
            l.actual_exit_date,
            l.forward_return,
            {feature_cols},
            CAST(DATE_DIFF('day', f.fiscal_date, f.date) AS DOUBLE)
                AS fundamental_age_days
        FROM read_parquet('{LABELS}') l
        JOIN read_parquet('{FEATURE_PANEL}') f
            ON f.symbol = l.symbol
           AND f.date = l.feature_date
        WHERE NOT l.censored
          AND NOT l.extreme_move_flag
        ORDER BY l.feature_date, l.symbol
    """).df()
    sessions = con.execute("""
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) - 1 AS session_number
        FROM calendar
    """).df()
    con.close()

    for col in ("feature_date", "actual_exit_date"):
        frame[col] = pd.to_datetime(frame[col])
    sessions["date"] = pd.to_datetime(sessions["date"])
    session_number = sessions.set_index("date")["session_number"]
    frame["session_number"] = frame["feature_date"].map(session_number)
    assert frame["session_number"].notna().all()
    return frame, sessions


def percentile_rank_target(frame):
    """Cross-sectional percentile rank of forward_return within each date.

    Ranks only among the rows passed in, so callers must filter to eligible
    rows *before* calling this.
    """
    return frame.groupby("feature_date")["forward_return"].rank(pct=True)


def make_fold(frame, sessions, year, train_every=TRAIN_EVERY_N_SESSIONS):
    """Expanding-window fold whose model is fitted at the year's first session."""
    fit_date = sessions.loc[sessions["date"] >= pd.Timestamp(f"{year}-01-01"), "date"].min()

    # Training: features before the fit date AND outcomes finished before it.
    eligible = (
        (frame["feature_date"] < fit_date)
        & (frame["actual_exit_date"] < fit_date)
        & (frame["session_number"] % train_every == 0)
    )
    train = frame.loc[eligible].copy()
    # Ranked only among labels that had matured by fit_date.
    train["target"] = percentile_rank_target(train)

    # Validation: every session of the year; drop any outcome in the test period.
    in_year = frame["feature_date"].dt.year == year
    valid = frame.loc[in_year & (frame["actual_exit_date"] < TEST_START)].copy()
    # Evaluation-only target; never used for fitting or preprocessing.
    valid["target"] = percentile_rank_target(valid)

    assert len(train) and len(valid)
    assert train["actual_exit_date"].max() < fit_date
    assert train["feature_date"].max() < valid["feature_date"].min()
    assert valid["feature_date"].min() == fit_date
    assert valid["actual_exit_date"].max() < TEST_START
    return Fold(year, fit_date, train, valid)


def describe_fold(fold):
    """One summary row of dates and sample sizes for a fold."""
    t, v = fold.train, fold.valid
    return {
        "year": fold.year,
        "fit_date": fold.fit_date.date(),
        "train_first_feature": t["feature_date"].min().date(),
        "train_last_feature": t["feature_date"].max().date(),
        "train_last_exit": t["actual_exit_date"].max().date(),
        "train_rows": len(t),
        "train_dates": t["feature_date"].nunique(),
        "train_stocks": t["symbol"].nunique(),
        "valid_first_feature": v["feature_date"].min().date(),
        "valid_last_feature": v["feature_date"].max().date(),
        "valid_last_exit": v["actual_exit_date"].max().date(),
        "valid_rows": len(v),
        "valid_dates": v["feature_date"].nunique(),
        "valid_stocks": v["symbol"].nunique(),
    }


if __name__ == "__main__":
    data, calendar = load_dataset()
    print(f"Usable labelled rows: {len(data):,}")
    print(data[FEATURES].isna().mean().rename("missing_share").to_string())
    folds = [make_fold(data, calendar, y) for y in (*TUNING_YEARS, HOLDOUT_YEAR)]
    print(pd.DataFrame([describe_fold(f) for f in folds]).T.to_string(header=False))
