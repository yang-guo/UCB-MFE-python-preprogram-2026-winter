"""Development rank check and a list of lots still open past their due date.

Uses the batch files already written by 03_run_test.py. It does not reload the database.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backtest import run_backtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "Lectures" / "modeling"))
import course_data as cd  # noqa: E402

PARTS = HERE / "artifacts" / "parts"
TEST_END = pd.Timestamp("2026-09-30")
# A 90-day label opened on this date finishes before the 2014 validation signals.
TRAIN_BEFORE = pd.Timestamp("2013-10-01")
VALID_START = pd.Timestamp("2014-01-01")


def read_parts(prefix):
    paths = sorted(PARTS.glob(f"{prefix}_*.parquet"))
    if not paths:
        raise FileNotFoundError(prefix)
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def development_check():
    train = read_parts("train")
    train["date"] = pd.to_datetime(train["date"])
    fit_rows = train.date.le(TRAIN_BEFORE)
    valid_rows = train.date.ge(VALID_START)
    pipeline = cd.make_pipeline()
    pipeline.fit(train.loc[fit_rows, cd.FEATURES], train.loc[fit_rows, "target"])
    valid = train.loc[valid_rows, ["date", "symbol", "target"]].copy()
    valid["prediction"] = pipeline.predict(train.loc[valid_rows, cd.FEATURES])
    ic = valid.groupby("date").apply(
        lambda frame: frame.prediction.corr(frame.target, method="spearman")
        if len(frame) >= 5 and frame.prediction.nunique() > 1 and frame.target.nunique() > 1 else np.nan,
        include_groups=False,
    )
    summary = pd.Series({
        "fit_rows": int(fit_rows.sum()),
        "valid_rows": int(valid_rows.sum()),
        "mean_daily_rank_IC": ic.mean(),
        "IC_dates": int(ic.notna().sum()),
    })
    summary.to_csv(HERE / "artifacts" / "full_universe_validation.csv", header=["value"])
    print(summary.to_string())
    return summary


def overdue_lots():
    saved = joblib.load(HERE / "artifacts" / "ridge_90d_full.joblib")
    score = read_parts("score")
    scored = score[["date", "symbol"]].copy()
    scored["score"] = saved["pipeline"].predict(score[saved["features"]])
    del score
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna()
    closes = read_parts("close")
    closes["date"] = pd.to_datetime(closes["date"])
    prices = closes.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    result = run_backtest(prices, scored, "2016-04-01", "2026-09-30")
    last = closes.sort_values("date").groupby("symbol").tail(1).set_index("symbol")
    rows = []
    for lot in result["open_lots"]:
        if lot.due_date > TEST_END:
            continue
        last_date = last.at[lot.symbol, "date"] if lot.symbol in last.index else pd.NaT
        last_price = last.at[lot.symbol, "adj_close"] if lot.symbol in last.index else np.nan
        rows.append({
            "symbol": lot.symbol,
            "entry_date": lot.entry_date.date().isoformat(),
            "due_date": lot.due_date.date().isoformat(),
            "last_close_date": pd.Timestamp(last_date).date().isoformat() if pd.notna(last_date) else "",
            "units": lot.units,
            "last_close": last_price,
            "marked_value": lot.units * last_price,
        })
    overdue = pd.DataFrame(rows)
    if not overdue.empty:
        overdue = overdue.sort_values(["due_date", "symbol"])
    path = HERE / "artifacts" / "overdue_open_lots.csv"
    overdue.to_csv(path, index=False)
    still_in_term = sum(lot.due_date > TEST_END for lot in result["open_lots"])
    marked = float(overdue["marked_value"].sum()) if not overdue.empty else 0.0
    print(f"overdue={len(overdue)} still_inside_90_days={still_in_term} overdue_marked_value={marked:.2f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    development_check()
    overdue_lots()
