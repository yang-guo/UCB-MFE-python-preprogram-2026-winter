"""Train my HW4 model again, like in HW4, and save it for the API.

I run this script one time. It needs the course database and environment:

    # from the root of the repository
    uv sync --locked
    cd Homeworks/HW5/Mila_Marsot
    ../../../.venv/bin/python export_hw4_model.py

The API does not use this script.
"""

import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

FOLDER = Path(__file__).resolve().parent
COURSE_FOLDER = FOLDER.parents[2] / "Lectures" / "modeling"
sys.path.insert(0, str(COURSE_FOLDER))

import course_data  # noqa: E402  (the course file, used in my HW4)

import hw4_model  # noqa: E402

DB_PATH = COURSE_FOLDER / "data" / "equities_top2000.duckdb"
TEST_START, TEST_END = "2016-04-01", "2026-09-30"  # same dates as in HW4


def build_hw4_data():
    """Make the HW4 data. Same code as in my HW4 notebook."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    all_stocks = con.sql("SELECT symbol FROM universe ORDER BY symbol").df().symbol.tolist()
    con.close()
    membership = course_data.load_membership(DB_PATH, end=TEST_END)

    pieces = []
    for i in range(0, len(all_stocks), 200):  # 200 stocks at a time
        group = all_stocks[i : i + 200]
        prices, financials, sectors, calendar = course_data.load_data(
            DB_PATH, end=TEST_END, symbols=group
        )
        days = pd.DatetimeIndex(calendar.date)
        rows = course_data.price_features(prices, days)
        rows = course_data.join_financials(
            rows, course_data.financial_events(financials), sectors
        )
        rows = course_data.apply_membership(rows, membership, days)
        rows = course_data.make_labels(rows, prices, days)
        month_ends = days.to_series().groupby(days.to_period("M")).max()
        keep = rows.date.isin(month_ends) & (rows.date >= "2010-01-01") & rows.eligible
        pieces.append(rows[keep])
    return pd.concat(pieces).sort_values(["date", "symbol"]).reset_index(drop=True)


def row_to_dict(row):
    """One row -> a dict for JSON. A missing value becomes None (null)."""
    features = {}
    for column in hw4_model.FEATURES:
        value = row[column].iloc[0]
        if pd.isna(value):
            features[column] = None
        elif column == "sector":
            features[column] = value
        else:
            features[column] = float(value)
    return features


def main():
    data = build_hw4_data()

    # Train the HW4 model, like in my notebook.
    # It only uses rows with a known 90-day return before the first test day.
    first_test_day = data.date[data.entry_date >= TEST_START].min()
    train = data[data.label_end < first_test_day]
    hw4_pipeline = course_data.make_pipeline(Ridge(alpha=100))
    hw4_pipeline.fit(train[course_data.FEATURES], train.target)

    # Train the same model with hw4_model.py. Check: same predictions.
    pipeline = hw4_model.make_pipeline(Ridge(alpha=100))
    pipeline.fit(train[hw4_model.FEATURES], train.target)
    test_rows = data[data.date >= first_test_day]
    assert np.array_equal(
        hw4_pipeline.predict(test_rows[course_data.FEATURES]),
        pipeline.predict(test_rows[hw4_model.FEATURES]),
    )
    print("Same predictions as HW4 on", len(test_rows), "test rows")

    # Save the model.
    bundle = {
        "pipeline": pipeline,
        "model_version": f"hw4-ridge-alpha100-trained-before-{first_test_day:%Y-%m-%d}",
        "features": hw4_model.FEATURES,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, FOLDER / "artifacts" / "hw4_ridge.joblib")

    # Save 2 real rows and their HW4 prediction. The tests use them.
    # NVDA has all the features. The other stock has a missing value.
    day_rows = test_rows[test_rows.date == first_test_day]
    nvda = day_rows[day_rows.symbol == "NVDA"].iloc[[0]]
    missing = day_rows[day_rows[hw4_model.FINANCIAL_FEATURES].isna().any(axis=1)].iloc[[0]]
    references = []
    for name, row in [("all_features", nvda), ("missing_financials", missing)]:
        references.append(
            {
                "name": name,
                "symbol": row.symbol.iloc[0],
                "signal_date": f"{row.date.iloc[0]:%Y-%m-%d}",
                "features": row_to_dict(row),
                "hw4_prediction": float(hw4_pipeline.predict(row[course_data.FEATURES])[0]),
            }
        )
    reference_path = FOLDER / "artifacts" / "reference_predictions.json"
    reference_path.write_text(json.dumps(references, indent=2) + "\n")

    # Print some numbers for the README.
    print("Model version:", bundle["model_version"])
    print("Training rows:", len(train))
    print("Training values (I use them to choose the input ranges):")
    print(train[hw4_model.NUMERIC_FEATURES].quantile([0, 0.01, 0.99, 0.999, 1]).T.round(3))


if __name__ == "__main__":
    main()
