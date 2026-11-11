"""Step 1: build the development panel and fit the Ridge pipeline.

Feature construction and the sklearn pipeline come from
Lectures/modeling/course_data.py. This script chooses the dates and saves the fit.

Default database is the class extract, which ends in 2015. Pass --full to use
equities_top2000.duckdb through the March 2016 buffer.
"""

import argparse
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODELING = REPO / "Lectures" / "modeling"
sys.path.insert(0, str(MODELING))

import course_data as cd  # noqa: E402

DATA = MODELING / "data"
ARTIFACTS = HERE / "artifacts"
TRAIN_START = "2010-01-01"
VALID_START = "2014-01-01"
DEV_LAST_SIGNAL = "2015-12-31"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use equities_top2000.duckdb")
    args = parser.parse_args()

    if args.full:
        database = DATA / "equities_top2000.duckdb"
        dev_end = "2016-03-31"
        test_start = "2016-04-01"
    else:
        database = DATA / "equities_class.duckdb"
        # The class extract has no 2016 prices, so late-2015 labels stay missing.
        dev_end = "2015-12-31"
        test_start = "2016-01-01"

    ARTIFACTS.mkdir(exist_ok=True)
    if not database.is_file():
        raise FileNotFoundError(database)

    print(f"building panel from {database.name} through {dev_end}")
    panel = cd.prepare_panel(database, end=dev_end)
    tag = "full" if args.full else "class"
    panel_path = ARTIFACTS / f"dev_panel_{tag}.parquet"
    panel.to_parquet(panel_path, index=False)
    print(f"wrote {panel_path} rows={len(panel)}")

    has_label = panel.eligible & panel.target.notna() & panel.date.between(TRAIN_START, DEV_LAST_SIGNAL)
    final_rows = has_label & panel.label_end.lt(test_start)
    first_valid_signal = panel.loc[has_label & panel.date.ge(VALID_START), "date"].min()
    train_rows = has_label & panel.date.lt(first_valid_signal) & panel.label_end.lt(first_valid_signal)
    valid_rows = has_label & panel.date.ge(VALID_START)
    print(
        f"train={int(train_rows.sum())} valid={int(valid_rows.sum())} "
        f"final={int(final_rows.sum())} first_valid_signal={first_valid_signal.date()}"
    )
    if panel.loc[train_rows, "label_end"].max() >= first_valid_signal:
        raise RuntimeError("a training label ends on or after the first validation signal")

    check = cd.make_pipeline()
    check.fit(panel.loc[train_rows, cd.FEATURES], panel.loc[train_rows, "target"])
    metrics = cd.prediction_metrics(panel.loc[valid_rows], check.predict(panel.loc[valid_rows, cd.FEATURES]))
    print(metrics.to_string())
    metrics.to_csv(ARTIFACTS / f"dev_validation_metrics_{tag}.csv", header=["value"])

    pipeline = cd.make_pipeline()
    pipeline.fit(panel.loc[final_rows, cd.FEATURES], panel.loc[final_rows, "target"])
    model_path = ARTIFACTS / f"ridge_90d_{tag}.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": list(cd.FEATURES),
            "model_version": f"ridge-alpha100-{tag}-2010-2015",
            "database": database.name,
            "train_start": TRAIN_START,
            "last_signal": DEV_LAST_SIGNAL,
            "label_end_before": test_start,
        },
        model_path,
    )
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
