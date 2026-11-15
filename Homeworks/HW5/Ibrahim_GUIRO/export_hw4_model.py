"""Reproduce and export the final HW4 model for the HW5 API."""

import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import joblib

HW4_COMMIT = "ff8f868"
HW4_DIR = "Homeworks/HW4/Ibrahim_GUIRO"
OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]


def main():
    with TemporaryDirectory() as directory:
        source_dir = Path(directory)
        for filename in ("validation_model.py", "backtest_riskadj_momentum.py"):
            source = subprocess.check_output(
                ["git", "show", f"{HW4_COMMIT}:{HW4_DIR}/{filename}"],
                cwd=REPO_ROOT,
            )
            (source_dir / filename).write_bytes(source)

        sys.path.insert(0, str(source_dir))
        try:
            import validation_model as hw4
        finally:
            sys.path.pop(0)

        database = REPO_ROOT / hw4.bt.DATABASE
        with duckdb.connect(str(database), read_only=True) as db:
            calendar, prices, _ = hw4.bt.load_inputs(db)
            scores = hw4.bt.make_scores(prices)
            cohorts = hw4.add_score_rank(hw4.make_cohorts(db, calendar, prices, scores))

        training = cohorts.loc[
            cohorts["entry"].ge(hw4.DEV_START)
            & cohorts["exit_date"].le(hw4.BUFFER_END)
            & cohorts["target"].notna()
        ]
        final_model = hw4.fit_model(training)

        # Use the first test month and pick one evaluated stock by ticker.
        reference_row = (
            cohorts.loc[
                cohorts["entry"].between(hw4.bt.FINAL_START, hw4.bt.FINAL_END)
                & cohorts["target"].notna()
            ]
            .sort_values(["entry", "symbol"])
            .head(1)
        )
        if reference_row.empty:
            raise RuntimeError("No HW4 test observation found for the reference.")

        row = reference_row.iloc[0]
        request = {"score_rank": float(row["score_rank"])}
        prediction = float(final_model.predict(reference_row[["score_rank"]])[0])
        if not math.isfinite(prediction):
            raise RuntimeError("The reference prediction is not finite.")

        reference = {
            "hw4_commit": HW4_COMMIT,
            "symbol": str(row["symbol"]),
            "signal_date": row["signal"].date().isoformat(),
            "entry_date": row["entry"].date().isoformat(),
            "input": request,
            "expected_prediction": prediction,
            "training_rows": len(training),
        }

        joblib.dump(final_model, OUTPUT_DIR / "hw4_model.joblib")
        for filename, data in (
            ("example_request.json", request),
            ("reference_prediction.json", reference),
        ):
            (OUTPUT_DIR / filename).write_text(
                json.dumps(data, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        print(json.dumps(reference, indent=2))
        print(f"\nExported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
