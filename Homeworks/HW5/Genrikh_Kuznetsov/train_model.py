"""Rebuild and save the final HW4 model. Run once, offline; the API never runs this.

HW4 refit a Ridge pipeline every quarter. The last refit used for trading was at the
2026-06-30 close; it scored the 2026-07-31 and 2026-08-31 month-end signals (the last
purchase was on 2026-09-01). That fit is the one served here.

Inputs (both produced by the HW4 notebook, both Git-ignored):
  * the cached HW4 feature panel  Homeworks/HW4/Genrikh_Kuznetsov/outputs/panel_equities_top2000.parquet
  * the HW4 test scores            Homeworks/HW4/Genrikh_Kuznetsov/outputs/test_scores.parquet

Outputs (in artifacts/):
  * model.joblib             {"pipeline", "model_version", "features"}
  * model_metadata.json      training window, hyperparameters, library versions, checksum
  * reference_examples.json  inputs + the score the HW4 notebook recorded for them
and example_request.json (a valid request body for the container demo).

Usage:  python train_model.py [--hw4-outputs PATH]
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from hw4_pipeline import FEATURES, make_pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_HW4_OUTPUTS = ROOT.parent.parent / "HW4" / "Genrikh_Kuznetsov" / "outputs"

# Choices made in HW4 (results/validation_rank_ic.csv and the notebook's CONFIG).
TRAIN_START = pd.Timestamp("2010-01-01")
FIT_DATE = pd.Timestamp("2026-06-30")  # last quarterly refit in the HW4 test
ALPHA = 1_000_000  # best mean validation rank IC, 2012-2015
TARGET_KIND = "rank"  # same-day percentile rank of the 90-day return, minus 0.5
MODEL_VERSION = "hw4-ridge-rank-a1e6-fit2026-06-30"
REFERENCE_DATE = pd.Timestamp("2026-08-31")  # last signal this model scored in HW4


def training_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Same filter as HW4 ``fit_and_score``: only labels already known at the fit date."""
    keep = (
        panel.eligible
        & panel.date.ge(TRAIN_START)
        & panel.date.lt(FIT_DATE)
        & panel.label_end.lt(FIT_DATE)
        & panel.target.notna()
    )
    return panel.loc[keep]


def rank_target(rows: pd.DataFrame) -> pd.Series:
    """HW4 ``training_target(rows, "rank")``."""
    return rows.target.groupby(rows.date).rank(pct=True) - 0.5


def to_request(row: pd.Series) -> dict:
    """One panel row -> a JSON request body (missing values become null)."""
    body = {}
    for name in FEATURES:
        value = row[name]
        if name == "sector":
            body[name] = str(value)
        elif pd.isna(value):
            body[name] = None
        elif name == "filing_age_days":
            body[name] = int(value)
        else:
            # float32 -> float64 is exact, so the API sees the same numbers HW4 scored.
            body[name] = float(value)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw4-outputs", type=Path, default=DEFAULT_HW4_OUTPUTS)
    args = parser.parse_args()

    columns = ["date", "symbol", "eligible", "target", "label_end"] + FEATURES
    panel = pd.read_parquet(
        args.hw4_outputs / "panel_equities_top2000.parquet", columns=columns
    )
    hw4_scores = pd.read_parquet(args.hw4_outputs / "test_scores.parquet")

    train = training_rows(panel)
    assert train.label_end.max() < FIT_DATE, "a training label was not yet known at the fit"
    pipeline = make_pipeline(ALPHA).fit(train[FEATURES], rank_target(train))
    print(
        f"Fitted on {len(train):,} rows, {train.date.min().date()} to {train.date.max().date()}"
    )

    # Check: reproduce every score HW4 recorded for this fit date.
    recorded = hw4_scores.loc[hw4_scores.fit_date.eq(FIT_DATE), ["date", "symbol", "score"]]
    scored = recorded.merge(panel[["date", "symbol"] + FEATURES], on=["date", "symbol"])
    assert len(scored) == len(recorded) > 0
    max_diff = float(np.max(np.abs(pipeline.predict(scored[FEATURES]) - scored.score)))
    print(f"Reproduced {len(scored):,} HW4 scores; max abs difference {max_diff:.2e}")
    assert max_diff < 1e-9, "rebuilt model does not match HW4"

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {"pipeline": pipeline, "model_version": MODEL_VERSION, "features": FEATURES}
    joblib.dump(bundle, artifacts / "model.joblib")

    # Reference examples: a complete row and one with missing (null) features,
    # both from the last signal date this model scored in HW4.
    last = scored.loc[scored.date.eq(REFERENCE_DATE)].sort_values("symbol")
    complete = last.loc[last[FEATURES].notna().all(axis=1)]
    has_missing = last.loc[last[FEATURES].isna().any(axis=1)]
    picks = [complete.loc[complete.symbol.eq("AAPL")].iloc[0], has_missing.iloc[0]]
    references = [
        {
            "symbol": row.symbol,
            "signal_date": str(row.date.date()),
            "hw4_score": float(row.score),
            "input": to_request(row),
        }
        for row in picks
    ]
    (artifacts / "reference_examples.json").write_text(
        json.dumps(
            {
                "source": "Homeworks/HW4/Genrikh_Kuznetsov/outputs/test_scores.parquet, "
                f"rows with fit_date {FIT_DATE.date()}, written by the HW4 notebook "
                "(section 5) before any API code existed.",
                "examples": references,
            },
            indent=2,
        )
        + "\n"
    )
    (ROOT / "example_request.json").write_text(
        json.dumps(references[0]["input"], indent=2) + "\n"
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "model": f"Ridge(alpha={ALPHA:g}) after hw4_pipeline preprocessing",
        "target": "same-day percentile rank of the 90-calendar-day forward return, minus 0.5",
        "prediction_meaning": "relative score; higher = expected to rank higher in 90-day "
        "return among that day's eligible stocks. Not a return forecast.",
        "fit_date": str(FIT_DATE.date()),
        "training_rows": int(len(train)),
        "training_signal_dates": [str(train.date.min().date()), str(train.date.max().date())],
        "features": FEATURES,
        "hw4_scores_reproduced": int(len(scored)),
        "hw4_max_abs_difference": max_diff,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "model_sha256": hashlib.sha256((artifacts / "model.joblib").read_bytes()).hexdigest(),
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {MODEL_VERSION} to {artifacts / 'model.joblib'}")


if __name__ == "__main__":
    main()
