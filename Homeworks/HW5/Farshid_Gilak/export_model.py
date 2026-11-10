"""Export the fitted HW4 pipeline served by this service, plus a reference case.

Run with the HW4 training environment (the repository .venv, which also has duckdb):
    ../../../.venv/bin/python export_model.py
It reuses the HW4 modules unchanged and refits only the last walk-forward model of the
HW4 final test, which uses the same training rows as in the HW4 notebook.
"""

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
DEV_SIGNAL = pd.Timestamp("2012-12-31")
TEST_SIGNAL = pd.Timestamp("2016-03-31")
LAST_SIGNAL = pd.Timestamp("2026-09-29")
ALPHA = 1e7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw4-dir", type=Path, default=ROOT.parents[1] / "HW4/Farshid_Gilak")
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT.parents[2] / "Lectures/modeling/data/equities_top2000.duckdb",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.hw4_dir.resolve()))
    data = importlib.import_module("hw4_data")
    models = importlib.import_module("hw4_model")

    market = data.load_market_data(args.database)
    signal_dates = data.weekly_signal_dates(
        market.calendar, "2010-01-01", LAST_SIGNAL, extra=[DEV_SIGNAL, TEST_SIGNAL]
    )
    ranked = data.cross_sectional_ranks(data.build_panel(market, signal_dates))
    fit_date = models.fit_dates(ranked.date.unique(), TEST_SIGNAL, LAST_SIGNAL)[-1]
    scores, audit, fitted = models.walk_forward(
        ranked,
        models.make_model("ridge", alpha=ALPHA),
        fit_date,
        LAST_SIGNAL,
        target="rank",
        fit_schedule=pd.DatetimeIndex([fit_date]),
    )
    pipeline, audit = fitted[0], audit.iloc[0]
    model_version = f"hw4-ridge-alpha1e7-rank-fit{fit_date.date()}"

    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    nullable = [
        data.NUMERIC_FEATURES[i] for i in numeric.named_steps["impute"].indicator_.features_
    ]
    sectors = pipeline.named_steps["preprocess"].named_transformers_["sector"].categories_[0]

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {"pipeline": pipeline, "model_version": model_version, "features": data.FEATURES}
    joblib.dump(bundle, artifacts / "model.joblib")
    metadata = {
        "model_version": model_version,
        "model": f"Ridge(alpha={ALPHA:g}) on same-date percentile-ranked features",
        "target": "same-date percentile rank of the 90-calendar-day holding return",
        "fit_date": str(fit_date.date()),
        "train_rows": int(audit.train_rows),
        "first_train_signal": str(audit.first_train_signal.date()),
        "last_train_signal": str(audit.last_train_signal.date()),
        "last_train_label_end": str(audit.last_train_label_end.date()),
        "scored_signals": [
            str(audit.first_scored_signal.date()),
            str(audit.last_scored_signal.date()),
        ],
        "features": data.FEATURES,
        "nullable_features": nullable,
        "sectors": sectors.tolist(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    last = scores.loc[scores.date.eq(scores.date.max())]
    top = last.loc[last.score.idxmax()]
    row = ranked.loc[ranked.date.eq(top.date) & ranked.symbol.eq(top.symbol)].iloc[0]
    request = {
        name: (None if pd.isna(row[name]) else float(row[name]))
        for name in data.NUMERIC_FEATURES
    }
    request["sector"] = row["sector"]
    (ROOT / "example_request.json").write_text(json.dumps(request, indent=2) + "\n")
    reference = {
        "source": "hw4_model.walk_forward score of the top-ranked stock on the last HW4 signal",
        "symbol": top.symbol,
        "signal_date": str(top.date.date()),
        "model_version": model_version,
        "prediction": float(top.score),
    }
    (ROOT / "tests/hw4_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps({**metadata, "reference": reference}, indent=2))


if __name__ == "__main__":
    main()
