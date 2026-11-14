"""Save the HW4 model that is served here, with a reference input and prediction.

Run once, in the HW4 environment (it reads parquet files), from this folder:

    python export_model.py --hw4-outputs ../../HW4/Quentin_Benchetrit/outputs

The model is the last quarterly refit of the HW4 test: the sign-constrained ridge,
fitted after the close of 30 June 2026 on every outcome that had ended by then, and
used by HW4 to score the signals of July to September 2026. The script refits it
with the HW4 code in hw4/, checks that its forecasts equal the ones HW4 recorded in
predictions.parquet, then writes the artifact and the reference example.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from hw4.model import PRIOR_SIGNS, ReturnModel, ridge, training_rows

ROOT = Path(__file__).resolve().parent
FIT_DATE = pd.Timestamp("2026-06-30")
LAST_SIGNAL = pd.Timestamp("2026-09-30")
MODEL_VERSION = "hw4-signed-ridge-2026-06-30"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hw4-outputs", type=Path, default=ROOT / "../../HW4/Quentin_Benchetrit/outputs"
    )
    outputs = parser.parse_args().hw4_outputs

    panel = pd.read_parquet(outputs / "panel.parquet")
    recorded = pd.read_parquet(outputs / "predictions.parquet")

    # Same specification as MODEL in the HW4 notebook (section 3.3).
    model = ReturnModel(
        ridge(alpha=1000.0, positive=True), list(PRIOR_SIGNS), PRIOR_SIGNS, "alpha"
    )
    train = training_rows(panel, FIT_DATE)
    model.fit(train)

    # The refit must reproduce every forecast HW4 made with its 30 June 2026 model.
    scored = panel.loc[panel.eligible & panel.date.between(FIT_DATE, LAST_SIGNAL)]
    hw4 = recorded.loc[recorded.fit_date == FIT_DATE].set_index(["date", "symbol"]).prediction
    mine = pd.Series(
        model.predict(scored), index=pd.MultiIndex.from_frame(scored[["date", "symbol"]])
    )
    assert len(hw4) == len(mine) and hw4.index.sort_values().equals(mine.index.sort_values())
    largest_gap = float((mine - hw4.reindex(mine.index)).abs().max())
    assert largest_gap < 1e-12, largest_gap

    # Reference example: the stock HW4 ranked first on its last signal date,
    # among those with every characteristic available.
    last = scored.loc[scored.date == scored.date.max()].dropna(subset=model.features)
    best = last.iloc[model.predict(last).argmax()]
    request = {name: float(best[name]) for name in model.features}
    reference = {
        "source": "HW4 predictions.parquet, written by walk_forward in the HW4 notebook",
        "signal_date": f"{best.date:%Y-%m-%d}",
        "symbol": best.symbol,
        "hw4_prediction": float(hw4[(best.date, best.symbol)]),
        "features": request,
    }

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {"model": model, "model_version": MODEL_VERSION, "features": model.features}
    joblib.dump(bundle, artifacts / "model.joblib")
    metadata = {
        "model_version": MODEL_VERSION,
        "model": "ReturnModel(ridge(alpha=1000, positive=True), signed features, target='alpha')",
        "fit_date": f"{FIT_DATE:%Y-%m-%d}",
        "training_rows": len(train),
        "training_signals": [f"{train.date.min():%Y-%m-%d}", f"{train.date.max():%Y-%m-%d}"],
        "last_training_outcome": f"{train.label_end.max():%Y-%m-%d}",
        "features": model.features,
        "input_units": "percentile rank among the eligible stocks on the signal date, in (0, 1]",
        "prediction": "90-calendar-day return minus shrunk beta times the S&P 500 TR return",
        "versions": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "check": f"matches the {len(hw4)} HW4 forecasts of this fit, largest gap {largest_gap:.1e}",
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (artifacts / "reference_example.json").write_text(json.dumps(reference, indent=2) + "\n")
    (ROOT / "example_request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(
        f"{MODEL_VERSION}: {len(train):,} training rows, {len(hw4):,} HW4 forecasts reproduced "
        f"(largest gap {largest_gap:.1e}); example {best.symbol} on {best.date:%Y-%m-%d}"
    )


if __name__ == "__main__":
    main()
