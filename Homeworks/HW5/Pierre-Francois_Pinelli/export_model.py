"""Save the fitted HW4 model for the API. Run once; the service never needs this file.

We run the HW4 code as it is: same panel, same universe, same ranks, same elastic net settings.
We keep the last quarterly refit of the HW4 test (fit date 2026-06-30), and one real row
with the score HW4 gave it, to check the API against it later.

Needs the HW4 folder next to HW5, the HW4 database (or its cached panel) and duckdb.
"""

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
HW4 = ROOT.parents[1] / "HW4" / "Pierre-Francois_Pinelli"
sys.path.insert(0, str(HW4))

import hw4_data as data  # noqa: E402
import hw4_model as model  # noqa: E402

ALPHA, L1_RATIO = 0.0003, 0.8  # chosen in the HW4 development backtest
FIT_DATE = pd.Timestamp("2026-06-30")  # last quarterly refit of the HW4 test
LAST_SIGNAL = pd.Timestamp("2026-08-31")  # last signal of the HW4 test
EXAMPLE_SYMBOL = "AAPL"
MODEL_VERSION = "hw4-elastic-net-fit-2026-06-30"


def main():
    # same steps as the HW4 notebook: panel, 1,000 biggest stocks, ranks
    panel = data.load_panel(cache=HW4 / "outputs" / f"panel_{data.DB.stem}.pkl")
    panel["eligible"] = panel.eligible & (panel.universe_rank <= 1000)
    panel = model.add_ranks(panel)

    # the HW4 function fits the model and scores the months after the fit
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    signals = dates[(dates >= FIT_DATE) & (dates <= LAST_SIGNAL)]
    scores, models, audit = model.walk_forward(panel, [FIT_DATE], signals, ALPHA, L1_RATIO)
    fitted = models[FIT_DATE]
    features = list(model.Z_COLS)
    assert list(fitted.feature_names_in_) == features

    # one real row, with the score that HW4 itself computed for it
    is_example = (panel.date == LAST_SIGNAL) & (panel.symbol == EXAMPLE_SYMBOL) & panel.eligible
    row = panel.loc[is_example, features].iloc[0]
    hw4_score = scores.loc[
        (scores.date == LAST_SIGNAL) & (scores.symbol == EXAMPLE_SYMBOL), "score"
    ].iloc[0]
    example = {name: float(row[name]) for name in features}

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = {"model": fitted, "model_version": MODEL_VERSION, "features": features}
    joblib.dump(bundle, artifacts / "model.joblib")

    fit = audit.iloc[0]
    metadata = {
        "model_version": MODEL_VERSION,
        "model": "sklearn ElasticNet, the last quarterly refit of the HW4 test",
        "alpha": ALPHA,
        "l1_ratio": L1_RATIO,
        "fit_date": str(FIT_DATE.date()),
        "training_rows": int(fit.rows),
        "last_training_signal": str(fit.last_training_signal.date()),
        "last_label_end": str(fit.last_label_end.date()),
        "features": features,
        "input_units": "cross-sectional rank of each characteristic, from -0.5 to 0.5",
        "target": "90-day return minus the average 90-day return of the month, as a fraction",
        "intercept": float(fitted.intercept_),
        "coefficients": {name: float(c) for name, c in zip(features, fitted.coef_)},
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    reference = {
        "source": "score from hw4_model.walk_forward (HW4 code), before any API",
        "signal_date": str(LAST_SIGNAL.date()),
        "symbol": EXAMPLE_SYMBOL,
        "input": example,
        "expected_prediction": float(hw4_score),
    }
    (artifacts / "reference_prediction.json").write_text(json.dumps(reference, indent=2) + "\n")
    (ROOT / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")

    print(f"saved {MODEL_VERSION}: {int(fit.rows):,} training rows")
    print(f"{EXAMPLE_SYMBOL} on {LAST_SIGNAL.date()}: HW4 score = {hw4_score:.10f}")


if __name__ == "__main__":
    main()
