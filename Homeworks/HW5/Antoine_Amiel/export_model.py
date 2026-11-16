"""Export the model served by HW5 from the HW4 code and database.

The served model is the last quarterly refit of the HW4 final test: the Ridge
pipeline (median imputer with missing-value indicators, then Ridge alpha=1000)
fitted after the close of 2026-07-01 and used for every signal from 2026-07-01
to 2026-09-30. It is rebuilt with the unchanged HW4 functions
(``WalkForwardScorer.model_for``) and must use exactly the training rows that
HW4 recorded in ``results/final/fit_audit.csv``.

The script also records, with the HW4 ``predict`` function, reference
predictions for real stocks on the last signal date of the test. The tests
compare the API with these values.

Run it once, from the repository root, in the course environment that trained
HW4 (``uv sync --locked``), with ``equities_top2000.duckdb`` in
``Lectures/modeling/data/``:

    uv run --locked python Homeworks/HW5/Antoine_Amiel/export_model.py
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
DEFAULT_HW4 = HERE.parents[1] / "HW4" / "Antoine_Amiel"
REFERENCE_SYMBOL = "AAPL"


def main(hw4_dir: Path) -> None:
    sys.path.insert(0, str(hw4_dir))
    from hw4.data import load_market_data
    from hw4.features import RANK_FEATURES, FeatureStore
    from hw4.model import WalkForwardScorer, build_panel, glitch_flags, predict, training_dates
    from run_final_test import TEST_END, TEST_START, quarterly_fit_dates

    config = json.loads((hw4_dir / "results/chosen_config.json").read_text())
    audit = pd.read_csv(hw4_dir / "results/final/fit_audit.csv", parse_dates=["fit_date"])

    # Same data, features, training panel and refit schedule as run_final_test.py.
    md = load_market_data(end=str(TEST_END.date()))
    store = FeatureStore(md)
    glitch = glitch_flags(store)
    panel = build_panel(store, training_dates(store.calendar, "2010-01-01", TEST_END))
    fit_dates = quarterly_fit_dates(store.calendar, TEST_START, TEST_END)
    scorer = WalkForwardScorer(
        store, panel, config["model"], fit_dates, config["max_rank"], glitch
    )

    # Last signal date of the test (its entry session is 2026-09-30).
    signal = store.calendar[store.calendar.get_loc(TEST_END) - 1]
    fit_date, model = scorer.model_for(signal)
    fitted = scorer.audit[-1]
    recorded = audit.iloc[-1]
    assert fit_date == fit_dates[-1] == recorded.fit_date, (fit_date, recorded.fit_date)
    assert fitted["train_rows"] == recorded.train_rows, (
        fitted["train_rows"],
        recorded.train_rows,
    )

    # Reference rows: the HW4 scoring cross-section of that date (top 1000, data screen applied).
    scores = scorer.scores(signal).set_index("symbol").score
    xs = store.cross_section([signal], max_rank=config["max_rank"])
    xs = xs.loc[~xs.symbol.map(glitch.loc[signal]).fillna(False).to_numpy()].reset_index(
        drop=True
    )
    xs["hw4_prediction"] = predict(model, xs)
    assert np.array_equal(xs.hw4_prediction.to_numpy(), scores.loc[xs.symbol].to_numpy())

    imputer = model.named_steps["simpleimputer"]
    nullable = [RANK_FEATURES[i] for i in imputer.indicator_.features_]
    with_null = xs.loc[xs[nullable].isna().any(axis=1) & xs.symbol.ne(REFERENCE_SYMBOL)]
    with_null = with_null.sort_values("symbol")
    cases = [xs.loc[xs.symbol.eq(REFERENCE_SYMBOL)].iloc[0], with_null.iloc[0]]

    def features(row):
        # float32 -> float is exact, so the API rebuilds the same float32 input.
        return {c: None if np.isnan(row[c]) else float(row[c]) for c in RANK_FEATURES}

    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    artifact = out / "model.joblib"
    joblib.dump(model, artifact)
    version = f"hw4-ridge-fit-{fit_date.date()}"
    metadata = {
        "model_version": version,
        "artifact": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "model": "Pipeline(SimpleImputer(strategy='median', add_indicator=True), Ridge(alpha=1000))",
        "hw4_fit": {
            "fit_date": str(fit_date.date()),
            "serves_signals": f"{fit_date.date()} to {TEST_END.date()} in the HW4 final test",
            "train_rows": int(fitted["train_rows"]),
            "first_training_signal": str(fitted["first_signal"].date()),
            "last_training_signal": str(fitted["last_signal"].date()),
            "last_label_end": str(fitted["last_label_end"].date()),
        },
        "target": "same-date percentile rank of the 90-day forward return",
        "features": RANK_FEATURES,
        "nullable_features": nullable,
        "input_representation": "numpy float32 array, one row, columns in 'features' order, NaN = missing",
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "exported_on": str(date.today()),
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    reference = {
        "source": (
            "HW4 code (hw4.model.predict with the 2026-07-01 fit) on the HW4 scoring "
            "cross-section of the last test signal date, written by export_model.py"
        ),
        "model_version": version,
        "cases": [
            {
                "signal_date": str(signal.date()),
                "symbol": row.symbol,
                "features": features(row),
                "hw4_prediction": float(row.hw4_prediction),
            }
            for row in cases
        ],
    }
    (HERE / "tests/reference_predictions.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    (HERE / "example_request.json").write_text(json.dumps(features(cases[0]), indent=2) + "\n")
    invalid = {**features(cases[0]), "r_mom_21": "0.5"}  # a numeric string: must be rejected
    (HERE / "evidence").mkdir(exist_ok=True)
    (HERE / "evidence/invalid_request.json").write_text(json.dumps(invalid, indent=2) + "\n")

    print(f"{version}: {fitted['train_rows']:,} training rows")
    sys.path.insert(0, str(HERE))
    from model_service import NULLABLE_FEATURES

    if set(nullable) != NULLABLE_FEATURES:
        print("WARNING: update NULLABLE_FEATURES and InputData. Nullable in the fitted model:")
        print(sorted(nullable))
    for row in cases:
        print(f"{signal.date()} {row.symbol}: {row.hw4_prediction:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw4-dir", type=Path, default=DEFAULT_HW4)
    main(parser.parse_args().hw4_dir.resolve())
