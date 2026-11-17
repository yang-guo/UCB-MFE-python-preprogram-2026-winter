"""Fit the HW4 round 1 model again and save it for the HW5 service.

Run from the repository root, with the course environment (it has the HW4 code and duckdb):

    uv run python Homeworks/HW5/Ziang_Gao/export_model.py

The script does the same steps as hw4.ipynb, sections 2, 4 and 7. It reads the HW4 cache
when this pandas version can read it. If not, it builds the same data again from the course
database with the HW4 function `build_panel` (1 to 3 minutes). It makes three files:

    artifacts/model.joblib         fitted pipeline (preprocessing + Ridge), version, features
    artifacts/model_metadata.json  versions, training facts, feature statistics, reference
    example_request.json           one real input row for the API

The reference prediction comes from the HW4 code path (`course_data.make_pipeline`, the
function that hw4.ipynb calls). The script stops if the served pipeline does not give the
same predictions.
"""

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "Lectures" / "modeling").is_dir())
HW4_DIR = ROOT / "Homeworks" / "HW4" / "Ziang_Gao"
CACHE = HW4_DIR / "cache" / "full_panel.pkl"
ARTIFACTS = HERE / "artifacts"

sys.path.insert(0, str(HW4_DIR))
import hw4_features as h  # noqa: E402  (my HW4 code; it also loads the lecture course_data)

from preprocessing import FEATURES, MODEL_VERSION, make_pipeline  # noqa: E402

# The same dates as hw4.ipynb, section 1.
END = "2026-09-30"
FIRST_TEST_DAY = "2016-04-01"
DEV_START, DEV_END = "2010-01-01", "2015-12-31"
TARGET_CLIP = (0.005, 0.995)
PREFERRED_SYMBOLS = ["AAPL", "MSFT", "JPM", "KO"]


def load_panel():
    """Return (panel, market calendar), as hw4.ipynb section 2 makes them."""
    if CACHE.exists():
        try:
            with open(CACHE, "rb") as file:
                data = pickle.load(file)  # my own HW4 cache file
            return data["panel"], data["wide"].index
        except Exception as error:
            # A pickle from one pandas version is not always readable in a different one.
            print(
                f"pandas {pd.__version__} cannot read the HW4 cache ({type(error).__name__})."
            )
    if not Path(h.DB_FULL).exists():
        sys.exit(f"Missing {h.DB_FULL}. Download it from bCourses, then run this again.")
    print("Building the data from the course database with the HW4 code. Wait 1 to 3 minutes.")
    panel, wide, _ = h.build_panel(
        h.DB_FULL, end=END, first_test_day=FIRST_TEST_DAY, verbose=False
    )
    return panel, wide.index


def load_training_data():
    """Return (training rows, test-period rows, fit date), built as in hw4.ipynb."""
    panel, calendar = load_panel()
    panel = panel[h.screen_mask(panel)].reset_index(drop=True)
    dev = panel[panel.date.between(DEV_START, DEV_END)].reset_index(drop=True)
    fit_date = calendar[calendar.searchsorted(pd.Timestamp(FIRST_TEST_DAY)) - 1]
    train = dev[(dev.label_end < fit_date) & dev.target.notna()]
    assert train.date.max() <= pd.Timestamp(DEV_END) and train.label_end.max() < fit_date
    return train, panel[panel.date >= fit_date], fit_date


def pick_reference_row(test_rows, fit_date):
    """One real stock on the first test signal date, with all 12 features present."""
    complete = np.isfinite(test_rows[FEATURES].astype(float)).all(axis=1)
    day = test_rows[complete & (test_rows.date == fit_date)]
    if day.empty:
        day = test_rows[complete]
    known = day[day.symbol.isin(PREFERRED_SYMBOLS)]
    return (known if not known.empty else day).sort_values("symbol").iloc[0]


def feature_stats(train):
    """Training range of each feature. I use it to set the input limits in app.py."""
    quantiles = train[FEATURES].astype(float).quantile([0, 0.001, 0.5, 0.999, 1])
    quantiles.index = ["min", "q0.1%", "median", "q99.9%", "max"]
    stats = quantiles.T
    stats["missing_share"] = train[FEATURES].isna().mean()
    return stats


def main():
    train, test_rows, fit_date = load_training_data()
    low, high = train.target.quantile(list(TARGET_CLIP))
    y = train.target.clip(low, high)
    print(f"fit date {fit_date.date()} | training rows {len(train):,} (hw4.ipynb: 573,544)")

    # 1. The HW4 code path. This is the reference.
    hw4_model = h.cd.make_pipeline(Ridge(alpha=100), FEATURES, include_sector=False)
    hw4_model.fit(train[FEATURES], y)
    # 2. The same pipeline from preprocessing.py. This is the model that the service loads.
    served = make_pipeline(Ridge(alpha=100), FEATURES).fit(train[FEATURES], y)

    gap = np.abs(hw4_model.predict(test_rows[FEATURES]) - served.predict(test_rows[FEATURES]))
    print(f"largest difference, HW4 pipeline against served pipeline: {gap.max():.2e}")
    assert gap.max() < 1e-12, "The served pipeline does not agree with the HW4 pipeline"

    coefficients = pd.Series(served.named_steps["model"].coef_[: len(FEATURES)], index=FEATURES)
    print("Ridge coefficients (compare with hw4.ipynb, section 7):")
    print(coefficients.sort_values().round(5).to_string())

    row = pick_reference_row(test_rows, fit_date)
    example = {name: float(row[name]) for name in FEATURES}
    reference = float(hw4_model.predict(pd.DataFrame([example], columns=FEATURES))[0])

    ARTIFACTS.mkdir(exist_ok=True)
    bundle = {"pipeline": served, "model_version": MODEL_VERSION, "features": FEATURES}
    joblib.dump(bundle, ARTIFACTS / "model.joblib")

    # Read the two files back, as the service does, and check the result again.
    (HERE / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    request = json.loads((HERE / "example_request.json").read_text())
    loaded = joblib.load(ARTIFACTS / "model.joblib")
    again = float(
        loaded["pipeline"].predict(pd.DataFrame([request], columns=loaded["features"]))[0]
    )
    assert abs(again - reference) < 1e-12, (again, reference)

    stats = feature_stats(train)
    metadata = {
        "model_version": MODEL_VERSION,
        "source": "HW4 round 1, hw4.ipynb section 7 (Ridge alpha=100, feature set C)",
        "features": FEATURES,
        "target": "return from the next session's close to the first close on or after "
        "90 calendar days, as a fraction (0.05 = 5%)",
        "units": "momentum and ratio features are fractions; filing_age_days is in days",
        "fit_date": str(fit_date.date()),
        "training_rows": int(len(train)),
        "last_training_signal": str(train.date.max().date()),
        "target_clip": {"low": float(low), "high": float(high)},
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "reference": {
            "symbol": str(row.symbol),
            "signal_date": str(pd.Timestamp(row.date).date()),
            "input": example,
            "prediction": reference,
            "origin": "course_data.make_pipeline fitted in export_model.py, the same call "
            "as fit_clipped in hw4.ipynb, before the API existed",
        },
        "feature_stats_training": json.loads(stats.to_json(orient="index")),
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nTraining range of each feature (use it for the limits in app.py):")
    print(stats.round(4).to_string())
    print(f"\nreference: {row.symbol} on {metadata['reference']['signal_date']}")
    print(f"reference prediction: {reference!r}")
    print(
        f"scikit-learn {sklearn.__version__} | numpy {np.__version__} | pandas {pd.__version__}"
    )
    for path in (ARTIFACTS / "model.joblib", ARTIFACTS / "model_metadata.json"):
        print(f"saved {path.relative_to(HERE)} ({path.stat().st_size:,} bytes)")
    print("saved example_request.json")


if __name__ == "__main__":
    main()
