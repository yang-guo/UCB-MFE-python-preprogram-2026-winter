"""One-off export of the HW4 blend into artifacts/hw4_blend_v1.joblib.

Needs the HW4 environment (the repository .venv), my HW4 code and its cached
panel in outputs/. The service never runs this file. From this folder:

    ../../../.venv/Scripts/python freeze_artifact.py --hw4-dir <folder with HW4 code>
    ../../../.venv/Scripts/python freeze_artifact.py --show

The first form fits the percentile reference on the last HW4 signal date with
a full tradable universe, checks the pipeline against HW4's own blend scores
for every one of those stocks, then writes the artifact, the test reference
files and example_request.json. --show prints what is inside the artifact.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

THIS_DIR = Path(__file__).parent.resolve()
REPO_TOP = THIS_DIR.parent.parent.parent
sys.path.insert(0, str(THIS_DIR))
from blend_model import CrossSectionPercentile, EqualWeightScore  # noqa: E402

BUNDLE_PATH = THIS_DIR / "artifacts" / "hw4_blend_v1.joblib"
VERSION_TAG = "blend20-sep25-2026-v1"
EXAMPLE_SYMBOL = "INCY"  # the highest blend score on the reference date


def build_pipeline(directions):
    return Pipeline(
        [("pct", CrossSectionPercentile(directions=directions)), ("blend", EqualWeightScore())]
    )


def export(hw4_dir):
    # course_data lives in the repo, HW4 code may live in a separate checkout
    sys.path.insert(0, str(REPO_TOP / "Lectures" / "modeling"))
    sys.path.insert(0, str(hw4_dir))
    import data_prep
    import signal_model as sm

    columns = list(sm.TILTS)
    directions = tuple(sm.TILTS.values())
    weekly, _, _ = data_prep.load_or_build(data_prep.BIG_DB)
    per_day = weekly[weekly.tradable].groupby("date").size()
    ref_day = per_day.index.max()
    day_rows = weekly[weekly.date == ref_day]

    # HW4's own numbers, computed by HW4 code before anything here touches them
    view = sm.percentile_view(day_rows)
    hw4 = sm.fixed_rule_score(view, "blend").set_index("symbol").score
    raw = view.set_index("symbol")[columns]

    pipe = build_pipeline(directions).fit(raw)
    mine = pd.Series(pipe.predict(raw), index=raw.index)
    gap = float((mine - hw4.loc[mine.index]).abs().max())
    if gap > 1e-12:
        raise SystemExit(f"pipeline disagrees with HW4 by {gap}")
    print(f"{len(raw)} stocks on {ref_day.date()}, largest gap vs HW4 = {gap:.2e}")

    frozen_choice = json.loads((Path(hw4_dir) / "results" / "frozen_choice.json").read_text())
    bundle = {
        "pipeline": pipe,
        "model_version": VERSION_TAG,
        "features": columns,
        "reference_date": str(ref_day.date()),
        "n_reference": int(len(raw)),
        "hw4_frozen_choice": frozen_choice,
        "built_with": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    BUNDLE_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)

    def as_json(row):
        return {c: (None if pd.isna(row[c]) else float(row[c])) for c in columns}

    example = as_json(raw.loc[EXAMPLE_SYMBOL])
    (THIS_DIR / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    reference = {
        "symbol": EXAMPLE_SYMBOL,
        "signal_date": str(ref_day.date()),
        "model_version": VERSION_TAG,
        "inputs": example,
        "hw4_score": float(hw4.loc[EXAMPLE_SYMBOL]),
        "source": "HW4 signal_model.fixed_rule_score(percentile_view(...), 'blend'), "
        "recorded by freeze_artifact.py before the API existed",
    }
    tests_dir = THIS_DIR / "tests"
    (tests_dir / "frozen_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    table = raw.assign(hw4_score=hw4.loc[raw.index]).reset_index()
    table.to_csv(tests_dir / "sep25_cross_section.csv", index=False, float_format="%.17g")
    print(f"wrote {BUNDLE_PATH.name}, example_request.json and two files in tests/")


def show():
    bundle = joblib.load(BUNDLE_PATH)  # my own file, built by export() above
    for key, value in bundle.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--hw4-dir", type=Path, default=THIS_DIR.parent.parent / "HW4" / "Arav_Shanghavi"
    )
    parser.add_argument("--show", action="store_true", help="print the saved artifact and stop")
    args = parser.parse_args()
    show() if args.show else export(args.hw4_dir.resolve())
