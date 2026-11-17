"""One-off export of the frozen HW4 model (run once; NOT used by the service).

What it does
    1. Imports my HW4 code (branch hw_4_Antonio_Rivera, PR #257) from --hw4-dir.
    2. Loads the HW4 final feature panel (outputs/panel_final.parquet, gitignored).
    3. Refits the frozen HW4 model exactly as hw4_walkforward.walk_forward_scores does
       at the LAST quarterly refit of the HW4 final test (first session of 2026-Q3):
       cd.make_pipeline(Ridge(alpha=100), F1_baseline) on target_rank, train rows =
       eligible, date >= 2010-01-01, target notna, label_end < fit date.
       This is the model whose scores HW4 actually used most recently, so it is the one
       to serve next.
    4. Checks the refit against scores HW4 recorded (scores_final_ridge_F1_baseline.parquet)
       for every row of that last quarter, then saves:
         model/pipeline.joblib   fitted pipeline (preprocessing + Ridge)
         model/metadata.json     version, fit date, feature order, library versions
         model/reference.json    one real HW4 input row + the score HW4 recorded for it
       and example_request.json (the same row, in the API's JSON format).

Run (course env, which matches the HW4 training versions; needs pyarrow + duckdb):
    git archive hw_4_Antonio_Rivera Homeworks/HW4/Antonio_Rivera | tar -x -C "$TMP"
    ln -s "$PWD/Lectures" "$TMP/Lectures"
    rmdir "$TMP/Homeworks/HW4/Antonio_Rivera/outputs" 2>/dev/null
    ln -s "$PWD/Homeworks/HW4/Antonio_Rivera/outputs" "$TMP/Homeworks/HW4/Antonio_Rivera/outputs"
    .venv/bin/python Homeworks/HW5/Antonio_Rivera/export_model.py \
        --hw4-dir "$TMP/Homeworks/HW4/Antonio_Rivera"
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hw5_transformers  # noqa: E402  (the class the pickle must reference)

TEST_START, TEST_END = "2016-04-01", "2026-09-30"  # HW4 final test window
MODEL_NAME, FEATURE_SET = "ridge", "F1_baseline"  # frozen HW4 choice


def main(hw4_dir: Path) -> None:
    sys.path.insert(0, str(hw4_dir))
    import hw4_build_cache as hc
    import hw4_model as hm
    import hw4_walkforward as wf

    cd = wf.cd
    # Build the pipeline with the duckdb-free copy of the clipper (same code), so the
    # pickled class path is hw5_transformers.QuantileClipper.
    cd.QuantileClipper = hw5_transformers.QuantileClipper

    numeric = list(hm.FEATURE_SETS[FEATURE_SET])
    cols = ["date", "symbol", "eligible", "target", "label_end", "sector"] + numeric
    panel = pd.read_parquet(hc.cache_path("final"), columns=cols)
    elig = panel.loc[panel.eligible].reset_index(drop=True)

    fit_date = wf.quarter_fit_dates(elig.date.unique(), TEST_START, TEST_END)[-1]
    train = elig.loc[
        elig.date.ge(wf.TRAIN_START) & elig.target.notna() & elig.label_end.lt(fit_date)
    ]
    assert train.label_end.max() < fit_date
    train = hm.add_target_rank(train)
    pipe = cd.make_pipeline(hm.make_model(MODEL_NAME), numeric).fit(train, train.target_rank)
    clipper = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["clip"]
    assert type(clipper).__module__ == "hw5_transformers"

    # Compare with what HW4 recorded for the whole last quarter.
    scores = pd.read_parquet(hc.OUT / f"scores_final_{MODEL_NAME}_{FEATURE_SET}.parquet")
    scores = scores.loc[scores.fit_date.eq(fit_date)]
    assert len(scores), "HW4 scores have no rows for this fit date"
    rows = elig.merge(scores, on=["date", "symbol"], how="inner")
    max_diff = float(np.max(np.abs(pipe.predict(rows) - rows.score.to_numpy())))
    print(
        f"fit {fit_date.date()}: {len(train):,} train rows; {len(rows):,} HW4 scores, "
        f"max |refit - HW4| = {max_diff:.2e}"
    )
    assert max_diff < 1e-9

    # Reference row: latest date, first symbol alphabetically with all features present.
    complete = rows.dropna(subset=numeric)
    ref = complete.loc[complete.date.eq(complete.date.max())].sort_values("symbol").iloc[0]
    features = {c: float(ref[c]) for c in numeric} | {"sector": str(ref["sector"])}

    out = HERE / "model"
    out.mkdir(exist_ok=True)
    version = f"hw4-ridge-F1-rank-fit{fit_date.date()}"
    joblib.dump(pipe, out / "pipeline.joblib")
    metadata = {
        "model_version": version,
        "source": "HW4 branch hw_4_Antonio_Rivera (PR #257), hw4_walkforward.py",
        "model": "Ridge(alpha=100) on cd.make_pipeline F1_baseline",
        "target": "target_rank: per-date percentile rank of 90-calendar-day forward return",
        "fit_date": str(fit_date.date()),
        "train_rows": int(len(train)),
        "train_dates": [str(train.date.min().date()), str(train.date.max().date())],
        "numeric_features": numeric,
        "categorical_features": ["sector"],
        "sectors": sorted(train.sector.dropna().unique().tolist()),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reference = {
        "source": "HW4 outputs/scores_final_ridge_F1_baseline.parquet (score recorded by "
        "hw4_walkforward.py during the HW4 final run)",
        "symbol": str(ref["symbol"]),
        "date": str(ref["date"].date()),
        "fit_date": str(fit_date.date()),
        "features": features,
        "hw4_score": float(ref["score"]),
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (HERE / "example_request.json").write_text(json.dumps(features, indent=2) + "\n")
    size = (out / "pipeline.joblib").stat().st_size
    print(
        f"saved {version}: pipeline.joblib {size:,} bytes; reference {ref['symbol']} "
        f"{ref['date'].date()} score {ref['score']:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw4-dir", type=Path, required=True)
    main(parser.parse_args().hw4_dir)
