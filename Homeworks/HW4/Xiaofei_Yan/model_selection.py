"""Select a Ridge penalty using chronological development folds.

Adapted from Lecture 8/2_research_workflow.ipynb: reuse date_folds and
GridSearchCV and the course preprocessing pipeline with our 20 features.
The final test period is not used for parameter selection.
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from feature_panel import DEFAULT_DATABASE, FEATURES, iter_feature_batches
from course_data import date_folds, make_pipeline
from price_quality import ZERO_VOLUME_SCALE, REVIEW_SCALE
from common_stock_screen import POLICY_VERSION, load_security_review


FIRST_FIT = pd.Timestamp("2016-03-31")
VALIDATION_YEARS = (2012, 2013, 2014)
PARAMETER_GRID = {"model__alpha": [1, 100, 1000]}


def load_development(database=DEFAULT_DATABASE, batch_size=100):
    """Collect eligible development signals with outcomes known before first fit.

    Read buffer-period quotes only to finish earlier targets. A late-2015
    signal with an exit on/after March 31 is excluded from this initial fit.
    Earlier quotes remain available inside the feature builder for lookbacks.
    """
    pieces = []
    columns = ["date", "symbol", "label_end", "target", *FEATURES]
    for batch_number, panel in enumerate(
        iter_feature_batches(database=database, end="2016-03-30", batch_size=batch_size),
        start=1,
    ):
        usable = (
            panel.eligible
            & panel.date.between("2010-01-01", "2015-12-31")
            & panel.label_end.lt(FIRST_FIT)
            & panel.target.notna()
            & np.isfinite(panel.target)
        )
        pieces.append(panel.loc[usable, columns].copy())
        print(f"Prepared development batch {batch_number}: {int(usable.sum()):,} rows", flush=True)
    if not pieces:
        raise ValueError("No development batches were produced")
    result = pd.concat(pieces, ignore_index=True).sort_values(["date", "symbol"])
    if result.empty:
        raise ValueError("No eligible, mature development samples")
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)


def make_model(parameters=None):
    """Reuse course clipping, median imputation, and scaling before Ridge.

    All preprocessing is fitted separately inside each training fold. Sector
    is omitted because the agreed input set contains 20 numeric features.
    """
    pipeline = make_pipeline(Ridge(), numeric_features=FEATURES, include_sector=False)
    if parameters:
        pipeline.set_params(**parameters)
    return pipeline


def select_parameters(development):
    """Compare three Ridge penalties using mature-label yearly folds.

    Folds run sequentially to avoid copying the full panel into parallel jobs.
    """
    folds = date_folds(development, validation_years=VALIDATION_YEARS)
    search = GridSearchCV(
        make_model(),
        PARAMETER_GRID,
        cv=folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
        error_score="raise",
        verbose=2,
    )
    search.fit(development[FEATURES], development.target)
    return search


def main():
    """Run development-only selection and save reproducible local outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "development")
    args = parser.parse_args()
    development = load_development(args.database, args.batch_size)
    search = select_parameters(development)
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_)
    results["mean_validation_rmse"] = -results.mean_test_score
    results.to_csv(args.output / "validation_results.csv", index=False)
    configuration = {
        "database": str(args.database.resolve()),
        "features": FEATURES,
        "development_signal_dates": ["2010-01-01", "2015-12-31"],
        "validation_years": list(VALIDATION_YEARS),
        "selection_metric": "mean fold RMSE",
        "model": "quarterly Ridge",
        "security_policy": POLICY_VERSION,
        "preprocessing": "course quantile clipping, median imputation with missing indicators, standard scaling",
        "price_quality_policy": {
            "zero_volume_scale_factor": ZERO_VOLUME_SCALE,
            "unresolved_transition_factor": REVIEW_SCALE,
            "trailing_market_sessions": 253,
            "censor_labels_crossing_flags": True,
            "documentation": "README.md: Price quality policy",
        },
        "parameter_grid": PARAMETER_GRID,
        "chosen_parameters": search.best_params_,
        "validation_rmse": float(-search.best_score_),
        "initial_fit_signal": str(FIRST_FIT.date()),
        "last_training_label_end": str(development.label_end.max().date()),
        "training_rows": len(development),
    }
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    load_security_review(args.database).to_csv(args.output / "security_type_review.csv", index=False)
    joblib.dump(search.best_estimator_, args.output / "initial_model.joblib")
    print("Selected parameters:", search.best_params_)
    print("Mean validation RMSE:", -search.best_score_)


if __name__ == "__main__":
    main()
