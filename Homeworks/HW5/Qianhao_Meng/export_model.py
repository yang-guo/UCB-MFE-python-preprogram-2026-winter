"""Export the HW4 Ridge comparison model, refit exactly as on 2014-12-31, for the HW5 service.

Run from the course repository root with the HW4 training environment:
    .venv/bin/python Homeworks/HW5/Qianhao_Meng/export_model.py
Reads the HW4 database once, offline. The service never touches it.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent
HW4_DIR = ROOT.parents[1] / "HW4" / "Qianhao_Meng"
sys.path.insert(0, str(HW4_DIR))
import data  # noqa: E402
import model  # noqa: E402

MODEL_VERSION = "hw4-ridge-ranks-20141231-v1"
HW4_COMMIT = "80640fc7"  # branch hw_4_Qianhao_Meng, the submitted HW4 code
ALPHA = 100.0  # hw4.ALPHA, the Ridge penalty of the HW4 development comparison
FIRST_SCORED_YEAR = 2015
FIT_DATE = pd.Timestamp("2014-12-31")
ARTIFACTS = ROOT / "artifacts"
# The 2014-12-31 row of "Ridge weights learned at each fit" in HW4 development.md.
HW4_RECORD = {"training_rows": 59_183, "shares": [0.31, 0.00, 0.00, 0.69]}


def load_ranked_panel(database):
    """Rebuild the HW4 development panel with its ranks, reading nothing after DEV_END."""
    # The last argument only starts the daily quote matrix, which this script discards.
    panel, market, _, _ = data.load_research_data(database, data.DEV_END, "2015-12-01")
    signals = data.monthly_signals(market.index)
    # HW4's rule: the signal before the first session of the first scored year.
    fit_date = signals[signals < f"{FIRST_SCORED_YEAR}-01-01"][-1]
    if fit_date != FIT_DATE:
        raise ValueError(f"HW4's rule selects {fit_date:%Y-%m-%d}, not {FIT_DATE:%Y-%m-%d}.")
    return model.add_ranks(panel), fit_date


def training_rows(ranked, fit_date):
    """Select what walk_forward_scores fitted on: outcomes that had ended before the fit date."""
    return ranked.loc[ranked.label_end.lt(fit_date) & ranked.target.notna()]


def check_against_hw4_record(training, fitted):
    """Stop unless this refit reproduces the training size and weight shares recorded in HW4."""
    shares = fitted.coef_ / fitted.coef_.sum()
    if len(training) != HW4_RECORD["training_rows"]:
        raise ValueError(
            f"{len(training):,} training rows, HW4 recorded {HW4_RECORD['training_rows']:,}."
        )
    if not np.allclose(shares, HW4_RECORD["shares"], atol=0.006):
        raise ValueError(
            f"Shares {np.round(shares, 3)} differ from HW4's {HW4_RECORD['shares']}."
        )
    return shares


def reference_example(ranked, fitted, fit_date):
    """Record the first signal this fit scored in HW4, with the score HW4's own code gives it."""
    scored, audit = model.walk_forward_scores(ranked, ALPHA, [fit_date])
    first = scored.loc[scored.date.eq(fit_date)].sort_values("symbol").iloc[0]
    row = ranked.loc[ranked.date.eq(fit_date) & ranked.symbol.eq(first.symbol), model.RANKS]
    direct = float(fitted.predict(row)[0])
    if not np.isclose(direct, first.score, rtol=0, atol=1e-9):
        raise ValueError(
            f"Frozen model gives {direct}, HW4 walk-forward scoring gave {first.score}."
        )
    example = {name: float(value) for name, value in row.iloc[0].items()}
    reference = {
        "symbol": first.symbol,
        "signal_date": f"{fit_date:%Y-%m-%d}",
        "input": example,
        "expected_prediction": direct,
        "source": "model.walk_forward_scores on the HW4 development panel, 2014-12-31 fit, "
        "recorded before the model was wrapped in the API",
    }
    return example, reference, audit.iloc[0]


def metadata(training, fitted, shares, fit_date, audit, database):
    """Describe what was exported and how to reproduce it."""
    limits = training.target.quantile([0.01, 0.99])
    return {
        "model_version": MODEL_VERSION,
        "model": "sklearn Ridge(alpha=100.0, positive=True) from HW4 model.fit_weights",
        "role_in_hw4": "comparison model of development.md, refit before each validation year",
        "hw4_source": f"branch hw_4_Qianhao_Meng, commit {HW4_COMMIT}, "
        "model.fit_weights and model.walk_forward_scores",
        "reproduction_check": "fit date, training rows and rounded weight shares match the audit "
        "values recorded in HW4 development.md. The HW4 source files are not verified against "
        "the commit by this script",
        "fit_date": f"{fit_date:%Y-%m-%d}",
        "training_rows": int(len(training)),
        "last_label_end": f"{audit.last_label_end:%Y-%m-%d}",
        "target": "gross adjusted-price return from the next session's close to the first quoted "
        "close on or after 90 calendar days, clipped to the training 1% and 99% quantiles",
        "target_clip": [float(limits.iloc[0]), float(limits.iloc[1])],
        "features": model.RANKS,
        "feature_rule": "same-date percentile ranks among eligible stocks, higher is better, "
        "low volatility already reversed, missing component = 0.5",
        "coef": fitted.coef_.tolist(),
        "intercept": float(fitted.intercept_),
        "weight_shares": np.round(shares, 4).tolist(),
        "database": Path(database).name,
        "packages": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_and_verify(fitted, meta, example, reference):
    """Write the bundle, metadata, example request and reference, then reload and check."""
    ARTIFACTS.mkdir(exist_ok=True)
    bundle = {"model": fitted, "model_version": MODEL_VERSION, "features": model.RANKS}
    joblib.dump(bundle, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (ARTIFACTS / "reference_prediction.json").write_text(json.dumps(reference, indent=2) + "\n")
    (ROOT / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    reloaded = joblib.load(ARTIFACTS / "model.joblib")
    frame = pd.DataFrame([example])[reloaded["features"]]
    prediction = float(reloaded["model"].predict(frame)[0])
    if not np.isclose(prediction, reference["expected_prediction"], rtol=0, atol=1e-12):
        raise ValueError(
            f"Reloaded bundle predicts {prediction}, not {reference['expected_prediction']}."
        )
    print(
        f"Saved {MODEL_VERSION}: {meta['training_rows']:,} rows, shares {meta['weight_shares']}"
    )
    print(f"Example {reference['symbol']} on {reference['signal_date']} -> {prediction:.6f}")


def main():
    database = Path(sys.argv[1]) if len(sys.argv) > 1 else data.DATABASE
    ranked, fit_date = load_ranked_panel(database)
    training = training_rows(ranked, fit_date)
    fitted = model.fit_weights(training, ALPHA)
    shares = check_against_hw4_record(training, fitted)
    example, reference, audit = reference_example(ranked, fitted, fit_date)
    meta = metadata(training, fitted, shares, fit_date, audit, database)
    save_and_verify(fitted, meta, example, reference)


if __name__ == "__main__":
    main()
