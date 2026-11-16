"""Recreate and serialize HW4's final 2026-06-30 quarterly fitted pipeline.

Run locally with the HW4 feature cache and saved score files. The service
itself uses only model.joblib and never opens the cache or course database.
"""

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from hw4_pipeline import FEATURES, make_model

HERE = Path(__file__).resolve().parent
HW4 = HERE.parents[1] / "HW4" / "Xiaofei_Yan" / "outputs" / "final"
FIT_DATE = pd.Timestamp("2026-06-30")
VERSION = "hw4-quarterly-ridge-2026-06-30"


def main():
    """Fit using the original matured labels, then check a saved HW4 score."""
    cache_path = HW4 / "features.duckdb"
    prediction_dir = HW4 / "predictions"
    settings = json.loads((prediction_dir / "configuration.json").read_text())
    assert settings["features"] == FEATURES
    assert settings["parameters"] == {"model__alpha": 1000}
    allowed = settings["allowed_symbols"]
    fit_audit = pd.read_csv(prediction_dir / "fit_audit.csv")
    recorded = fit_audit.loc[fit_audit.fit_date.eq(str(FIT_DATE.date()))].iloc[0]
    columns = ", ".join(FEATURES)

    with duckdb.connect(str(cache_path), read_only=True) as cache:
        train = cache.execute(
            f"SELECT date, label_end, target, {columns} FROM features "
            "WHERE date < ? AND label_end < ? AND target IS NOT NULL "
            "AND isfinite(target) AND symbol IN (SELECT UNNEST(?))",
            [FIT_DATE.to_pydatetime(), FIT_DATE.to_pydatetime(), allowed],
        ).fetchdf()
        if len(train) != int(recorded.training_rows):
            raise ValueError("Training row count differs from HW4's fit audit")
        if pd.Timestamp(train.label_end.max()) != pd.Timestamp(recorded.last_label_end):
            raise ValueError("Training label cutoff differs from HW4's fit audit")
        model = make_model(alpha=1000).fit(train[FEATURES], train.target)
        del train

        saved_scores = pd.read_csv(prediction_dir / f"scores_{FIT_DATE.date()}.csv")
        cache.register("saved_score_keys", saved_scores[["date", "symbol"]])
        candidates = cache.execute(
            f"SELECT f.date, f.symbol, {', '.join('f.' + c for c in FEATURES)} "
            "FROM features f JOIN saved_score_keys k "
            "ON f.date = CAST(k.date AS DATE) AND f.symbol = k.symbol "
            "ORDER BY f.date, f.symbol"
        ).fetchdf()

    candidates = candidates.dropna(subset=FEATURES)
    candidates = candidates.loc[np.isfinite(candidates[FEATURES].to_numpy()).all(axis=1)]
    if candidates.empty:
        raise ValueError("No complete finite HW4 score row for the example request")
    example = candidates.iloc[[0]]
    source = saved_scores.loc[
        saved_scores.date.eq(str(example.iloc[0].date.date()))
        & saved_scores.symbol.eq(example.iloc[0].symbol)
    ].iloc[0]
    prediction = float(model.predict(example[FEATURES])[0])
    if not np.isclose(prediction, float(source.score), rtol=1e-7, atol=1e-9):
        raise AssertionError(f"Exported pipeline {prediction} differs from HW4 {source.score}")

    joblib.dump(model, HERE / "model.joblib")
    request = {"features": [float(example.iloc[0][name]) for name in FEATURES]}
    (HERE / "example_request.json").write_text(json.dumps(request, indent=2) + "\n")
    reference = {
        "model_version": VERSION,
        "fit_date": str(FIT_DATE.date()),
        "prediction_horizon": "next-session entry to first observed close on/after 90 calendar days",
        "features": FEATURES,
        "example_date": str(example.iloc[0].date.date()),
        "example_symbol": str(example.iloc[0].symbol),
        "expected_prediction": float(source.score),
        "reproduced_prediction": prediction,
        "source": f"HW4 outputs/final/predictions/scores_{FIT_DATE.date()}.csv",
        "training_rows": int(recorded.training_rows),
    }
    (HERE / "model_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Saved {VERSION}: {prediction:.12g} matches HW4 score {float(source.score):.12g}")


if __name__ == "__main__":
    main()
