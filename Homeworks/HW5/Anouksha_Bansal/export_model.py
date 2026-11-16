"""Re-create and save the HW4 production model (training side; NOT used by the service).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from hw4_model import FEATURES, MODEL_VERSION

HERE = Path(__file__).resolve().parent
HW4_OUT = HERE.parents[1] / "HW4" / "Anouksha_Bansal" / "outputs"
ARTIFACT_DIR = HERE / "artifacts"

# Settings copied from HW4
TRAIN_START = pd.Timestamp("2010-01-01")
MIN_DOLLAR_VOL = 5e6
FIT_DATE = pd.Timestamp("2026-01-02")
TEST_END = pd.Timestamp("2026-09-30")
EXPECTED_N_TRAIN = 1_196_310
RAW_FEATURES = [f.removeprefix("r_") for f in FEATURES]


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.loc[
        panel.date.ge(TRAIN_START)
        & panel.eligible
        & panel.log_dollar_vol_63.ge(np.log(MIN_DOLLAR_VOL))
        & panel.max_abs_ret_63.lt(1.0)
    ].copy()
    g = p.groupby("date")
    for f in RAW_FEATURES:
        p["r_" + f] = (g[f].rank(pct=True) - 0.5).astype("float32")
    p["excess_target"] = p.target - g.target.transform("mean")
    return p.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "impute_missing_rank_as_median",
                SimpleImputer(strategy="constant", fill_value=0.0),
            ),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def main() -> None:
    cols = ["date", "symbol", "eligible", "log_dollar_vol_63", "max_abs_ret_63", "target"]
    cols += ["label_end", *RAW_FEATURES]
    p = prepare(pd.read_parquet(HW4_OUT / "panel.parquet", columns=cols))
    p = p.loc[p.date.le(TEST_END)]

    train = p.loc[p.date.lt(FIT_DATE) & p.label_end.lt(FIT_DATE) & p.target.notna()]
    assert train.label_end.max() < FIT_DATE
    assert len(train) == EXPECTED_N_TRAIN, len(train)

    y = train.excess_target.to_numpy()
    clip = np.nanquantile(y, [0.01, 0.99])
    y = np.clip(y, *clip)

    pipeline = build_pipeline().fit(train[FEATURES].astype("float32"), y)

    recorded = pd.read_parquet(HW4_OUT / "test_predictions.parquet")
    recorded = recorded.loc[
        recorded.fit_date.eq(FIT_DATE), ["date", "symbol", "score", "n_train"]
    ]
    assert recorded.n_train.eq(EXPECTED_N_TRAIN).all()
    rows = p.loc[p.date.ge(FIT_DATE)].merge(recorded, on=["date", "symbol"], how="inner")
    assert len(rows) == len(recorded), (len(rows), len(recorded))
    rows["refit"] = pipeline.predict(rows[FEATURES].astype("float32"))
    max_abs_diff = float((rows.refit - rows.score).abs().max())
    print(
        f"refit vs HW4 recorded scores on {len(rows):,} rows: max |diff| = {max_abs_diff:.3e}"
    )
    assert max_abs_diff < 1e-6

    # Save artifact
    ARTIFACT_DIR.mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")

    ridge = pipeline.named_steps["ridge"]
    metadata = {
        "model_version": MODEL_VERSION,
        "hw4_config": "ridge_core_ret: Ridge(alpha=10) on 7 same-date ranks, excess-return target",
        "fit_date": str(FIT_DATE.date()),
        "n_train": len(train),
        "target": "90-day holding return minus same-date screened-universe mean (decimal)",
        "target_clip_1pct_99pct": [float(c) for c in clip],
        "features_in_order": FEATURES,
        "coefficients": dict(zip(FEATURES, map(float, ridge.coef_), strict=True)),
        "intercept": float(ridge.intercept_),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "check_vs_hw4_recorded_scores": {"rows": len(rows), "max_abs_diff": max_abs_diff},
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    last = rows.loc[rows.date.eq(rows.date.max())].sort_values("symbol")
    complete = last.dropna(subset=FEATURES)
    picks = complete.loc[complete.symbol.isin(["AAPL", "JPM", "XOM"])]
    with_missing = last.loc[last[FEATURES].isna().any(axis=1)].head(1)
    reference = []
    for _, r in pd.concat([picks, with_missing]).iterrows():
        features = {f: None if pd.isna(r[f]) else float(r[f]) for f in FEATURES}
        reference.append(
            {
                "symbol": r.symbol,
                "signal_date": str(r.date.date()),
                "features": features,
                "hw4_recorded_score": float(r.score),
            }
        )
    (HERE / "tests" / "hw4_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (HERE / "example_request.json").write_text(
        json.dumps(reference[0]["features"], indent=2) + "\n"
    )
    print(f"saved {ARTIFACT_DIR / 'model.joblib'} and {len(reference)} reference rows")


if __name__ == "__main__":
    main()
