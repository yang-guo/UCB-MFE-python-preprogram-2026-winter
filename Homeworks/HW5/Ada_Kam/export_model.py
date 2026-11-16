"""Refit and save HW4 final model (03_backtest.ipynb, cell 2) for HW5"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from course_data import make_pipeline

ROOT = Path(__file__).resolve().parent
PANEL = ROOT.parents[1] / "HW4/Ada_Kam/outputs/panel_full.parquet"
MODEL_VERSION = "hw4-hgb-rank-v1"
FEATURES = [
    "rsi_21",
    "dma_20",
    "dma_50",
    "dma_100",
    "momentum_252_21",
    "volatility_63",
    "roe",
    "debt_to_equity",
    "book_to_px",
    "earnings_yield",
    "fcf_yield",
    "ebitda_ev",
]


def main():
    panel = pd.read_parquet(PANEL)

    dev = panel.loc[
        panel.eligible
        & panel.date.between("2010-01-01", "2015-12-31")
        & panel.target.notna()
        & panel.label_end.le("2016-03-31")
    ].copy()
    dev["target_rank"] = dev.groupby("date").target.rank(pct=True) - 0.5

    model = make_pipeline(
        HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=200,
            l2_regularization=10,
            early_stopping=False,
            random_state=42,
        ),
        numeric_features=FEATURES,
        include_sector=False,
    ).fit(dev[FEATURES], dev.target_rank)

    eligible = panel.loc[panel.eligible]
    ref_date = eligible.date.max()
    test = eligible.loc[eligible.date.eq(ref_date)].dropna(subset=FEATURES)
    ref = test[FEATURES].iloc[[0]]
    ref_pred = float(model.predict(ref)[0])

    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": model, "features": FEATURES, "model_version": MODEL_VERSION},
        ROOT / "artifacts/model.joblib",
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "source": "HW4 03_backtest.ipynb cell 2; HistGradientBoostingRegressor",
        "features": FEATURES,
        "train_window": "2010-01-01 to 2015-12-31, label_end <= 2016-03-31",
        "target": "same-date percentile rank of 90-day return minus 0.5 (range ~[-0.5, 0.5])",
        "sklearn_version": sklearn.__version__,
        "reference_input": ref.iloc[0].to_dict(),
        "reference_symbol_date": str(ref_date.date()),
        "reference_prediction": ref_pred,
    }

    (ROOT / "artifacts/model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (ROOT / "example_request.json").write_text(
        json.dumps(ref.iloc[0].to_dict(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
