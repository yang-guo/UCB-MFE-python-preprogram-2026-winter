"""Fit the HW5 serving artifact from the HW4 data. Run from the repo root with the course env:

    .venv/bin/python Homeworks/HW5/Emil_Blaignan/train_model.py

Reads HW4's generated outputs/ (weekly.parquet, adv_weekly.parquet) and the course DuckDB file
(sector labels only). Writes model.joblib, example_request.json and reference.json next to this file.
The sample rows follow HW4's benchrel.load_core(): the 300 most traded stocks per week and the
90-day return relative to the sector median, clipped at development quantiles.
"""

import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fm_model import FamaMacBethRegressor, QuantileClipper, ZClip  # noqa: E402

ROOT = HERE.parents[2]
HW4 = ROOT / "Homeworks" / "HW4" / "Emil_Blaignan"
DB = ROOT / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
FEATS = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "beta_mkt",
    "beta_mom",
    "resid_mom",
    "idisc",
    "adv_log",
]  # same list and order as HW4 helpers.FEATS
FIT_DATE = pd.Timestamp("2015-12-31")  # same cut as HW4's development window
MATURE = pd.Timedelta(days=102)  # HW4: labels end up to 101 days after the signal date
MODEL_VERSION = "hw4-fm-linear-2015-12-31-v1"


def load_rows() -> pd.DataFrame:
    w = pd.read_parquet(HW4 / "outputs" / "weekly.parquet")
    w = w.merge(pd.read_parquet(HW4 / "outputs" / "adv_weekly.parquet"), on=["symbol", "date"])
    w = w[w.adv.notna()].copy()
    w["adv_rank"] = w.groupby("date").adv.rank(ascending=False, method="first")
    w = w[w.adv_rank <= 300].reset_index(drop=True)
    w["adv_log"] = np.log(w.adv)
    sec = duckdb.connect(str(DB), read_only=True).execute("select symbol, sector from universe")
    w = w.merge(sec.fetchdf(), on="symbol", how="left")
    w["sector"] = w.sector.fillna("Unknown")
    r = w.target.clip(-1, 3)
    rel = r - r.groupby([w.date, w.sector]).transform("median")
    dev = (w.date <= "2015-12-31") & (w.label_end <= "2016-03-31")
    lo, hi = rel[dev].quantile([0.001, 0.999])
    w["y"] = rel.clip(lo, hi)
    return w


def main() -> None:
    w = load_rows()
    # Only outcomes that were already finished on the fit date, as in HW4.
    train = w[(w.date <= FIT_DATE - MATURE) & w.y.notna()]
    n_before = len(train)
    train = train.dropna(
        subset=FEATS
    )  # the API rejects missing values, so train on complete rows
    print(
        f"training rows: {len(train):,} (dropped {n_before - len(train):,} with missing features)"
    )
    assert (train.label_end <= FIT_DATE).all()

    pipe = Pipeline(
        [
            ("clip", QuantileClipper(0.01, 0.99)),
            ("scale", StandardScaler()),
            ("zclip", ZClip(3.0)),
            ("fm", FamaMacBethRegressor(lag=13)),
        ]
    )
    pipe.fit(train[FEATS], train.y, fm__dates=train.date.to_numpy())
    fm = pipe.named_steps["fm"]
    table = pd.DataFrame(
        {"mean": fm.mean_slopes_, "t_stat": fm.t_stats_, "premium": fm.coef_}, index=FEATS
    )
    print(table.round(4).to_string())

    # Observed 0.1% / 99.9% range per feature, used to justify the API's allowed ranges.
    ranges = train[FEATS].quantile([0.0, 0.001, 0.999, 1.0]).T
    ranges.columns = ["min", "p0.1", "p99.9", "max"]
    print(ranges.to_string())

    # Reference example: a real HW4 row, scored by the pipeline BEFORE any API code touches it.
    row = train.sort_values(["date", "symbol"]).iloc[len(train) // 2]
    example = {f: float(row[f]) for f in FEATS}
    expected = float(pipe.predict(pd.DataFrame([example], columns=FEATS))[0])

    artifact = {
        "pipeline": pipe,
        "model_version": MODEL_VERSION,
        "features": FEATS,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, HERE / "model.joblib", compress=3)
    (HERE / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    reference = {
        "source": f"HW4 weekly row {row.symbol} signal date {row.date.date()}, scored by the fitted "
        "pipeline directly in train_model.py (no API).",
        "input": example,
        "expected_prediction": expected,
        "model_version": MODEL_VERSION,
    }
    (HERE / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    table.round(4).to_csv(HERE / "premia.csv")
    ranges.round(4).to_csv(HERE / "feature_ranges.csv")
    print("expected prediction:", expected)


if __name__ == "__main__":
    main()
