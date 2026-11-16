"""Export the fitted HW4 preprocessing + Ridge pipeline and a reference test sample."""

from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["mom_20d", "mom_60d", "vol_20d"]
MODEL_VERSION = "hw4-ridge-alpha10-v1"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    db_path = repo_root / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
    con = duckdb.connect(str(db_path), read_only=True)

    query = """
    WITH clean_prices AS (
        SELECT
            date,
            symbol AS ticker,
            adj_close AS close,
            LEAD(date) OVER (PARTITION BY symbol ORDER BY date) AS next_date,
            LEAD(adj_close) OVER (PARTITION BY symbol ORDER BY date) AS entry_price,
            LEAD(date, 64) OVER (PARTITION BY symbol ORDER BY date) AS exit_date,
            LEAD(adj_close, 64) OVER (PARTITION BY symbol ORDER BY date) AS exit_price
        FROM prices
        WHERE date >= '2009-09-01'
          AND adj_close > 0
          AND isfinite(adj_close)
    )
    SELECT
        p.date,
        p.ticker,
        p.close,
        p.next_date,
        p.entry_price,
        p.exit_date,
        p.exit_price,
        CASE WHEN m.symbol IS NOT NULL THEN 1 ELSE 0 END AS is_eligible
    FROM clean_prices p
    LEFT JOIN universe_membership m
        ON p.ticker = m.symbol
        AND p.next_date >= m.quarter_start
        AND p.next_date < m.quarter_start + INTERVAL '3 months'
    ORDER BY p.ticker, p.date
    """
    df = con.execute(query).df()
    con.close()

    df["date"] = pd.to_datetime(df["date"])
    df["next_date"] = pd.to_datetime(df["next_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    ret_1d = df.groupby("ticker")["close"].pct_change()
    df["mom_20d"] = df["close"] / df.groupby("ticker")["close"].shift(20) - 1
    df["mom_60d"] = df["close"] / df.groupby("ticker")["close"].shift(60) - 1
    df["vol_20d"] = (
        ret_1d.groupby(df["ticker"]).rolling(20).std().reset_index(level=0, drop=True)
    )
    df["target_90d"] = df["exit_price"] / df["entry_price"] - 1

    train_mask = (
        (df["is_eligible"] == 1)
        & (df["next_date"] >= "2010-01-01")
        & (df["exit_date"] <= "2015-12-31")
    )
    train_df = df[train_mask].dropna(subset=FEATURE_NAMES + ["target_90d"]).copy()

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["target_90d"].clip(-1.0, 5.0)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    test_mask = (
        (df["is_eligible"] == 1)
        & (df["next_date"] >= "2016-04-01")
        & (df["next_date"] <= "2026-09-30")
    )
    test_df = df[test_mask].dropna(subset=FEATURE_NAMES + ["entry_price"]).copy()

    # Pick AAPL's first eligible test signal as a concrete reference example
    ref_row = test_df[test_df["ticker"] == "AAPL"].iloc[0]
    ref_features = {col: round(float(ref_row[col]), 8) for col in FEATURE_NAMES}
    ref_df = pd.DataFrame([ref_features], columns=FEATURE_NAMES)
    ref_pred = float(pipeline.predict(ref_df)[0])

    artifact = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "model_version": MODEL_VERSION,
        "reference_metadata": {
            "ticker": str(ref_row["ticker"]),
            "signal_date": str(ref_row["date"].date()),
            "entry_date": str(ref_row["next_date"].date()),
        },
        "reference_input": ref_features,
        "reference_prediction": ref_pred,
    }

    out_path = Path(__file__).resolve().parent / "model_artifact.joblib"
    joblib.dump(artifact, out_path)

    print(f"Saved artifact to: {out_path} ({out_path.stat().st_size} bytes)")
    print("Reference metadata:", artifact["reference_metadata"])
    print("Reference input:", artifact["reference_input"])
    print(f"Reference prediction: {ref_pred:.12f}")


if __name__ == "__main__":
    main()
