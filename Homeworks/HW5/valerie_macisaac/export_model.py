"""One-time export script to fit the HW4 15-feature model and save model_artifact.joblib."""

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from model_utils import (
    FEATURES_15,
    CrossSectionalRankTransformer,
    HW4StockReturnEnsemble,
)

MODEL_VERSION = "hw4-15feat-ridge-gbdt-v1.0.0"
DB_PATH = Path("equities_top2000.duckdb")
ARTIFACT_PATH = Path("model_artifact.joblib")
EXAMPLE_JSON_PATH = Path("example_request.json")


def main() -> None:
    assert DB_PATH.exists(), f"Database not found at {DB_PATH.resolve()}"
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Extract weekly cross-sections across 2010-2016 (matching Cell 16 & Cell 34)
    sched = con.execute("""
        SELECT date AS signal_date
        FROM calendar
        WHERE date >= '2010-01-01' AND date <= '2016-06-30'
        ORDER BY date
    """).df()
    active_dates = sched.iloc[::5][["signal_date"]].copy()
    con.register("active_signal_dates_tbl", active_dates)

    query = """
    WITH trading_days AS (
        SELECT
            date AS signal_date,
            LEAD(date, 1) OVER (ORDER BY date) AS entry_date,
            LAG(date, 62) OVER (ORDER BY date) AS date_minus_62d
        FROM calendar
        WHERE date >= '2008-06-01'
    ),
    schedule AS (
        SELECT
            t.signal_date,
            t.entry_date,
            t.date_minus_62d,
            MIN(c_exit.date) AS exit_date
        FROM trading_days t
        JOIN active_signal_dates_tbl a ON a.signal_date = t.signal_date
        LEFT JOIN calendar c_exit
            ON c_exit.date >= t.entry_date + INTERVAL 90 DAY
        WHERE t.entry_date IS NOT NULL
        GROUP BY t.signal_date, t.entry_date, t.date_minus_62d
    ),
    daily_returns AS (
        SELECT
            symbol,
            date,
            close,
            adj_close,
            volume,
            LAG(date, 62) OVER w AS stock_date_minus_62d,
            adj_close / NULLIF(LAG(adj_close, 1) OVER w, 0) - 1.0 AS ret_1d,
            adj_close / NULLIF(LAG(adj_close, 21) OVER w, 0) - 1.0 AS ret_21d,
            adj_close / NULLIF(LAG(adj_close, 252) OVER w, 0) - 1.0 AS ret_252d,
            (high - low) / NULLIF(close, 0) AS hl_spread
        FROM prices
        WHERE date >= '2008-06-01'
        WINDOW w AS (PARTITION BY symbol ORDER BY date)
    ),
    price_features AS (
        SELECT
            dr.symbol,
            dr.date AS signal_date,
            dr.close,
            dr.stock_date_minus_62d,
            dr.ret_252d,
            (1.0 + dr.ret_252d) / NULLIF(1.0 + dr.ret_21d, 0) - 1.0 AS mom_12m_skip1m,
            STDDEV_SAMP(dr.ret_1d) OVER w63 AS vol_63d,
            AVG(dr.hl_spread) OVER w21 AS avg_hl_spread_21d,
            AVG(dr.close * dr.volume) OVER w21 AS dollar_vol_21d,
            MIN(dr.volume) OVER w21 AS min_vol_21d
        FROM daily_returns dr
        WINDOW
            w21 AS (PARTITION BY dr.symbol ORDER BY dr.date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW),
            w63 AS (PARTITION BY dr.symbol ORDER BY dr.date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW)
    ),
    dedup_q AS (
        SELECT
            symbol,
            fiscal_date,
            filing_date,
            filing_date + INTERVAL 1 DAY AS available_date,
            revenue,
            gross_profit,
            operating_income,
            net_income,
            ebitda,
            operating_cash_flow,
            free_cash_flow,
            total_assets,
            total_equity,
            ROW_NUMBER() OVER (PARTITION BY symbol, fiscal_date ORDER BY filing_date ASC) AS rn
        FROM fundamentals
        WHERE filing_date IS NOT NULL AND period = 'quarter'
    ),
    fund_ttm AS (
        SELECT
            symbol,
            available_date,
            revenue,
            gross_profit,
            operating_income,
            net_income,
            ebitda,
            operating_cash_flow,
            free_cash_flow,
            total_assets,
            total_equity,
            SUM(revenue) OVER w_curr4 AS rev_ttm,
            COUNT(revenue) OVER w_curr4 AS n_curr4
        FROM dedup_q
        WHERE rn = 1
        WINDOW w_curr4 AS (PARTITION BY symbol ORDER BY fiscal_date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
    )
    SELECT
        s.signal_date,
        s.entry_date,
        s.exit_date,
        um.symbol,
        um.universe_rank,
        pf.close AS signal_close,
        (pf.stock_date_minus_62d = s.date_minus_62d) AS has_full_63d_trading,
        pf.min_vol_21d,
        pf.dollar_vol_21d,
        um.market_cap_proxy / NULLIF(f.total_assets, 0) AS mcap_to_assets_sanity,
        f.ebitda / NULLIF(f.total_assets, 0) AS ebitda_to_assets,
        f.net_income / NULLIF(f.total_assets, 0) AS roa,
        f.free_cash_flow / NULLIF(f.total_assets, 0) AS fcf_to_assets,
        f.operating_cash_flow / NULLIF(f.total_assets, 0) AS ocf_to_assets,
        f.operating_income / NULLIF(f.revenue, 0) AS op_margin,
        f.gross_profit / NULLIF(f.revenue, 0) AS gross_margin,
        pf.ret_252d,
        pf.mom_12m_skip1m,
        f.net_income / NULLIF(um.market_cap_proxy, 0) AS earnings_yield,
        f.free_cash_flow / NULLIF(um.market_cap_proxy, 0) AS fcf_yield,
        f.total_equity / NULLIF(um.market_cap_proxy, 0) AS book_to_market,
        -CAST(um.universe_rank AS DOUBLE) AS neg_universe_rank,
        pf.vol_63d,
        pf.avg_hl_spread_21d,
        CASE
            WHEN f.n_curr4 = 4 AND f.rev_ttm > 0
            THEN um.market_cap_proxy / f.rev_ttm
            ELSE NULL
        END AS mcap_to_rev_ttm,
        p_ent.adj_close AS entry_adj_close,
        (p_ext.adj_close / NULLIF(p_ent.adj_close, 0)) - 1.0 AS target_90d_ret
    FROM schedule s
    JOIN universe_membership um
        ON s.signal_date BETWEEN um.effective_start AND um.quarter_end
    LEFT JOIN price_features pf
        ON pf.symbol = um.symbol AND pf.signal_date = s.signal_date
    ASOF LEFT JOIN fund_ttm f
        ON f.symbol = um.symbol AND f.available_date <= s.signal_date
    LEFT JOIN prices p_ent
        ON p_ent.symbol = um.symbol AND p_ent.date = s.entry_date
    LEFT JOIN prices p_ext
        ON p_ext.symbol = um.symbol AND p_ext.date = s.exit_date
    ORDER BY s.signal_date, um.symbol
    """

    df = con.execute(query).df()
    con.close()

    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    screen_mask = (
        (df["has_full_63d_trading"] == True)  # noqa: E712
        & (df["min_vol_21d"] > 0)
        & (df["dollar_vol_21d"] >= 5_000_000)
        & (df["signal_close"].between(5.0, 5000.0))
        & (df["mcap_to_assets_sanity"].between(0.01, 500.0))
        & (df["entry_adj_close"].notna())
    )
    screened = df[screen_mask].copy()
    screened["target_rank"] = (
        screened.groupby("signal_date")["target_90d_ret"].rank(pct=True, na_option="keep") - 0.5
    )

    # Fit on Development period (completed 90d trades prior to 2016-03-31)
    train_df = screened[
        (screened["exit_date"] <= pd.Timestamp("2016-03-31"))
        & (screened["target_rank"].notna())
    ].copy()

    pipeline = Pipeline(
        steps=[
            ("ranker", CrossSectionalRankTransformer(feature_names=FEATURES_15)),
            (
                "ensemble",
                HW4StockReturnEnsemble(ridge_weight=0.6, gbdt_weight=0.4, random_state=42),
            ),
        ]
    )
    pipeline.fit(train_df[FEATURES_15], train_df["target_rank"].to_numpy(dtype=float))

    # Pick a clean, complete out-of-sample test row (first test signal date 2016-04-01+)
    test_df = screened[
        (screened["signal_date"] >= pd.Timestamp("2016-04-01"))
        & (screened[FEATURES_15].notna().all(axis=1))
    ].copy()
    ref_row = test_df.iloc[0]
    ref_input = {col: round(float(ref_row[col]), 8) for col in FEATURES_15}
    ref_df = pd.DataFrame([ref_input], columns=FEATURES_15)
    expected_pred = float(pipeline.predict(ref_df)[0])

    artifact = {
        "pipeline": pipeline,
        "feature_names": FEATURES_15,
        "model_version": MODEL_VERSION,
        "target_description": "Centered cross-sectional 90-calendar-day forward return rank in [-0.5, 0.5]",
        "reference_metadata": {
            "symbol": str(ref_row["symbol"]),
            "signal_date": str(ref_row["signal_date"].date()),
            "actual_90d_ret": round(float(ref_row["target_90d_ret"]), 8),
        },
        "reference_input": ref_input,
        "reference_prediction": expected_pred,
    }
    joblib.dump(artifact, ARTIFACT_PATH, compress=3)
    EXAMPLE_JSON_PATH.write_text(json.dumps(ref_input, indent=2) + "\n")

    print(
        f"Trained on {len(train_df):,} rows | Saved {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size / 1024:.1f} KB)"
    )
    print(f"Reference symbol: {ref_row['symbol']} on {ref_row['signal_date'].date()}")
    print("Reference input:", json.dumps(ref_input))
    print(f"Reference prediction: {expected_pred:.12f}")


if __name__ == "__main__":
    main()
