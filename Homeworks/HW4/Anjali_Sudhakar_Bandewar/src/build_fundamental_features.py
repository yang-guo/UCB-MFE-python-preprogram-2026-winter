import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

PRICE_PANEL = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/price_feature_panel.parquet"
)

OUTPUT = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/combined_feature_panel.parquet"
)

con = duckdb.connect(str(DB), read_only=True)

query = f"""
WITH clean_fundamentals AS (
    SELECT
        symbol,
        fiscal_date,
        filing_date,
        revenue,
        net_income,
        ebitda,
        total_debt,
        total_equity,
        total_assets,
        free_cash_flow
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
),
-- Range join instead of ASOF: some later filings carry an older fiscal_date,
-- so the latest *filing* is not always the latest *fiscal period*. Join every
-- eligible record, then keep the best one per stock-date. The freshness rule
-- lives in the join so stale-only observations survive with NULL fundamentals.
joined AS (
    SELECT
        p.*,
        f.fiscal_date,
        f.filing_date,
        f.revenue,
        f.net_income,
        f.ebitda,
        f.total_debt,
        f.total_equity,
        f.total_assets,
        f.free_cash_flow
    FROM read_parquet('{PRICE_PANEL}') p
    LEFT JOIN clean_fundamentals f
        ON p.symbol = f.symbol
       AND f.filing_date <= p.date
       AND f.fiscal_date <= p.date
       AND DATE_DIFF('day', f.fiscal_date, p.date) <= 548
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY p.symbol, p.date
        ORDER BY f.fiscal_date DESC NULLS LAST, f.filing_date DESC NULLS LAST
    ) = 1
)
SELECT
    *,
    net_income / NULLIF(revenue, 0) AS profit_margin,
    ebitda / NULLIF(revenue, 0) AS ebitda_margin,
    total_debt / NULLIF(total_equity, 0) AS debt_to_equity,
    revenue / NULLIF(total_assets, 0) AS asset_turnover,
    free_cash_flow / NULLIF(revenue, 0) AS fcf_margin
FROM joined
"""

print("Building combined price and fundamental feature panel...")

con.execute(f"""
    COPY ({query})
    TO '{OUTPUT}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nPANEL SUMMARY")
print("=" * 70)

print(con.execute(f"""
    SELECT
        COUNT(*) AS observations,
        COUNT(DISTINCT symbol) AS stocks,
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        COUNT(*) FILTER (WHERE filing_date IS NULL)
            AS missing_fundamental_records,
        COUNT(*) FILTER (WHERE profit_margin IS NULL)
            AS missing_profit_margin,
        COUNT(*) FILTER (WHERE ebitda_margin IS NULL)
            AS missing_ebitda_margin,
        COUNT(*) FILTER (WHERE debt_to_equity IS NULL)
            AS missing_debt_to_equity,
        COUNT(*) FILTER (WHERE asset_turnover IS NULL)
            AS missing_asset_turnover,
        COUNT(*) FILTER (WHERE fcf_margin IS NULL)
            AS missing_fcf_margin
    FROM read_parquet('{OUTPUT}')
""").df().to_string(index=False))

print("\nSAMPLE OBSERVATIONS")
print("=" * 70)

print(con.execute(f"""
    SELECT
        symbol,
        date,
        fiscal_date,
        filing_date,
        profit_margin,
        ebitda_margin,
        debt_to_equity,
        asset_turnover,
        fcf_margin
    FROM read_parquet('{OUTPUT}')
    ORDER BY date, universe_rank
    LIMIT 10
""").df().to_string(index=False))

con.close()

print(f"\nSaved combined panel to: {OUTPUT}")
