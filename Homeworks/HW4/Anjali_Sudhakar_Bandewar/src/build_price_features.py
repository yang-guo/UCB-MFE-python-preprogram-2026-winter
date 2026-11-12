import duckdb
from pathlib import Path

from exclusions import excluded_symbols_sql

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

OUTPUT = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/price_feature_panel.parquet"
)

con = duckdb.connect(str(DB), read_only=True)

query = f"""
WITH historical_prices AS (
    SELECT
        symbol,
        date,
        adj_close,
        LAG(adj_close, 21) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS price_1m,
        LAG(adj_close, 63) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS price_3m,
        LAG(adj_close, 126) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS price_6m,
        LAG(adj_close, 252) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS price_12m
    FROM prices
    WHERE date BETWEEN DATE '2008-01-01' AND DATE '2016-03-31'
      AND adj_close > 0
),
price_features AS (
    SELECT
        symbol,
        date,
        adj_close,
        adj_close / NULLIF(price_1m, 0) - 1 AS momentum_1m,
        adj_close / NULLIF(price_3m, 0) - 1 AS momentum_3m,
        adj_close / NULLIF(price_6m, 0) - 1 AS momentum_6m,
        adj_close / NULLIF(price_12m, 0) - 1 AS momentum_12m
    FROM historical_prices
)
SELECT
    p.*,
    u.universe_rank
FROM price_features p
INNER JOIN universe_membership u
    ON p.symbol = u.symbol
   AND p.date >= u.effective_start
   AND p.date <= u.quarter_end
WHERE p.date BETWEEN DATE '2010-01-01' AND DATE '2016-03-31'
  -- Drop contaminated tickers; other stocks keep their original
  -- universe_rank (no re-ranking).
  AND p.symbol NOT IN {excluded_symbols_sql()}
"""

print("Building universe-filtered price feature panel...")

con.execute(f"""
    COPY ({query})
    TO '{OUTPUT}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS observations,
        COUNT(DISTINCT symbol) AS stocks,
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        COUNT(*) FILTER (WHERE momentum_12m IS NULL)
            AS missing_12m_momentum
    FROM read_parquet('{OUTPUT}')
""").df()

print("\nPANEL SUMMARY")
print("=" * 65)
print(summary.to_string(index=False))

print("\nSAMPLE OBSERVATIONS")
print("=" * 65)

print(con.execute(f"""
    SELECT *
    FROM read_parquet('{OUTPUT}')
    ORDER BY date, universe_rank
    LIMIT 10
""").df().to_string(index=False))

con.close()

print(f"\nSaved panel to: {OUTPUT}")
