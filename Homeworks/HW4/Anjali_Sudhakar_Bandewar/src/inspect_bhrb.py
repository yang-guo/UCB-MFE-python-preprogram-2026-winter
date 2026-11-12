import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

print("\nBHRB: OVERALL PRICE COVERAGE")
print("=" * 65)
print(con.execute("""
    SELECT
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        COUNT(*) AS total_rows,
        COUNT(*) FILTER (WHERE adj_close <= 0) AS invalid_adj_close,
        COUNT(*) FILTER (WHERE adj_close > 0) AS valid_adj_close
    FROM prices
    WHERE symbol = 'BHRB'
""").df().to_string(index=False))

print("\nBHRB: ZERO ADJUSTED-PRICE OBSERVATIONS BY YEAR")
print("=" * 65)
print(con.execute("""
    SELECT
        YEAR(date) AS year,
        COUNT(*) AS zero_price_rows,
        MIN(date) AS first_date,
        MAX(date) AS last_date
    FROM prices
    WHERE symbol = 'BHRB'
      AND adj_close <= 0
    GROUP BY YEAR(date)
    ORDER BY year
""").df().to_string(index=False))

print("\nBHRB: FIRST 10 VALID ADJUSTED-PRICE OBSERVATIONS")
print("=" * 65)
print(con.execute("""
    SELECT date, close, adj_close, volume
    FROM prices
    WHERE symbol = 'BHRB'
      AND adj_close > 0
    ORDER BY date
    LIMIT 10
""").df().to_string(index=False))

print("\nBHRB: LAST 10 VALID ADJUSTED-PRICE OBSERVATIONS")
print("=" * 65)
print(con.execute("""
    SELECT date, close, adj_close, volume
    FROM prices
    WHERE symbol = 'BHRB'
      AND adj_close > 0
    ORDER BY date DESC
    LIMIT 10
""").df().to_string(index=False))

con.close()
