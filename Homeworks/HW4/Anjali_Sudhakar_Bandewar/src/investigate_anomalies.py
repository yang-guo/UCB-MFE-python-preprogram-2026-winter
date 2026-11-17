import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

print("\n1. NON-POSITIVE ADJUSTED PRICES")
print("=" * 65)

print(con.execute("""
    SELECT symbol, date, close, adj_close, volume
    FROM prices
    WHERE adj_close <= 0
    ORDER BY symbol, date
    LIMIT 50
""").df().to_string(index=False))

print("\nAffected stocks:")
print(con.execute("""
    SELECT COUNT(DISTINCT symbol)
    FROM prices
    WHERE adj_close <= 0
""").fetchone()[0])

print("\n2. FUNDAMENTALS WITH FILING DATE BEFORE FISCAL DATE")
print("=" * 65)

print(con.execute("""
    SELECT symbol, fiscal_date, filing_date, period,
           revenue, net_income
    FROM fundamentals
    WHERE filing_date < fiscal_date
    ORDER BY symbol, fiscal_date
""").df().to_string(index=False))

print("\n3. PRICE RECORDS WITH INVALID RAW PRICES")
print("=" * 65)

print(con.execute("""
    SELECT
        COUNT(*) FILTER (WHERE close IS NULL) AS missing_close,
        COUNT(*) FILTER (WHERE close <= 0) AS nonpositive_close,
        COUNT(*) FILTER (WHERE volume < 0) AS negative_volume
    FROM prices
""").df().to_string(index=False))

con.close()
