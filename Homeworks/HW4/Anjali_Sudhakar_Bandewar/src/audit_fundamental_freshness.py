import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

print("\nQUARTERLY RECORDS BY FILING LAG")
print("=" * 65)

print(con.execute("""
    SELECT
        COUNT(*) AS total_records,
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 180
        ) AS over_180_days,
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 365
        ) AS over_365_days,
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 548
        ) AS over_18_months,
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 730
        ) AS over_2_years
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
""").df().to_string(index=False))

print("\nRECORDS WITH FILING LAG ABOVE 18 MONTHS")
print("=" * 65)

print(con.execute("""
    SELECT
        symbol,
        fiscal_date,
        filing_date,
        DATE_DIFF('day', fiscal_date, filing_date) AS lag_days
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
      AND DATE_DIFF('day', fiscal_date, filing_date) > 548
    ORDER BY lag_days DESC
    LIMIT 15
""").df().to_string(index=False))

con.close()
