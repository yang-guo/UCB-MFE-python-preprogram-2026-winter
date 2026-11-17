import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

print("\nTOP 20 LONGEST QUARTERLY FILING LAGS")
print("=" * 80)

print(con.execute("""
    SELECT
        symbol,
        fiscal_date,
        filing_date,
        period,
        DATE_DIFF('day', fiscal_date, filing_date) AS lag_days,
        revenue,
        net_income
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
    ORDER BY lag_days DESC
    LIMIT 20
""").df().to_string(index=False))

print("\nFILING LAG DISTRIBUTION")
print("=" * 80)

print(con.execute("""
    SELECT
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 180
        ) AS over_180_days,
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 365
        ) AS over_365_days,
        COUNT(*) FILTER (
            WHERE DATE_DIFF('day', fiscal_date, filing_date) > 730
        ) AS over_2_years
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
""").df().to_string(index=False))

con.close()
