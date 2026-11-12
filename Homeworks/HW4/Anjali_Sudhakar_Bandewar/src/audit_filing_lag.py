import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

print("\nQUARTERLY FUNDAMENTAL FILING LAG")
print("=" * 65)

print(con.execute("""
    SELECT
        COUNT(*) AS records,
        ROUND(AVG(DATE_DIFF('day', fiscal_date, filing_date)), 1)
            AS average_lag_days,
        MEDIAN(DATE_DIFF('day', fiscal_date, filing_date))
            AS median_lag_days,
        QUANTILE_CONT(
            DATE_DIFF('day', fiscal_date, filing_date), 0.90
        ) AS p90_lag_days,
        MIN(DATE_DIFF('day', fiscal_date, filing_date))
            AS minimum_lag_days,
        MAX(DATE_DIFF('day', fiscal_date, filing_date))
            AS maximum_lag_days
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
""").df().to_string(index=False))

print("\nFILING LAG BY FILING YEAR")
print("=" * 65)

print(con.execute("""
    SELECT
        YEAR(filing_date) AS filing_year,
        COUNT(*) AS records,
        ROUND(
            MEDIAN(DATE_DIFF('day', fiscal_date, filing_date)), 1
        ) AS median_lag_days
    FROM fundamentals
    WHERE period = 'quarter'
      AND filing_date >= fiscal_date
    GROUP BY filing_year
    ORDER BY filing_year
""").df().to_string(index=False))

con.close()
