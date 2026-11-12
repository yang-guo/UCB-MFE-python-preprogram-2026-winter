import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

print("\nFUNDAMENTALS BY PERIOD")
print("=" * 65)

print(con.execute("""
    SELECT
        period,
        COUNT(*) AS records,
        COUNT(DISTINCT symbol) AS stocks,
        MIN(filing_date) AS first_filing,
        MAX(filing_date) AS last_filing
    FROM fundamentals
    GROUP BY period
    ORDER BY period
""").df().to_string(index=False))

print("\nFUNDAMENTAL FIELD AVAILABILITY")
print("=" * 65)

print(con.execute("""
    SELECT
        COUNT(*) AS total_records,
        COUNT(revenue) AS revenue_available,
        COUNT(net_income) AS net_income_available,
        COUNT(ebitda) AS ebitda_available,
        COUNT(total_assets) AS assets_available,
        COUNT(total_debt) AS debt_available,
        COUNT(total_equity) AS equity_available,
        COUNT(operating_cash_flow) AS operating_cash_flow_available,
        COUNT(free_cash_flow) AS free_cash_flow_available,
        COUNT(eps) AS eps_available,
        COUNT(market_cap) AS market_cap_available
    FROM fundamentals
""").df().to_string(index=False))

print("\nFUNDAMENTAL RECORDS BY FILING YEAR")
print("=" * 65)

print(con.execute("""
    SELECT
        YEAR(filing_date) AS filing_year,
        COUNT(*) AS records,
        COUNT(DISTINCT symbol) AS stocks
    FROM fundamentals
    GROUP BY filing_year
    ORDER BY filing_year
""").df().to_string(index=False))

con.close()
