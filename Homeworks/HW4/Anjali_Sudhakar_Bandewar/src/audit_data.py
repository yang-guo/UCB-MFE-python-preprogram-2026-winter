import duckdb
from pathlib import Path

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

con = duckdb.connect(str(DB), read_only=True)

queries = {
    "PRICE COVERAGE": """
        SELECT
            MIN(date), MAX(date),
            COUNT(*), COUNT(DISTINCT symbol),
            COUNT(*) FILTER (WHERE adj_close IS NULL),
            COUNT(*) FILTER (WHERE adj_close <= 0)
        FROM prices
    """,

    "DUPLICATE PRICE RECORDS": """
        SELECT COUNT(*)
        FROM (
            SELECT symbol, date
            FROM prices
            GROUP BY symbol, date
            HAVING COUNT(*) > 1
        )
    """,

    "BENCHMARK COVERAGE": """
        SELECT symbol, MIN(date), MAX(date), COUNT(*),
               COUNT(*) FILTER (WHERE adj_close IS NULL)
        FROM benchmark
        GROUP BY symbol
    """,

    "UNIVERSE MEMBERSHIP": """
        SELECT
            MIN(quarter_start), MAX(quarter_start),
            COUNT(*), COUNT(DISTINCT symbol),
            COUNT(DISTINCT quarter_start)
        FROM universe_membership
    """,

    "MEMBERSHIP COUNTS BY QUARTER": """
        SELECT quarter_start, COUNT(*) AS members
        FROM universe_membership
        GROUP BY quarter_start
        ORDER BY quarter_start
        LIMIT 5
    """,

    "FUNDAMENTALS COVERAGE": """
        SELECT
            MIN(fiscal_date), MAX(fiscal_date),
            MIN(filing_date), MAX(filing_date),
            COUNT(*),
            COUNT(*) FILTER (WHERE filing_date IS NULL),
            COUNT(*) FILTER (
                WHERE filing_date < fiscal_date
            )
        FROM fundamentals
    """,

    "DUPLICATE FUNDAMENTAL RECORDS": """
        SELECT COUNT(*)
        FROM (
            SELECT symbol, fiscal_date, period
            FROM fundamentals
            GROUP BY symbol, fiscal_date, period
            HAVING COUNT(*) > 1
        )
    """,

    "DATABASE METADATA": """
        SELECT * FROM dataset_metadata ORDER BY key
    """
}

for title, query in queries.items():
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    result = con.execute(query)
    print(result.df().to_string(index=False))

con.close()
