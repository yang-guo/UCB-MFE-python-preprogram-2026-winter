from pathlib import Path
import duckdb

DB_PATH = (
    Path(__file__).resolve().parents[3]
    / "Lectures"
    / "modeling"
    / "data"
    / "equities_top2000.duckdb"
)

db = duckdb.connect(str(DB_PATH), read_only=True)

print("DATABASE SUMMARY")

for table in ["prices", "benchmark", "universe_membership"]:
    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {count:,} rows")

print("\nSTOCK PRICE DATE RANGE")
print(
    db.execute(
        "SELECT MIN(date), MAX(date), COUNT(DISTINCT symbol) FROM prices"
    ).fetchone()
)

print("\nBENCHMARK SYMBOLS")
print(
    db.execute(
        "SELECT DISTINCT symbol FROM benchmark"
    ).fetchall()
)

print("\nUNIVERSE DATE RANGE")
print(
    db.execute(
        """
        SELECT MIN(quarter_start), MAX(quarter_start),
               COUNT(DISTINCT symbol)
        FROM universe_membership
        """
    ).fetchone()
)
print("\nPRICE DATA QUALITY")

print(
    "Missing adjusted prices:",
    db.execute(
        "SELECT COUNT(*) FROM prices WHERE adj_close IS NULL"
    ).fetchone()[0]
)

print(
    "Nonpositive adjusted prices:",
    db.execute(
        "SELECT COUNT(*) FROM prices WHERE adj_close <= 0"
    ).fetchone()[0]
)

print(
    "Duplicate stock/date combinations:",
    db.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT symbol, date
            FROM prices
            GROUP BY symbol, date
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
)

print("\nAAM PRICE INVESTIGATION")

rows = db.execute(
    """
    SELECT
        date,
        close,
        adj_close,
        source
    FROM prices
    WHERE symbol = ?
      AND date BETWEEN ? AND ?
    ORDER BY date
    """,
    ["AAM", "2012-01-20", "2012-04-30"]
).fetchall()

for row in rows:
    print(row)

print("\nAAM EXTREME RETURN CHECK")

result = db.execute(
    """
    SELECT
        entry.adj_close AS entry_price,
        exit.adj_close AS exit_price,
        exit.adj_close / NULLIF(entry.adj_close, 0) - 1 AS calculated_return
    FROM prices entry
    JOIN prices exit
      ON entry.symbol = exit.symbol
    WHERE entry.symbol = ?
      AND entry.date = ?
      AND exit.date = ?
    """,
    ["AAM", "2012-01-25", "2012-04-24"]
).fetchone()

print(result)

db.close()