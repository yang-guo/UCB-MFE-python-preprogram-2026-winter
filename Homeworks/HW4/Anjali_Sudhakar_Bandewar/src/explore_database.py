import duckdb
import sys
from pathlib import Path


def explore_database(db_path):
    db_path = Path(db_path).expanduser().resolve()

    if not db_path.exists():
        raise FileNotFoundError(db_path)

    print(f"\n{'=' * 70}")
    print(f"DATABASE: {db_path.name}")
    print(f"{'=' * 70}")

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        tables = con.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
        """).fetchall()

        print(f"\nNumber of tables: {len(tables)}")

        for schema, table in tables:
            print(f"\n{'-' * 70}")
            print(f"TABLE: {schema}.{table}")

            quoted_table = (
                '"' + schema.replace('"', '""') + '"'
                + '.'
                + '"' + table.replace('"', '""') + '"'
            )

            columns = con.execute(
                f"DESCRIBE SELECT * FROM {quoted_table}"
            ).fetchall()

            print("\nColumns:")
            for column in columns:
                print(f"  {column[0]:30} {column[1]}")

            count = con.execute(
                f"SELECT COUNT(*) FROM {quoted_table}"
            ).fetchone()[0]

            print(f"\nRow count: {count:,}")

            print("\nSample rows:")
            print(
                con.execute(
                    f"SELECT * FROM {quoted_table} LIMIT 3"
                ).df().to_string(index=False)
            )

    finally:
        con.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explore_database.py <database_path>")
        sys.exit(1)

    explore_database(sys.argv[1])
