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

def load_development_data():
    return db.execute(query).df()
    
query = """
WITH daily_prices AS (
    SELECT
        symbol,
        date,
        adj_close,
        LAG(adj_close) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS previous_close
    FROM prices
),
price_features AS (
    SELECT
        symbol,
        date,
        adj_close,

        LAG(adj_close, 20) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS price_20d_ago,

        LAG(adj_close, 60) OVER (
            PARTITION BY symbol ORDER BY date
        ) AS price_60d_ago,

        STDDEV_SAMP(
            CASE
                WHEN adj_close > 0 AND previous_close > 0
                THEN adj_close / previous_close - 1
                ELSE NULL
            END
        ) OVER (
            PARTITION BY symbol
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS volatility_20d, 
        COUNT(
    CASE
        WHEN adj_close > 0 AND previous_close > 0
        THEN 1
        ELSE NULL
    END
) OVER (
    PARTITION BY symbol
    ORDER BY date
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
) AS valid_return_count
    FROM daily_prices
),
entry_dates AS (
    SELECT
        date AS feature_date,
        LEAD(date) OVER (ORDER BY date) AS entry_date
    FROM (
        SELECT DISTINCT date
        FROM benchmark
    )
),
candidates AS (
    SELECT
        f.symbol,
        f.date AS feature_date,
        f.adj_close,
        f.adj_close / f.price_20d_ago - 1 AS return_20d,
        f.adj_close / f.price_60d_ago - 1 AS return_60d,
        f.volatility_20d,
        e.entry_date
    FROM price_features f
    JOIN entry_dates e
        ON f.date = e.feature_date
    WHERE f.date BETWEEN '2010-01-01' AND '2015-12-31'
      AND f.adj_close > 0
      AND f.valid_return_count = 20
      AND f.price_20d_ago > 0
      AND f.price_60d_ago > 0
),
target_dates AS (
    SELECT
        c.*,
        (
            SELECT MIN(b.date)
            FROM benchmark b
            WHERE b.date >= c.entry_date + INTERVAL 90 DAY
        ) AS exit_date
    FROM candidates c
)
SELECT
    t.symbol,
    t.feature_date,
    t.adj_close,
    t.return_20d,
    t.return_60d,
    t.volatility_20d,
    t.entry_date,
    t.exit_date,
    exit_price.adj_close / entry_price.adj_close - 1 AS target_90d
FROM target_dates t
JOIN prices entry_price
    ON entry_price.symbol = t.symbol
   AND entry_price.date = t.entry_date
JOIN prices exit_price
    ON exit_price.symbol = t.symbol
   AND exit_price.date = t.exit_date
WHERE entry_price.adj_close > 0
  AND exit_price.adj_close > 0
  AND t.volatility_20d IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM universe_membership u
      WHERE u.symbol = t.symbol
        AND u.effective_start <= t.feature_date
        AND t.feature_date <= u.quarter_end
        AND u.selection_date <= t.feature_date
  )
ORDER BY t.symbol, t.feature_date
"""
if __name__ == "__main__":
    result = db.execute(query)

    print("FEATURES AND 90-DAY TARGET PREVIEW")
    for row in result.fetchmany(10):
        print(row)

    db.close()