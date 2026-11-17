import duckdb
from pathlib import Path

from exclusions import excluded_symbols_sql

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)

FEATURE_PANEL = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/combined_feature_panel.parquet"
)

OUTPUT = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/training_labels.parquet"
)

# Development feature dates only; the Jan-Mar 2016 buffer never supplies
# feature dates. Exits are not capped: a label may mature after the test
# period starts, and model fitting must keep only exit_date < fit_date.
FEATURE_START = "2010-01-01"
FEATURE_END = "2015-12-31"

HOLDING_DAYS = 90

# An exit may fill at most this many market sessions after the scheduled exit
# session (first session on/after maturity); otherwise the label is censored.
MAX_EXIT_DELAY_SESSIONS = 5

# Single-day adjusted-price ratios outside [1/4, 4] are flagged for review
# (possible unadjusted splits or bad prints), never deleted or altered.
EXTREME_MOVE_RATIO = 4.0

con = duckdb.connect(str(DB), read_only=True)

query = f"""
WITH sessions AS (
    SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS session_number
    FROM calendar
),
quotes AS (
    -- Positive adjusted closes on market sessions. Zero-volume rows are kept
    -- here so the extreme-move check sees every recorded price.
    SELECT
        p.symbol,
        p.date,
        p.adj_close,
        p.volume,
        p.adj_close / LAG(p.adj_close) OVER (
            PARTITION BY p.symbol ORDER BY p.date
        ) AS daily_ratio
    FROM prices p
    SEMI JOIN calendar c ON p.date = c.date
    WHERE p.adj_close > 0
),
fills AS (
    -- Executable prices: positive adjusted close AND positive volume.
    SELECT q.symbol, q.date, q.adj_close, s.session_number
    FROM quotes q
    JOIN sessions s ON q.date = s.date
    WHERE q.volume > 0
),
feature_dates AS (
    SELECT DISTINCT symbol, date AS feature_date
    FROM read_parquet('{FEATURE_PANEL}')
    WHERE date BETWEEN DATE '{FEATURE_START}' AND DATE '{FEATURE_END}'
      AND symbol NOT IN {excluded_symbols_sql()}
),
entries AS (
    -- Entry is the next market session's close; it is never delayed.
    SELECT
        f.symbol,
        f.feature_date,
        nxt.date AS entry_date,
        nxt.session_number - fs.session_number - 1 AS entry_session_gap,
        e.adj_close AS entry_price,
        e.symbol IS NOT NULL AS has_entry_fill,
        nxt.date + {HOLDING_DAYS} AS maturity_date
    FROM feature_dates f
    JOIN sessions fs
        ON f.feature_date = fs.date
    LEFT JOIN sessions nxt
        ON nxt.session_number = fs.session_number + 1
    LEFT JOIN fills e
        ON e.symbol = f.symbol
       AND e.date = nxt.date
),
scheduled AS (
    -- Scheduled exit: maturity, or the next market session if closed.
    SELECT n.*, sx.date AS scheduled_exit_date, sx.session_number AS scheduled_sn
    FROM entries n
    ASOF LEFT JOIN sessions sx
        ON sx.date >= n.maturity_date
),
candidate_exits AS (
    -- First executable price on/after the scheduled exit session.
    SELECT
        s.*,
        x.date AS candidate_exit_date,
        x.adj_close AS candidate_exit_price,
        x.session_number - s.scheduled_sn AS candidate_delay
    FROM scheduled s
    ASOF LEFT JOIN fills x
        ON x.symbol = s.symbol
       AND x.date >= s.scheduled_exit_date
),
labels AS (
    SELECT
        *,
        CASE
            WHEN entry_date IS NULL THEN 'no_next_session'
            WHEN NOT has_entry_fill THEN 'no_valid_entry_fill'
            WHEN scheduled_exit_date IS NULL THEN 'maturity_after_data_end'
            WHEN candidate_exit_date IS NULL THEN 'no_exit_fill_after_maturity'
            WHEN candidate_delay > {MAX_EXIT_DELAY_SESSIONS}
                THEN 'exit_delay_exceeds_{MAX_EXIT_DELAY_SESSIONS}_sessions'
        END AS censor_reason
    FROM candidate_exits
),
finished AS (
    SELECT
        symbol,
        feature_date,
        entry_date,
        maturity_date,
        scheduled_exit_date,
        CASE WHEN censor_reason IS NULL THEN candidate_exit_date END
            AS actual_exit_date,
        CASE WHEN has_entry_fill THEN entry_price END AS entry_price,
        CASE WHEN censor_reason IS NULL THEN candidate_exit_price END
            AS exit_price,
        CASE WHEN has_entry_fill THEN entry_session_gap END
            AS entry_delay_sessions,
        CASE WHEN censor_reason IS NULL THEN candidate_delay END
            AS exit_delay_sessions,
        censor_reason IS NOT NULL AS censored,
        censor_reason
    FROM labels
),
extreme_moves AS (
    SELECT symbol, date
    FROM quotes
    WHERE daily_ratio > {EXTREME_MOVE_RATIO}
       OR daily_ratio < 1 / {EXTREME_MOVE_RATIO}
),
flagged AS (
    -- Flag labels whose holding window (entry day through actual exit, or
    -- through the scheduled exit when censored) contains an extreme move.
    -- The entry day is included so a bad entry print is also caught.
    SELECT DISTINCT f.symbol, f.feature_date
    FROM finished f
    JOIN extreme_moves m
        ON m.symbol = f.symbol
       AND m.date >= f.entry_date
       AND m.date <= COALESCE(f.actual_exit_date, f.scheduled_exit_date)
)
SELECT
    f.symbol,
    f.feature_date,
    f.entry_date,
    f.maturity_date,
    f.scheduled_exit_date,
    f.actual_exit_date,
    f.entry_price,
    f.exit_price,
    f.exit_price / f.entry_price - 1 AS forward_return,
    f.entry_delay_sessions,
    f.exit_delay_sessions,
    f.censored,
    f.censor_reason,
    g.symbol IS NOT NULL AS extreme_move_flag
FROM finished f
LEFT JOIN flagged g
    ON g.symbol = f.symbol
   AND g.feature_date = f.feature_date
"""

print("Building 90-day forward-return labels...")

con.execute(f"""
    COPY ({query} ORDER BY feature_date, symbol)
    TO '{OUTPUT}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nLABEL SUMMARY")
print("=" * 70)

print(con.execute(f"""
    SELECT
        COUNT(*) AS observations,
        COUNT(*) FILTER (WHERE NOT censored) AS valid_labels,
        COUNT(*) FILTER (WHERE censored) AS censored_labels,
        COUNT(DISTINCT symbol) AS stocks,
        CAST(MIN(feature_date) AS VARCHAR) AS first_feature_date,
        CAST(MAX(feature_date) AS VARCHAR) AS last_feature_date,
        CAST(MIN(entry_date) AS VARCHAR) AS first_entry_date,
        CAST(MAX(entry_date) AS VARCHAR) AS last_entry_date,
        CAST(MIN(actual_exit_date) AS VARCHAR) AS first_exit_date,
        CAST(MAX(actual_exit_date) AS VARCHAR) AS last_exit_date,
        COUNT(*) FILTER (WHERE extreme_move_flag) AS extreme_move_flags,
        COUNT(*) FILTER (WHERE extreme_move_flag AND NOT censored)
            AS extreme_move_flags_valid
    FROM read_parquet('{OUTPUT}')
""").df().T.to_string(header=False))

print("\nCENSORING REASONS")
print("=" * 70)

print(con.execute(f"""
    SELECT censor_reason, COUNT(*) AS labels, COUNT(DISTINCT symbol) AS stocks
    FROM read_parquet('{OUTPUT}')
    WHERE censored
    GROUP BY 1
    ORDER BY 2 DESC
""").df().to_string(index=False))

print("\nEXIT DELAY (valid labels)")
print("=" * 70)

print(con.execute(f"""
    SELECT exit_delay_sessions, COUNT(*) AS labels
    FROM read_parquet('{OUTPUT}')
    WHERE NOT censored
    GROUP BY 1
    ORDER BY 1
""").df().to_string(index=False))

print("\nVALIDATION CHECKS (all should be 0)")
print("=" * 70)

print(con.execute(f"""
    WITH l AS (SELECT * FROM read_parquet('{OUTPUT}'))
    SELECT
        COUNT(*) FILTER (WHERE entry_date <= feature_date)
            AS entry_not_after_feature,
        COUNT(*) FILTER (WHERE actual_exit_date < maturity_date)
            AS exit_before_maturity,
        COUNT(*) FILTER (WHERE exit_delay_sessions > {MAX_EXIT_DELAY_SESSIONS})
            AS exit_delay_over_limit,
        COUNT(*) FILTER (WHERE entry_delay_sessions <> 0)
            AS delayed_entries,
        (SELECT COUNT(*) FROM l JOIN prices p
            ON p.symbol = l.symbol AND p.date = l.entry_date
         WHERE l.entry_price IS NOT NULL AND COALESCE(p.volume, 0) <= 0)
            AS zero_volume_entry_fills,
        (SELECT COUNT(*) FROM l JOIN prices p
            ON p.symbol = l.symbol AND p.date = l.actual_exit_date
         WHERE COALESCE(p.volume, 0) <= 0)
            AS zero_volume_exit_fills,
        COUNT(*) FILTER (WHERE symbol IN {excluded_symbols_sql()})
            AS excluded_symbol_rows,
        COUNT(*) - COUNT(DISTINCT (symbol, feature_date))
            AS duplicate_labels,
        COUNT(*) FILTER (WHERE NOT censored AND NOT isfinite(forward_return))
            AS valid_without_finite_return,
        COUNT(*) FILTER (WHERE censored AND forward_return IS NOT NULL)
            AS censored_with_return,
        COUNT(*) FILTER (WHERE feature_date > DATE '{FEATURE_END}')
            AS feature_dates_after_development
    FROM l
""").df().T.to_string(header=False))

print("\nLARGEST 20 FORWARD RETURNS")
print("=" * 70)

print(con.execute(f"""
    SELECT
        symbol, feature_date, entry_date, actual_exit_date,
        entry_price, exit_price, forward_return,
        exit_delay_sessions, extreme_move_flag
    FROM read_parquet('{OUTPUT}')
    WHERE NOT censored
    ORDER BY forward_return DESC
    LIMIT 20
""").df().to_string(index=False))

con.close()

print(f"\nSaved labels to: {OUTPUT}")
