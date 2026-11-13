"""Read-only data-quality audit of equities_top2000.duckdb (counts only, no performance)."""
import sys
import duckdb
import pandas as pd

pd.set_option("display.width", 160)
db = sys.argv[1]
con = duckdb.connect(db, read_only=True)
q = lambda s: con.execute(s).fetchdf()

print("== 1. Price rows: invalid values, duplicates, off-calendar")
print(q("""SELECT count(*) n_rows,
  sum((adj_close IS NULL)::INT) null_adj, sum((adj_close <= 0)::INT) nonpos_adj,
  sum((close <= 0 OR close IS NULL)::INT) bad_close,
  sum((volume = 0)::INT) zero_volume, sum((volume IS NULL)::INT) null_volume
  FROM prices""").T.to_string())
print("duplicate (symbol,date):", q("SELECT count(*) n FROM (SELECT symbol,date FROM prices GROUP BY 1,2 HAVING count(*)>1)").n[0])
print("off-calendar rows:", q("SELECT count(*) n FROM prices p ANTI JOIN calendar c USING(date)").n[0])

print("\n== 2. Extreme one-day adjusted moves (adjacent quote rows, per symbol)")
print(q("""WITH r AS (SELECT symbol, date, adj_close / lag(adj_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS ret
  FROM prices WHERE adj_close > 0)
  SELECT sum((ret > 1)::INT) up_over_100pct, sum((ret < -0.8)::INT) down_over_80pct,
         sum((abs(ret) > 0.5)::INT) abs_over_50pct, count(DISTINCT CASE WHEN abs(ret) > 1 THEN symbol END) symbols_over_100pct
  FROM r""").T.to_string())

print("\n== 3. Internal quote gaps (calendar sessions between consecutive quotes)")
print(q("""WITH cal AS (SELECT date, row_number() OVER (ORDER BY date) k FROM calendar),
  p AS (SELECT symbol, k FROM prices JOIN cal USING(date)),
  g AS (SELECT symbol, k - lag(k) OVER (PARTITION BY symbol ORDER BY k) - 1 AS missed FROM p)
  SELECT sum((missed BETWEEN 1 AND 5)::INT) gaps_1_5, sum((missed BETWEEN 6 AND 21)::INT) gaps_6_21,
         sum((missed > 21)::INT) gaps_over_21, count(DISTINCT CASE WHEN missed > 5 THEN symbol END) symbols_gap_over_5
  FROM g""").T.to_string())

print("\n== 4. Price history ends before 2026-09-30 (possible delisting, no payout table)")
print(q("""WITH last AS (SELECT symbol, max(date) last_date FROM prices GROUP BY 1)
  SELECT count(*) symbols_ending_early,
    sum((u.is_active_snapshot = false)::INT) of_which_inactive,
    sum((last_date < DATE '2016-04-01')::INT) end_before_test,
    sum((last_date BETWEEN DATE '2016-04-01' AND DATE '2026-09-29')::INT) end_during_test
  FROM last JOIN universe u USING(symbol) WHERE last_date < DATE '2026-09-30'""").T.to_string())

print("\n== 5. Universe members with implausible capitalization proxy (basis/unit errors)")
print(q("""SELECT count(*) member_rows,
  sum((market_cap_proxy > 4e12)::INT) over_4T, sum((market_cap_proxy < 1e7)::INT) under_10M,
  sum((selection_close > 100000)::INT) close_over_100k,
  count(DISTINCT CASE WHEN market_cap_proxy > 4e12 OR selection_close > 100000 THEN symbol END) symbols_flagged
  FROM universe_membership""").T.to_string())
print(q("""SELECT symbol, count(*) n_quarters, round(max(market_cap_proxy)/1e9) max_cap_bn, max(selection_close) max_close
  FROM universe_membership WHERE market_cap_proxy > 4e12 OR selection_close > 100000
  GROUP BY 1 ORDER BY max_cap_bn DESC LIMIT 12""").to_string(index=False))

print("\n== 6. Coverage by period (eligible-universe size)")
print(q("""SELECT CASE WHEN quarter_start < DATE '2010-01-01' THEN '2000-2009'
                    WHEN quarter_start < DATE '2016-04-01' THEN '2010-2016Q1 (dev+buffer)'
                    ELSE '2016Q2-2026Q3 (test)' END period,
  count(*) n_quarters, min(selected_count) min_selected, max(selected_count) max_selected
  FROM universe_rebalances GROUP BY 1 ORDER BY 1""").to_string(index=False))

print("\n== 7. Fundamentals")
print(q("""SELECT count(*) n_rows, sum((period <> 'quarter')::INT) non_quarter,
  sum((filing_date IS NULL OR filing_date <= fiscal_date)::INT) bad_filing,
  sum((filing_date - fiscal_date > 365)::INT) filed_over_1y_late,
  sum((shares_outstanding <= 0 OR shares_outstanding IS NULL)::INT) bad_shares,
  sum((total_assets <= 0 OR total_assets IS NULL)::INT) bad_assets
  FROM fundamentals""").T.to_string())
