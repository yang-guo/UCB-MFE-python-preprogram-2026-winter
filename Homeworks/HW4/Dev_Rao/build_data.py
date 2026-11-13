"""Step 1: build the weekly modeling panel and the daily price table.

    uv run python build_data.py --db class      # ~10 s, wiring check
    uv run python build_data.py --db top2000    # ~10-15 min on a laptop

Outputs (gitignored) in outputs/cache_<db>.duckdb:
  panel_raw  weekly signal rows of dated universe members (raw features + labels)
  panel      ELIGIBLE rows with same-date ranks / sector-relative features
  px_daily   daily adjusted closes for every symbol from 2012-10-01 (backtests)
  bench      benchmark adjusted close
plus audit CSVs (feature coverage by year, implausible capitalization inputs).
"""

import argparse
import time
import warnings

import duckdb
import numpy as np
import pandas as pd

from hw4 import config, data, features as F

BATCH = 250
KEEP = (["date", "symbol", "sector", "universe_rank", "selection_date", "membership_quarter",
         "is_member", "eligible", "n_history", "dv_med_63", "close", "adj_close",
         "fiscal_date", "available_date", "entry_date", "entry_price", "label_end",
         "exit_price", "target"] + F.RAW_FEATURES + F.REGIME_FEATURES)


def build(db_key):
    t0 = time.time()
    database = config.DATABASES[db_key]
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    cache_path = config.OUTPUT_DIR / f"cache_{db_key}.duckdb"
    cache_path.unlink(missing_ok=True)
    cache = duckdb.connect(str(cache_path))

    calendar, bench = data.load_calendar_and_benchmark(database)
    end = min(pd.Timestamp(config.TEST_END), calendar[-1])
    mkt_ret, regime = F.benchmark_daily(bench, calendar)
    signals = F.weekly_signal_dates(calendar, config.TRAIN_START, end)
    membership = data.load_membership(database)
    sectors = data.load_sectors(database)
    fund = data.load_fundamentals(database)
    events = F.fundamental_events(fund)
    print(f"[{time.time()-t0:5.0f}s] {len(signals)} weekly signals, {len(events):,} filing events "
          f"({events.attrs['dropped_nonpositive_lag']:,} nonpositive-lag rows dropped)")

    cache.register("bench_df", bench.rename("adj_close").reset_index())
    cache.execute("CREATE TABLE bench AS SELECT * FROM bench_df")
    symbols = data.list_symbols(database)
    created = False
    n_quarantine = {"warnings": []}
    for b in range(0, len(symbols), BATCH):
        batch = symbols[b:b + BATCH]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prices = data.load_prices(database, batch, calendar)
        n_quarantine["warnings"] += [str(x.message) for x in w]
        frames = [F.price_features_for_symbol(g, calendar, mkt_ret, signals)
                  for _, g in prices.groupby("symbol", sort=True)]
        panel = pd.concat(frames, ignore_index=True)
        panel = F.apply_membership(panel, membership, calendar)
        panel = panel.loc[panel.is_member]      # only members are ever trained on / scored
        panel = F.join_fundamentals(panel, events.loc[events.symbol.isin(batch)])
        panel = panel.join(regime, on="date")
        panel["sector"] = panel.symbol.map(sectors).fillna("Unknown")
        panel = panel[KEEP]
        num = [c for c in F.RAW_FEATURES + F.REGIME_FEATURES + ["dv_med_63", "n_history"]]
        panel[num] = panel[num].astype("float32")
        daily = prices.loc[prices.date >= "2012-10-01", ["symbol", "date", "adj_close"]]
        cache.register("p_df", panel)
        cache.register("d_df", daily)
        if not created:
            cache.execute("CREATE TABLE panel_raw AS SELECT * FROM p_df")
            cache.execute("CREATE TABLE px_daily AS SELECT * FROM d_df")
            created = True
        else:
            cache.execute("INSERT INTO panel_raw SELECT * FROM p_df")
            cache.execute("INSERT INTO px_daily SELECT * FROM d_df")
        cache.unregister("p_df"); cache.unregister("d_df")
        print(f"[{time.time()-t0:5.0f}s] symbols {b + len(batch):>5}/{len(symbols)}  "
              f"member rows this batch {len(panel):,}")
    for msg in n_quarantine["warnings"]:
        print("  quarantine:", msg)

    # ---- same-date cross-sectional features, one year at a time (bounded RAM) ----
    years = range(signals[0].year, signals[-1].year + 1)
    created = False
    for y in years:
        p = cache.execute("SELECT * FROM panel_raw WHERE eligible AND year(date)=?", [y]).fetchdf()
        if p.empty:
            continue
        p = F.cross_sectional(p)
        cache.register("x_df", p)
        cache.execute(("CREATE TABLE panel AS " if not created else "INSERT INTO panel ") + "SELECT * FROM x_df")
        cache.unregister("x_df")
        created = True
    cache.execute("CREATE INDEX IF NOT EXISTS ix_px ON px_daily(date)")
    audits(cache, db_key)
    print(f"[{time.time()-t0:5.0f}s] done -> {cache_path}")
    cache.close()


def audits(cache, db_key):
    """Print + save coverage by year and implausible capitalization inputs."""
    feats = F.RAW_FEATURES
    cov = cache.execute("SELECT year(date) AS year, count(*) AS rows, count(DISTINCT symbol) AS symbols, "
                        "avg(target IS NOT NULL::INT) AS label_cov, "
                        + ", ".join(f"avg(({c} IS NOT NULL)::INT) AS {c}" for c in feats)
                        + " FROM panel GROUP BY 1 ORDER BY 1").fetchdf().set_index("year")
    cov.round(3).to_csv(config.OUTPUT_DIR / f"audit_feature_coverage_{db_key}.csv")
    counts = cache.execute("""SELECT year(date) y, count(*) members, sum(eligible::INT) eligible,
        count(DISTINCT date) dates FROM panel_raw GROUP BY 1 ORDER BY 1""").fetchdf()
    counts.to_csv(config.OUTPUT_DIR / f"audit_eligibility_{db_key}.csv", index=False)
    print(counts.to_string(index=False))
    print("feature coverage by year (eligible rows):")
    print((cov[["rows", "label_cov"] + feats].T.iloc[:, ::2]).round(2).to_string())
    # implausible inputs: split-adjusted close > $100k or cap proxy > $5T
    bad = cache.execute("""SELECT symbol, min(date) first_date, max(date) last_date, count(*) n_rows,
        max(close) max_close, max(exp(log_mcap))/1e9 AS max_cap_bn, bool_or(eligible) ever_eligible
        FROM panel_raw WHERE close > 1e5 OR log_mcap > ln(5e12) GROUP BY 1 ORDER BY max_cap_bn DESC""").fetchdf()
    bad.to_csv(config.OUTPUT_DIR / f"audit_implausible_caps_{db_key}.csv", index=False)
    print(f"implausible capitalization inputs: {len(bad)} symbols")
    print(bad.head(15).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=list(config.DATABASES), default="class")
    build(ap.parse_args().db)
