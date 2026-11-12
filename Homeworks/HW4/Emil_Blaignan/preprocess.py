"""Builds the research panel with the Lecture 7 and 8 helpers in Lectures/modeling/course_data.py.

    python Homeworks/HW4/Emil_Blaignan/preprocess.py

Steps, in the same order as course_data.prepare_panel. I run it in batches of 250 tickers so the
15.8 million row price table never sits in memory at once.
  1. load_data gets prices, fundamentals, the universe and the benchmark. Bad rows are set aside.
  1b. clean_prices drops impossible reverting prices. They become gaps, not filled values.
  2. price_features builds momentum, volatility and drawdown on the market calendar.
  3. join_financials adds fundamentals from the day after the filing date.
  4. apply_membership uses the quarterly top 2000 list for the next session's entry.
  5. make_labels builds the 90 day target from the next session's close. Targets are never features.
It also adds a tradable flag (eligible and close of at least $2) and a target_rank column.
Output goes to outputs/panel.parquet and outputs/benchmark.parquet, which git ignores.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Lectures" / "modeling"))
import course_data as cd  # noqa: E402  (course helpers, cited per assignment)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_prices import clean_prices  # noqa: E402  (causal bad-print screen, step 1b)

DB = ROOT / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
OUT = Path(__file__).resolve().parent / "outputs"
END = "2026-09-30"
BATCH = 250
MIN_PRICE = 2.0   # stocks under $2 are not tradable. This also removes unadjusted reverse split artifacts

KEEP = ["symbol", "date", "close", "adj_close", "volume", "ret_1d",
        "momentum_21", "momentum_63", "momentum_252_21", "volatility_63", "drawdown_252",
        "net_margin", "operating_roa", "debt_to_assets", "cash_to_assets", "fcf_to_assets",
        "accruals_to_assets", "filing_age_days", "sector", "eligible", "is_universe_member", "universe_rank",
        "entry_date", "entry_price", "label_end", "exit_price", "tradable", "target"]


def main():
    OUT.mkdir(exist_ok=True)
    import duckdb
    with duckdb.connect(str(DB), read_only=True) as con:
        symbols = [r[0] for r in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    membership = cd.load_membership(DB, end=END)
    events_cache = None
    parts, quarantine, flagged = [], [], []
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prices, fundamentals, universe, benchmark = cd.load_data(DB, end=END, symbols=batch)
        quarantine += [str(x.message) for x in w]
        calendar = benchmark.date
        prices, bad = clean_prices(prices)
        flagged.append(bad)
        panel = cd.price_features(prices, calendar)
        panel["ret_1d"] = panel.groupby("symbol").adj_close.pct_change(fill_method=None)
        # NOTE: pct_change above is on observed rows; gaps are therefore multi-day returns.
        # Recompute on the calendar so a gap is a missing return, not a long return.
        pos = pd.Series(np.arange(len(calendar)), index=calendar)
        panel["_pos"] = panel.date.map(pos)
        prev_pos = panel.groupby("symbol")._pos.shift()
        panel.loc[panel._pos - prev_pos != 1, "ret_1d"] = np.nan
        panel = cd.join_financials(panel, cd.financial_events(fundamentals), universe)
        panel = cd.apply_membership(panel, membership, calendar)
        panel = cd.make_labels(panel, prices, calendar)
        # carry raw close for incident checks (price_features keeps it in p already)
        panel["tradable"] = panel.eligible & panel.close.ge(MIN_PRICE)
        parts.append(panel[KEEP])
        print(f"batch {i // BATCH + 1}/{-(-len(symbols) // BATCH)}  rows={len(panel):,}", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    # Cross-sectional percentile of the 90d target among that day's tradable names.
    # Needs no fitted statistics, so it is a safe robust training target.
    panel["target_rank"] = panel.target.where(panel.tradable).groupby(panel.date).rank(pct=True)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    pd.concat(flagged, ignore_index=True).to_csv(OUT / "flagged_prices.csv", index=False)
    cd_bench = cd.load_data(DB, end=END, symbols=symbols[:1])[3]
    cd_bench.to_parquet(OUT / "benchmark.parquet", index=False)
    pd.Series(sorted(set(quarantine))).to_csv(OUT / "quarantine_warnings.csv", index=False, header=["warning"])
    print("saved", len(panel), "rows;", "warnings:", sorted(set(quarantine)))


if __name__ == "__main__":
    main()
