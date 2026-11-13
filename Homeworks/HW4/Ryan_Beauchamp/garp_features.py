"""GARP-style features in their own parquet file, keyed on (date, symbol).

Merged onto the stage 1 panel on (date, symbol); the panel is never rebuilt.

Availability rules (same as course_data.financial_events)
* Only quarterly records with filing_date > fiscal_date.
* A quarter is usable from filing_date + 1 calendar day (``available_date``).
* At each signal date the known set is every quarter with available_date <= date
  (backward as-of join within each stock). A late filing of an old quarter can
  fill a hole in the history but never replaces the latest fiscal quarter.
* All features are blanked when the latest known fiscal period is more than
  550 days old at the signal date.

Definitions (TTM = sum of the latest four consecutive known fiscal quarters;
consecutive = successive fiscal dates 80-100 days apart, else missing;
prior TTM = the four consecutive quarters before those, i.e. 8 in a row)
* EV = close * shares_outstanding + total_debt - cash_and_equivalents, with the
  latest known quarter's shares/debt/cash and the signal-date RAW close (close,
  shares and EPS are on the current split basis in this database; adj_close
  differs from close by dividends for most stocks, see writeup notes).
  Missing unless close > 0, shares > 0 and EV > 0 (missing debt or cash -> missing).
* fcf_yield = FCF_TTM / EV;  ebitda_yield = EBITDA_TTM / EV
* earnings_yield = EPS_TTM / close
* rev_growth_ttm = Rev_TTM / Rev_TTM_prior - 1 (both > 0)
* eps_change_yield = (EPS_TTM - EPS_TTM_prior) / close
* op_margin_ttm = OperatingIncome_TTM / Rev_TTM (Rev_TTM > 0)
* sue_q = UE_q / std(UE_q, UE_q-1, ..., UE_q-7), UE_k = EPS_k - EPS_k-4: the 8
  UEs ending at the latest quarter, which need exactly 12 consecutive quarters
  of EPS (sample std, ddof=1). Missing if fewer than 12 consecutive quarters,
  any EPS missing, or std <= 0. No drift term.
* garp_composite = mean of same-date cross-sectional percentile ranks, among
  eligible clean rows, of fcf_yield, ebitda_yield, earnings_yield,
  rev_growth_ttm, eps_change_yield; missing if fewer than 3 of the 5 are valid.
  Computed after all stocks are combined.
No winsorizing here (the model pipeline clips per fold on training rows).
EPS uses the vendor ``eps`` column (``eps_diluted`` is mostly missing).

Run from the repository root:
    development (data through 2016-03-31): .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/garp_features.py
    full period (data preparation only):   .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/garp_features.py --full

Credits: availability rules follow Lectures/modeling/course_data.py
(financial_events/join_financials); code written here. Written with AI
assistance (Claude Code) and reviewed by the author.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import duckdb
import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
CLASS_DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_class.duckdb"
DEV_END, FULL_END = pd.Timestamp("2016-03-31"), pd.Timestamp("2026-09-30")
MAX_LOG_MOVE, MIN_CLOSE = np.log(5), 1.0       # same clean flag as walkforward.py

COMPOSITE_INPUTS = ["fcf_yield", "ebitda_yield", "earnings_yield", "rev_growth_ttm", "eps_change_yield"]
GARP_FEATURES = COMPOSITE_INPUTS + ["op_margin_ttm", "sue_q"]
GARP_SET = GARP_FEATURES + ["garp_composite"]
EVENT_COLUMNS = ["fcf_ttm", "ebitda_ttm", "eps_ttm", "eps_ttm_prior", "rev_ttm", "rev_ttm_prior",
                 "opinc_ttm", "shares", "debt", "cash", "sue_q"]
STALE_DAYS = 550


def peak_gb():
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9


def load_quarters(database, end, symbols=None):
    """Valid quarterly records known by `end` (filing after fiscal end, filed by `end`)."""
    where = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    params = [end.date(), end.date()] + ([list(symbols)] if symbols is not None else [])
    with duckdb.connect(str(database), read_only=True) as con:
        q = con.execute(f"""SELECT symbol, fiscal_date, filing_date, revenue, ebitda, eps, operating_income,
                   free_cash_flow, total_debt, cash_and_equivalents, shares_outstanding
            FROM fundamentals WHERE period = 'quarter' AND fiscal_date <= ? AND filing_date <= ?
            AND filing_date > fiscal_date {where}""", params).fetchdf()
    for column in ["fiscal_date", "filing_date"]:
        q[column] = pd.to_datetime(q[column]).astype("datetime64[ns]")
    q["available_date"] = q.filing_date + pd.Timedelta(days=1)
    return q


def chain_length(fiscal_desc):
    """Number of consecutive quarters (80-100 day gaps) starting from the latest."""
    if len(fiscal_desc) == 0:
        return 0
    gaps = -np.diff(fiscal_desc).astype("timedelta64[D]").astype(int)
    ok = (gaps >= 80) & (gaps <= 100)
    bad = np.flatnonzero(~ok)
    return len(fiscal_desc) if not len(bad) else int(bad[0]) + 1


def stock_events(q):
    """One row per availability date with the price-free inputs known on that date."""
    q = q.sort_values("fiscal_date")
    fiscal = q.fiscal_date.to_numpy()
    avail = q.available_date.to_numpy()
    values = {c: q[c].to_numpy(float) for c in ["revenue", "ebitda", "eps", "operating_income",
                                                 "free_cash_flow", "total_debt", "cash_and_equivalents",
                                                 "shares_outstanding"]}
    rows = []
    for a in np.unique(avail):
        known = np.flatnonzero(avail <= a)[::-1]           # known quarters, latest fiscal first
        assert (avail[known] <= a).all()
        n = chain_length(fiscal[known])
        idx = known[:min(n, 12)]

        def ttm(c, lo=0):
            x = values[c][idx[lo:lo + 4]]
            return float(x.sum()) if len(x) == 4 and np.isfinite(x).all() else np.nan

        latest = known[0]
        row = {"available_date": a, "latest_fiscal": fiscal[latest],
               "fcf_ttm": ttm("free_cash_flow"), "ebitda_ttm": ttm("ebitda"), "eps_ttm": ttm("eps"),
               "eps_ttm_prior": ttm("eps", 4), "rev_ttm": ttm("revenue"), "rev_ttm_prior": ttm("revenue", 4),
               "opinc_ttm": ttm("operating_income"), "shares": values["shares_outstanding"][latest],
               "debt": values["total_debt"][latest], "cash": values["cash_and_equivalents"][latest],
               "sue_q": np.nan}
        if n >= 12:
            eps = values["eps"][idx]                         # 12 consecutive, latest first
            if np.isfinite(eps).all():
                ue = eps[0:8] - eps[4:12]                    # UE_q ... UE_q-7
                sd = ue.std(ddof=1)
                if sd > 0:
                    row["sue_q"] = ue[0] / sd
        rows.append(row)
    return pd.DataFrame(rows)


def compute_features(panel, quarters):
    """panel: date, symbol, close. Returns panel keys + GARP features (no composite) + audit columns."""
    events = [stock_events(g).assign(symbol=s) for s, g in quarters.groupby("symbol", sort=True)]
    events = pd.concat(events, ignore_index=True).sort_values("available_date")
    events["available_date"] = events.available_date.astype("datetime64[ns]")
    left = panel[["date", "symbol", "close"]].sort_values("date")
    f = pd.merge_asof(left, events, left_on="date", right_on="available_date", by="symbol", direction="backward")
    assert (f.available_date.isna() | f.available_date.le(f.date)).all(), "a quarter was used before filing + 1 day"
    stale = (f.date - f.latest_fiscal).dt.days.gt(STALE_DAYS)
    f.loc[stale, EVENT_COLUMNS] = np.nan

    close = f.close.where(f.close.gt(0))
    shares = f.shares.where(f.shares.gt(0))
    ev = close * shares + f.debt - f.cash
    ev = ev.where(ev.gt(0))
    rev, rev_prior = f.rev_ttm.where(f.rev_ttm.gt(0)), f.rev_ttm_prior.where(f.rev_ttm_prior.gt(0))
    out = f[["date", "symbol"]].copy()
    out["ev"] = ev
    out["fcf_yield"] = f.fcf_ttm / ev
    out["ebitda_yield"] = f.ebitda_ttm / ev
    out["earnings_yield"] = f.eps_ttm / close
    out["rev_growth_ttm"] = rev / rev_prior - 1
    out["eps_change_yield"] = (f.eps_ttm - f.eps_ttm_prior) / close
    out["op_margin_ttm"] = f.opinc_ttm / rev
    out["sue_q"] = f.sue_q
    out["garp_available_date"] = f.available_date.where(~stale)
    out["garp_latest_fiscal"] = f.latest_fiscal.where(~stale)
    out[GARP_FEATURES] = out[GARP_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def add_composite(features, eligible_clean):
    """Same-date percentile ranks among eligible clean rows; mean of >= 3 valid inputs."""
    f = features.merge(eligible_clean, on=["date", "symbol"], how="left", validate="one_to_one")
    pool = f.in_pool.eq(True)
    ranks = f.loc[pool, ["date"] + COMPOSITE_INPUTS].groupby("date")[COMPOSITE_INPUTS].rank(pct=True)
    valid = ranks.notna().sum(axis=1)
    f["garp_composite"] = np.nan
    f.loc[ranks.index, "garp_composite"] = ranks.mean(axis=1).where(valid >= 3)
    return f.drop(columns="in_pool")


def eligible_clean_rows(panel_dir, measures_path, end):
    p = pd.read_parquet(panel_dir, columns=["date", "symbol", "eligible"], filters=[("date", "<=", end)])
    m = pd.read_parquet(measures_path, columns=["date", "symbol", "close", "max_abs_log_move_252"],
                        filters=[("date", "<=", end)])
    p = p.merge(m, on=["date", "symbol"], how="left", validate="one_to_one")
    clean = ~(p.max_abs_log_move_252.gt(MAX_LOG_MOVE) | p.close.lt(MIN_CLOSE))
    return p.loc[p.eligible & clean, ["date", "symbol"]].assign(in_pool=True)


def build(panel_dir, measures_path, end, database=DATABASE):
    panel = pd.read_parquet(panel_dir, columns=["date", "symbol", "close"], filters=[("date", "<=", end)])
    quarters = load_quarters(database, end)
    features = compute_features(panel, quarters)
    return add_composite(features, eligible_clean_rows(panel_dir, measures_path, end)), quarters


# --------------------------------------------------------------------------- checks
def check_aapl(features, quarters, date="2014-06-30"):
    """Recompute AAPL's features on `date` by hand from the raw quarterly rows."""
    date = pd.Timestamp(date)
    q = quarters.loc[quarters.symbol.eq("AAPL") & quarters.available_date.le(date)].sort_values("fiscal_date",
                                                                                                 ascending=False)
    q12 = q.head(12)
    row = features.loc[features.symbol.eq("AAPL") & features.date.eq(date)].iloc[0]
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        close = con.execute("SELECT close FROM prices WHERE symbol='AAPL' AND date=?", [date.date()]).fetchone()[0]
    t4, p4 = q12.iloc[:4], q12.iloc[4:8]
    latest = q12.iloc[0]
    ev = close * latest.shares_outstanding + latest.total_debt - latest.cash_and_equivalents
    eps = q12.eps.to_numpy()
    ue = eps[0:8] - eps[4:12]
    hand = {"fcf_yield": t4.free_cash_flow.sum() / ev, "ebitda_yield": t4.ebitda.sum() / ev,
            "earnings_yield": t4.eps.sum() / close, "rev_growth_ttm": t4.revenue.sum() / p4.revenue.sum() - 1,
            "eps_change_yield": (t4.eps.sum() - p4.eps.sum()) / close,
            "op_margin_ttm": t4.operating_income.sum() / t4.revenue.sum(), "sue_q": ue[0] / ue.std(ddof=1)}
    print(f"\nCheck 1: AAPL on {date.date()} (close {close:.2f}; quarters known by then, latest first)")
    print(q12[["fiscal_date", "filing_date", "available_date", "revenue", "ebitda", "eps", "operating_income",
               "free_cash_flow"]].to_string(index=False))
    print(f"  latest quarter {latest.fiscal_date.date()}: shares {latest.shares_outstanding:,.0f}, "
          f"debt {latest.total_debt:,.0f}, cash {latest.cash_and_equivalents:,.0f}")
    print(f"  EV = {close:.2f} x {latest.shares_outstanding:,.0f} + {latest.total_debt:,.0f} - "
          f"{latest.cash_and_equivalents:,.0f} = {ev:,.0f}   (module: {row.ev:,.0f})")
    print(f"  TTM: FCF {t4.free_cash_flow.sum():,.0f}  EBITDA {t4.ebitda.sum():,.0f}  EPS {t4.eps.sum():.2f}  "
          f"Rev {t4.revenue.sum():,.0f}  OpInc {t4.operating_income.sum():,.0f}")
    print(f"  prior TTM: EPS {p4.eps.sum():.2f}  Rev {p4.revenue.sum():,.0f};  UE_q..UE_q-7 = {np.round(ue, 2)}")
    ok = True
    for name, value in hand.items():
        match = np.isclose(value, row[name], rtol=1e-12, atol=0)
        ok &= bool(match)
        print(f"  {name:<17} hand {value:+.6f}   module {row[name]:+.6f}   {'OK' if match else 'MISMATCH'}")
    return ok


def check_class_ev():
    """Computed EV vs vendor enterprise_value on the class database (close on/before fiscal date)."""
    with duckdb.connect(str(CLASS_DATABASE), read_only=True) as con:
        f = con.execute("""SELECT f.symbol, f.fiscal_date, f.shares_outstanding, f.total_debt,
                   f.cash_and_equivalents, f.enterprise_value,
                   (SELECT p.close FROM prices p WHERE p.symbol=f.symbol AND p.date<=f.fiscal_date
                    ORDER BY p.date DESC LIMIT 1) AS close
            FROM fundamentals f WHERE f.period='quarter' AND f.enterprise_value IS NOT NULL""").fetchdf()
    f["ev"] = f.close * f.shares_outstanding + f.total_debt - f.cash_and_equivalents
    r = (f.ev / f.enterprise_value).dropna()
    print(f"\nCheck 2: class database, computed EV / vendor enterprise_value on {len(r)} quarters: "
          f"mean {r.mean():.4f}, median {r.median():.4f}, min {r.min():.4f}, max {r.max():.4f}")
    return r


def check_prefix(full_features, panel_dir, measures_path, cutoff, database=DATABASE):
    cutoff = pd.Timestamp(cutoff)
    truncated, _ = build(panel_dir, measures_path, cutoff, database)
    earlier = full_features.loc[full_features.date.le(cutoff)]
    both = truncated.merge(earlier, on=["date", "symbol"], suffixes=("_cut", "_full"), validate="one_to_one")
    assert len(both) == len(truncated) == len(earlier)
    bad = {}
    for c in GARP_SET + ["ev"]:
        a, b = both[f"{c}_cut"], both[f"{c}_full"]
        bad[c] = int((~((a.isna() & b.isna()) | np.isclose(a, b, rtol=1e-12, atol=0))).sum())
    print(f"\nCheck 3: prefix test at {cutoff.date()} on {len(both):,} rows: differences {bad}")
    assert not any(bad.values()), "a feature changed when later data was added"


def coverage_and_extremes(features, eligible_clean):
    f = features.merge(eligible_clean, on=["date", "symbol"], how="inner")
    year = f.date.dt.year
    coverage = (f[GARP_SET].notna().groupby(year).mean() * 100).round(1)
    limits = {"fcf_yield": (-1, 1), "ebitda_yield": (-1, 1), "earnings_yield": (-1, 1),
              "rev_growth_ttm": (-0.9, 5), "eps_change_yield": (-1, 1), "op_margin_ttm": (-5, 1),
              "sue_q": (-10, 10)}
    extremes = pd.DataFrame({c: {"rows outside": int(((f[c] < lo) | (f[c] > hi)).sum()),
                                 "stocks": int(f.loc[(f[c] < lo) | (f[c] > hi), "symbol"].nunique()),
                                 "bounds": f"[{lo}, {hi}]", "p1": f[c].quantile(0.01), "p50": f[c].median(),
                                 "p99": f[c].quantile(0.99)} for c, (lo, hi) in limits.items()}).T
    print("\nCheck 5a: coverage (% of eligible clean rows with a value) by year")
    print(coverage.to_string())
    print("\nCheck 5b: extreme values (eligible clean rows)")
    print(extremes.to_string())
    return coverage, extremes


def main(full=False):
    start = time.perf_counter()
    end = FULL_END if full else DEV_END
    panel_dir = OUT / ("panel_full" if full else "panel_stage1")
    measures = OUT / ("clean_measures_full" if full else "clean_measures") / "measures.parquet"
    out_dir = OUT / ("garp_full" if full else "garp_dev")
    tag = "garp_full" if full else "garp_dev"

    features, quarters = build(panel_dir, measures, end)
    print(f"Built {len(features):,} rows from {len(quarters):,} valid quarters in "
          f"{time.perf_counter() - start:.0f}s; peak mem {peak_gb():.1f} GB")
    assert not features.duplicated(["date", "symbol"]).any()
    assert features.date.max() <= end

    ok = check_aapl(features, quarters)
    if not full:
        check_class_ev()
    for cutoff in (["2013-06-28"] if not full else ["2021-06-30"]):
        check_prefix(features, panel_dir, measures, cutoff)
    if full:
        # Prefix test at the development cutoff without a rebuild: the dev file was built from data
        # ending 2016-03-31, so every feature must match on earlier dates. The composite can differ only
        # on 2016-03-31 itself (the dev panel had no next session, so no membership that day).
        dev = pd.read_parquet(OUT / "garp_dev" / "garp_features.parquet")
        both = dev.merge(features.loc[features.date.le(DEV_END)], on=["date", "symbol"],
                         suffixes=("_dev", "_full"), validate="one_to_one")
        assert len(both) == len(dev)
        diffs = {}
        for c in GARP_SET + ["ev"]:
            a, b = both[f"{c}_dev"], both[f"{c}_full"]
            same = (a.isna() & b.isna()) | np.isclose(a, b, rtol=1e-12, atol=0)
            diffs[c] = int((~same & both.date.lt(DEV_END)).sum())
        print(f"\nCheck 3b: full vs development file on {len(both):,} rows before {DEV_END.date()}: differences {diffs}")
        assert not any(diffs.values())
    used = features.garp_available_date.notna()
    assert features.loc[used, "garp_available_date"].le(features.loc[used, "date"]).all()
    print(f"\nCheck 4: on all {int(used.sum()):,} rows with fundamentals, the quarter used was available "
          f"(filing + 1 day) on or before the signal date  OK")
    pool = eligible_clean_rows(panel_dir, measures, end)
    coverage, extremes = coverage_and_extremes(features, pool)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    features.to_parquet(out_dir / "garp_features.parquet", index=False)
    coverage.to_csv(OUT / f"{tag}_coverage.csv")
    extremes.to_csv(OUT / f"{tag}_extremes.csv")
    print(f"\nSaved {out_dir / 'garp_features.parquet'}; total {time.perf_counter() - start:.0f}s; "
          f"peak mem {peak_gb():.1f} GB")
    if not ok:
        raise SystemExit("AAPL hand check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    main(full=parser.parse_args().full)
